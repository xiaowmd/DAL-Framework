#本脚本用于生成与样本量相关的实验结果。
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from DLModels import (
    GBN,
    GBAN,

)
import torch.distributed as dist
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
torch.cuda.empty_cache()

def init_process(rank, world_size):
    """
    初始化分布式训练
    Args:
        rank (int): 当前进程排名
        world_size (int): 总进程数
    """
    dist.init_process_group(
        backend='nccl',  # 使用NCCL后端
        init_method='env://',  # 使用环境变量初始化
        world_size=world_size,  # 总进程数
        rank=rank  # 当前进程排名
    )

def get_free_gpus(threshold=0.1):
    """
    获取空闲的GPU
    """
    free_gpus = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            memory = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_free = memory_total - memory
            if memory_free / memory_total > threshold:
                free_gpus.append(i)
    return free_gpus

# 定义数据集
class TextEmbeddingDataset(Dataset):
    def __init__(self, embedding_folder, label_file, max_files=None):
        """
        加载 BERT 嵌入和对应标签
        Args:
            label_file (str): 包含每个文档对应的标签的文件路径
            max_files (int): 最大文件数，用于测试
            label_file (str): 包含每个文档对应的标签的文件路径
        """
        self.embedding_folder = embedding_folder  # 存储嵌入文件夹路径
        # 过滤出只包含 .npy 文件的文件名
        self.file_names = sorted(
            [
                f
                for f in os.listdir(embedding_folder)
                if f.endswith(".npy")
                and os.path.isfile(os.path.join(embedding_folder, f))
            ]
        )

        # 加载标签
        with open(label_file, "r") as f:
            self.labels = [int(line.strip()) for line in f]

            # 如果指定了最大文件数量，截取文件名列表
        if max_files is not None:
            # self.file_names = self.file_names[:max_files]
            # self.labels = self.labels[:max_files]
            indices = random.sample(range(len(self.file_names)), max_files)
            self.file_names = [self.file_names[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]  # 获取当前索引对应的文件名
        file_path = os.path.join(self.embedding_folder, file_name)

        # 按需加载嵌入
        try:
            embedding = np.load(file_path)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            raise e  # 重新抛出异常

        label = self.labels[idx]

        # 填充或截断嵌入（假设 max_length 已定义）
        # ... 填充或截断的代码 ...

        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


def collate_fn(batch):
    """
    自定义 Collate 函数，对文档句子数量进行动态填充。
    Args:
        batch (list of tuples): 每个元素是 (sentence_embeddings, label)。
    Returns:
        padded_embeddings (Tensor): 填充后的嵌入 (batch_size, max_sentences, embedding_dim)。
        labels (Tensor): 标签 (batch_size)。
    """
    # 找到批次中句子数量的最大值
    max_sentences = max(embeddings.shape[0] for embeddings, _ in batch)

    # 动态填充句子嵌入
    padded_embeddings = []
    labels = []

    for embeddings, label in batch:
        num_sentences, embedding_dim = embeddings.shape[0], embeddings.shape[1]
        # 创建填充值（全零张量）
        padding = torch.zeros((max_sentences - num_sentences, embeddings.shape[1]))
        # 填充嵌入
        padded_embeddings.append(torch.cat([embeddings, padding], dim=0))
        labels.append(label)

    # 转换为批次张量
    padded_embeddings = torch.stack(
        padded_embeddings
    )  # (batch_size, max_sentences, embedding_dim)
    labels = torch.tensor(labels)  # (batch_size)

    return padded_embeddings, labels

def validate_model(model, val_loader, criterion, device, cover=1.0):
    """
    验证模型
    Args:
        model (nn.Module): 待验证模型
        val_loader (DataLoader): 验证数据
        criterion: 损失函数
        device: 使用的设备（CPU 或 GPU）

    Returns:
        val_loss (float): 验证损失
        val_acc (float): 验证准确率
    """
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    val_probs = []

    with torch.no_grad():
        for embeddings, labels in tqdm(val_loader, desc="Validating"):
            embeddings, labels = embeddings.to(device), labels.to(device)

            # 前向传播
            outputs, weights = model(embeddings)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 记录预测结果和概率
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_probs.extend(torch.exp(outputs).cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_report = classification_report(
        val_labels, val_preds, output_dict=True
    )  # 使用 output_dict=True 以获取字典形式的报告

    # 计算分类概率并筛选样本
    if cover < 1.0:
        val_probs = np.array(val_probs)
        max_probs = np.max(val_probs, axis=1)
        threshold = np.percentile(max_probs, (1 - cover) * 100)
        mask = max_probs >= threshold

        # 计算筛选后的样本的性能指标
        cover_labels = np.array(val_labels)[mask]
        cover_preds = np.array(val_preds)[mask]
        cover_acc = accuracy_score(cover_labels, cover_preds)
        cover_report = classification_report(
            cover_labels, cover_preds, output_dict=True
        )

        val_report["cover_metrics"] = {
            "accuracy": cover_acc,
            "f1-score": cover_report["macro avg"]["f1-score"],
            "precision": cover_report["macro avg"]["precision"],
            "recall": cover_report["macro avg"]["recall"],
        }
        val_report["cover_threshold"] = threshold
    else:
        val_report["cover_metrics"] = {
            "accuracy": val_acc,
            "f1-score": val_report["macro avg"]["f1-score"],
            "precision": val_report["macro avg"]["precision"],
            "recall": val_report["macro avg"]["recall"],
        }
        val_report["cover_threshold"] = 0.0
    return val_loss, val_acc, val_report

def train_with_cross_validation(
    model_class,
    dataset,
    num_epochs,
    batch_size,
    device,
    model_name,
    label_num,
    k_folds=5,
    attn_method="dot_scaled",
    save_dir="./saved_models",
    save_results=True,
    results_save_path="./DL_results",
    cover=1.0,
    save_model=True,
):
    """
    使用 K 折交叉验证训练模型
    Args:
        model_class (nn.Module): 模型类（如 HierarchicalEncoder 或 HierarchicalEncoderWithAttention）
        dataset (Dataset): 数据集
        num_epochs (int): 每折的训练轮数
        batch_size (int): 批大小
        device (torch.device): 运行设备
        k_folds (int): 折数
        save_dir (str): 模型保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 定义 K 折划分器
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    avg_f1 = 0.0
    avg_precision = 0.0
    avg_recall = 0.0
    avg_accuracy = 0.0
    avg_cover_accuracy = 0.0
    avg_cover_f1 = 0.0
    avg_cover_precision = 0.0
    avg_cover_recall = 0.0
    
    # with torch.profiler.profile(
    # # activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    # # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    # # on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    # # record_shapes=True
    # ) as prof:

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Starting Fold {fold + 1}/{k_folds}")

        # 获取当前折的训练集和验证集
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )  # num_workers根据 CPU 核心数调整（ 4-8）
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        for batch_embeddings, batch_labels in train_loader:
            print("Batch Embeddings Shape:", batch_embeddings.shape)
            print("Batch Labels Shape:", batch_labels.shape)
            break

        # 初始化模型
        model = model_class().to(device)
            # 计算每个类别的权重
        train_labels = np.array(dataset.labels)[train_idx]

        # 动态获取训练集中存在的类别
        unique_train_labels = np.unique(train_labels)

        # 计算存在的类别权重
        if len(unique_train_labels) > 0:
            try:
                class_weights = compute_class_weight(
                    "balanced",
                    classes=unique_train_labels,
                    y=train_labels
                )
            except ValueError as e:
                print(f"Error in compute_class_weight: {e}")
                raise

            # 补全缺失类别的权重（默认1.0）
            full_class_weights = np.ones(2, dtype=np.float32)
            for idx, cls in enumerate(unique_train_labels):
                if cls >= 5:
                    raise ValueError(f"无效类别标签 {cls} (最大允许值: {label_num-1})")
                full_class_weights[cls] = class_weights[idx]
            
            class_weights = torch.FloatTensor(full_class_weights).to(device)
        else:
            # 极端情况：训练集为空，赋均等权重
            class_weights = torch.ones(label_num, dtype=torch.float32).to(device)

        print("Class Weights:", class_weights)
        # 定义损失函数和优化器
        criterion = torch.nn.NLLLoss(weight=class_weights)  # 分类问题使用 NLLLoss #注意损失函数是否加权
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # 开始训练
        best_val_acc = 0.0
        best_val_report = None
        # 有一个bug，在测试模式下，best_val_report为None，导致报告无法保存
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            for embeddings, labels in tqdm(
                train_loader,
                desc=f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            ):
                embeddings, labels = embeddings.to(device), labels.to(device)
                optimizer.zero_grad()
                if model_name == "GBAN":
                    outputs, attentions = model(embeddings)
                    # print(outputs.shape)
                    loss = criterion(outputs, labels)
                else:
                    outputs, weights = model(embeddings)
                    # print(outputs.shape)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            # 记录当前步骤（每个 batch）的性能数据
            # prof.step()

            train_acc = accuracy_score(train_labels, train_preds)
            print(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}"
            )

            # 验证阶段
            val_loss, val_acc, val_report = validate_model(
                model, val_loader, criterion, device, cover
            )
            print(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
            )

        # 保存当前折的验证集结果
        print(f"Fold {fold + 1} Validation Best Report:")
        print(best_val_acc)
        print(f"Fold {fold + 1} Validation Final Report:")
        print(val_acc)
        fold_results.append((fold + 1, val_acc, val_report))
        print(f"Fold {fold + 1} append End")

    # 输出每折的验证结果
    print("\nCross-Validation Results:")
    results_df = pd.DataFrame(
        columns=[
            "Fold",
            "Accuracy",
            "F1 Score",
            "Precision",
            "Recall",
            "Cover Accuracy",
            "Cover F1 Score",
            "Cover Precision",
            "Cover Recall",
            "Cover Threshold",
        ]
    )
    for fold, acc, report in fold_results:
        f1_score = report["macro avg"]["f1-score"]
        precision = report["macro avg"]["precision"]
        recall = report["macro avg"]["recall"]
        cover_accuracy = report["cover_metrics"]["accuracy"]
        cover_f1_score = report["cover_metrics"]["f1-score"]
        cover_precision = report["cover_metrics"]["precision"]
        cover_recall = report["cover_metrics"]["recall"]
        cover_threshold = report["cover_threshold"]
        print(
            f"Fold {fold}: Best Val Accuracy: {acc:.4f}, F1 Score: {f1_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )
        print(
            f"Cover Metrics: Accuracy: {cover_accuracy:.4f}, F1 Score: {cover_f1_score:.4f}, Precision: {cover_precision:.4f}, Recall: {cover_recall:.4f}, Threshold: {cover_threshold:.4f}"
        )

        # 记录每折的结果
        results_df = results_df._append(
            {
                "Fold": fold,
                "Accuracy": acc,
                "F1 Score": f1_score,
                "Precision": precision,
                "Recall": recall,
                "Cover Accuracy": cover_accuracy,
                "Cover F1 Score": cover_f1_score,
                "Cover Precision": cover_precision,
                "Cover Recall": cover_recall,
                "Cover Threshold": cover_threshold,
            },
            ignore_index=True,
        )
        print(results_df)

        avg_f1 += f1_score
        avg_precision += precision
        avg_recall += recall
        avg_accuracy += acc
        avg_cover_accuracy += cover_accuracy
        avg_cover_f1 += cover_f1_score
        avg_cover_precision += cover_precision
        avg_cover_recall += cover_recall

    print("End of Cross-Validation")

    avg_f1 /= k_folds
    avg_precision /= k_folds
    avg_recall /= k_folds
    avg_accuracy /= k_folds
    avg_cover_accuracy /= k_folds
    avg_cover_f1 /= k_folds
    avg_cover_precision /= k_folds
    avg_cover_recall /= k_folds

    print(
        f"Average Validation Accuracy: {avg_accuracy:.4f}, F1 Score: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}"
    )
    if save_results:
        results_df = results_df._append(
            {
                "Fold": "Average",
                "Accuracy": avg_accuracy,
                "F1 Score": avg_f1,
                "Precision": avg_precision,
                "Recall": avg_recall,
                "Cover Accuracy": avg_cover_accuracy,
                "Cover F1 Score": avg_cover_f1,
                "Cover Precision": avg_cover_precision,
                "Cover Recall": avg_cover_recall,
                "Cover Threshold": None,
            },
            ignore_index=True,
        )
        results_df.to_excel(
            os.path.join(
                results_save_path, "cross_validation_results" + str(label_num) + ".xlsx"
            ),
            index=False,
        )
        print(f"Results saved to {results_save_path}/cross_validation_results.xlsx")

    return avg_f1

if __name__ == "__main__":
    embedding_folder = "Embeddings_cls"  # 嵌入文件路径
    label_file = "labels/4.txt"  # 标签文件路径
    label_num = 4  # 标签对应的问题编号
    label_folder = "labels"  # 标签文件夹路径

    # 参数设置
    k_folds = 5
    batch_size = 2
    num_epochs = 20
    embedding_dim = 768
    hidden_dim = 128
    num_layers = 3
    dropout = 0.25
    cover = 0.8  # 保留分类可靠性排在前cover的样本
    attn_method = "dot_scaled"  # dot_scaled 或 additive 注意力机制

    cross_vali = True  # 是否进行交叉验证
    split = 0.8  # 训练集和验证集的比例

#以下为新加代码
    experiment_set = [50, 100, 150, 200, 250, 300, 510]  # 实验数据集大小可以调整
    max_questions = 16  # 最大问题数
    # experiment_set = [50]  # 测试
    # max_questions = 1  # 测试
    save_results = False  # 是否保存结果
    save_dir = "./saved_models_GBN"  # 模型保存目录
    All_questions = True  # 选择一组label或是label_folder里的所有labels
    model_names = ["GBAN"] # GBN 或 GBAN 模型
    results_save_path = "size_experiment"  # 结果保存路径 Data_size_test用于确定测试集大小，DL_results用于保存结果,DL_results_GBN    
    all_results = {}
    experiment_times = 5  # 实验次数
    os.makedirs(results_save_path, exist_ok=True)  # 确保结果保存路径的目录存在

    for data_size in experiment_set:
        print(f"Experiment on {data_size} data size")
        # 加载数据集
        if All_questions:
            for question_num in range(1, max_questions + 1):
                current_f1_score_list = []
                for experiment_time in range(experiment_times): 
                    label_file = "labels/" + str(question_num) + ".txt"
                    dataset = TextEmbeddingDataset(
                        embedding_folder, label_file, max_files=data_size
                    )
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    for model_name in model_names:
                        print(f"Label file: {label_file}, Data size: {data_size},model_name: {model_name}")

                        def model_class():
                            if model_name == "GBN":
                                return GBN(
                                    embedding_dim,
                                    hidden_dim,
                                    num_layers,
                                    dropout,
                                    attn_method=attn_method,
                                )
                            elif model_name == "GBAN":
                                return GBAN(
                                    embedding_dim,
                                    hidden_dim,
                                    num_layers,
                                    dropout,
                                    attn_method=attn_method,
                                )
                                raise ValueError("Invalid mode_name")

                        current_f1_score=train_with_cross_validation(
                            model_class,
                            dataset,
                            num_epochs,
                            batch_size,
                            device,
                            model_name,
                            question_num,
                            k_folds,
                            attn_method=attn_method,
                            save_dir="./saved_models",
                            save_results=save_results,
                            results_save_path=results_save_path,
                            cover=cover,
                        )
                        current_f1_score_list.append(current_f1_score)
                current_f1_score_list = np.array(current_f1_score_list)
                current_time_f1_score = np.mean(current_f1_score_list)                        
                if data_size not in all_results:
                    all_results[data_size] = {}
                if model_name not in all_results[data_size]:
                    all_results[data_size][model_name] = {}
                all_results[data_size][model_name][question_num] = current_f1_score
        
    with open(os.path.join(results_save_path, 'experiment_results.json'), 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

    print(f"Results saved to {results_save_path}/experiment_results.json")        

