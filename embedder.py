import torch
from transformers import BertTokenizer, BertModel
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset


#TODO 配置BERT模型
# model_name = "bert-base-chinese" # 选择BERT模型
model_name = "hfl/chinese-bert-wwm" # 选择BERT模型wwm


need_process = False  #TODO 表示是否需要处理
is_cls_embedding = True  #TODO 表示是否只保存最后一层隐藏状态的CLS向量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备


def load_text_from_folder(folder_path):
    """
    读取文件夹中的所有txt文件，将每行文本作为列表元素返回
    """
    texts = {}
    for file_path in Path(folder_path).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            texts[file_path.stem] = file.readlines()  # 保存文件名（不含扩展名）和内容
    return texts

class TextDataset(Dataset):
    def __init__(self, text_lines):
        self.texts = text_lines  # 传入单个文本的每行内容

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def save_embeddings(embeddings, output_path):
    """
    将嵌入保存到指定路径
    """
    np.save(output_path, embeddings)

# 原嵌入函数，保存了最后一层隐藏状态的所有值
def process_and_save_embeddings(input_folder, output_folder, batch_size=8):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取文件夹下的所有文本，每个文件的每行作为独立样本
    texts = load_text_from_folder(input_folder)
    
    for file_name, text_lines in texts.items():
        # 使用自定义的数据集和 DataLoader
        dataset = TextDataset(text_lines)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 保存该文件的所有句子向量
        cls_embeddings = []

        # 使用 tqdm 显示处理进度
        for batch_sentences in tqdm(data_loader, desc=f"Processing embeddings for {file_name}"):
            # 对每一批次的句子进行编码，确保所有句子填充到 max_length（512）
            inputs = tokenizer(batch_sentences, padding='max_length', truncation=True, return_tensors="pt", max_length=512)

            # 将编码好的数据放到 GPU 上，包括 attention_mask
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # 计算嵌入，不计算梯度
            with torch.no_grad():
                outputs = model(**inputs)

            # 获取最后一层隐藏状态并将其移回 CPU
            cls_embeddings.append(outputs.last_hidden_state.cpu().numpy())

        # 合并该文本的所有批次嵌入
        cls_embeddings_np = np.vstack(cls_embeddings)

        # 保存嵌入到指定文件夹，以文件名命名
        output_path = os.path.join(output_folder, f"{file_name}_embeddings.npy")
        save_embeddings(cls_embeddings_np, output_path)
        print(f"Embeddings for {file_name} saved to {output_path}")

# 新嵌入函数，只保存最后一层隐藏状态的CLS向量
def process_and_save_cls_embeddings(input_folder, output_folder, batch_size=8):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    texts = load_text_from_folder(input_folder)
    
    for file_name, text_lines in texts.items():
        # 使用自定义的数据集和 DataLoader
        dataset = TextDataset(text_lines)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 保存该文件的所有句子向量
        cls_embeddings = []

        # 使用 tqdm 显示处理进度
        for batch_sentences in tqdm(data_loader, desc=f"Processing embeddings for {file_name}"):
            # 对每一批次的句子进行编码，确保所有句子填充到 max_length（512）
            inputs = tokenizer(batch_sentences, padding='max_length', truncation=True, return_tensors="pt", max_length=512)

            # 将编码好的数据放到 GPU 上，包括 attention_mask
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # 计算嵌入，不计算梯度
            with torch.no_grad():
                outputs = model(**inputs)

            # 获取最后一层隐藏状态的CLS向量并将其移回 CPU
            cls_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            # 合并该文本的所有批次嵌入
        cls_embeddings_np = np.vstack(cls_embeddings)

        # 保存嵌入到指定文件夹，以文件名命名
        output_path = os.path.join(output_folder, f"{file_name}_embeddings.npy")
        save_embeddings(cls_embeddings_np, output_path)
        print(f"Embeddings for {file_name} saved to {output_path}")


def load_and_view_embeddings(file_path):
    """
    加载并查看保存的嵌入文件
    Args:
        file_path (str): 嵌入文件的路径（.npy 格式）
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # 加载嵌入文件
    embeddings = np.load(file_path)
    
    # 查看嵌入的形状
    print(f"Embedding shape: {embeddings.shape}")
    
    # 查看前几个嵌入向量
    print("First few token embeddings:\n", embeddings[:5])


if __name__ == '__main__':
    if need_process:  # 是否需要处理
        if is_cls_embedding:  # 是否只保存最后一层隐藏状态的CLS向量
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            model.to(device)
            model.eval()
            input_folder = 'DataForDL'  # TODO 预处理文本的文件夹路径
            output_folder = 'Embeddings_cls'  #TODO  保存嵌入结果的文件夹路径
            process_and_save_cls_embeddings(input_folder, output_folder)
        else:  # 保存最后一层隐藏状态的所有值
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            model.to(device)
            model.eval()
            input_folder = 'DataForDL'  # TODO 预处理文本的文件夹路径
            output_folder = 'Embeddings'  #TODO  保存嵌入结果的文件夹路径
            process_and_save_embeddings(input_folder, output_folder)

    else:  # 直接加载并查看嵌入
        embedding_path = 'Embeddings_cls/A001_embeddings.npy'  # 嵌入文件路径
        load_and_view_embeddings(embedding_path)

    


