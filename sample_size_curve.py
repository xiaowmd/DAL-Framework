import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json
import os
# --- 核心函数：曲线拟合模型 ---

def inverse_power_law(x, a, b, c):
    """
    定义逆幂律模型函数。
    这是一个常用的学习曲线模型，表示性能会随着样本量的增加而饱和。
    - a: 最终的饱和性能的上限（理论最大F1值）
    - b: 学习速率相关的系数
    - c: 曲线的弯曲程度
    模型形式: F1(x) = a - b * x^c
    """
    return a - b * np.power(x, c)

def weighted_curve_fit(x_data, y_data):
    """
    定义非线性加权最小二乘拟合函数。
    增加了参数边界以确保拟合曲线是单调递增的。
    """
    if len(x_data) < 3:
        return None, None
    m = len(x_data)
    weights = np.array([(j + 1) / m for j in range(m)])
    initial_guess = [np.max(y_data), 0.1, -0.1]
    
    # --- 修改部分：增加参数边界 ---
    # 约束 a (最大F1) 在 [当前最大F1, 1.05] 之间
    # 约束 b (缩放系数) > 0
    # 约束 c (指数) < 0
    # 这将强制拟合函数为单调递增的饱和曲线
    lower_bounds = [np.min(y_data), 0, -np.inf]
    upper_bounds = [1.05, np.inf, 0]
    bounds = (lower_bounds, upper_bounds)
    
    try:
        popt, pcov = curve_fit(
            inverse_power_law, 
            x_data, 
            y_data, 
            p0=initial_guess, 
            sigma=1/weights, 
            maxfev=10000,
            bounds=bounds  # 应用边界
        )
        return popt, pcov
    except RuntimeError:
        print(f"Warning: Curve fitting failed for data x={x_data}, y={y_data}")
        return None, None

# --- 新增函数：预测样本量 ---

def predict_sample_size(popt, target_f1):
    """
    根据拟合参数预测达到目标F1分数所需的样本量。
    从 y = a - b * x^c  解出 x:  x = ((a - y) / b)^(1/c)
    """
    a, b, c = popt
    if c == 0 or b == 0:
        return None
    if target_f1 >= a:
        return ">= Max"
    base = (a - target_f1) / b
    if base <= 0:
        return None
    try:
        predicted_x = (base) ** (1 / c)
        return predicted_x
    except (ValueError, OverflowError):
        return None

# --- 数据处理函数 ---

def read_experiment_results(file_path):
    """
    从指定的JSON文件路径读取实验结果。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")
    
    with open(file_path, 'r', encoding='utf-8') as json_file:
        try:
            results = json.load(json_file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON from {file_path}: {e.msg}", e.doc, e.pos)
    
    return results

def restructure_data(results):
    """
    将单个文件中的原始数据重组为按问题分组的格式，并只筛选 'HEA' 模型的结果。
    """
    questions_data = {}
    sample_sizes = sorted([int(s) for s in results.keys()])

    for size in sample_sizes:
        size_str = str(size)
        for model_name, questions in results[size_str].items():
            if not model_name.startswith('HEA'):
                continue
            
            for question_str, f1_score in questions.items():
                if question_str not in questions_data:
                    questions_data[question_str] = {'x': [], 'y': []}
                
                questions_data[question_str]['x'].append(size)
                questions_data[question_str]['y'].append(f1_score)
            
    return questions_data

# --- 绘图函数 (已重构) ---

def plot_all_in_one_figure(all_data, output_dir="fitting_curves"):
    """
    为每个分类类型创建一个包含所有问题子图的大图。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 遍历每个分类类型 (e.g., '2class', '5class')
    for class_type, questions in all_data.items():
        print(f"\nProcessing classification type: {class_type}...")
        
        # 创建一个 4x4 的子图网格
        fig, axes = plt.subplots(4, 4, figsize=(20, 18), constrained_layout=True)
        fig.suptitle(f'F1-Score vs. Sample Size (Model: GBAN, Type: {class_type})', fontsize=24)
        
        # 将2D的axes数组扁平化为1D，方便遍历
        axes = axes.flatten()
        
        # 对问题进行排序，确保绘图顺序一致
        sorted_questions = sorted(questions.items(), key=lambda item: int(item[0]))

        # 遍历16个问题并填充子图
        for i, (question, values) in enumerate(sorted_questions):
            if i >= len(axes): break # 防止问题数超过子图数
            
            ax = axes[i]
            x_data = np.array(values['x'])
            y_data = np.array(values['y'])
            popt, _ = weighted_curve_fit(x_data, y_data)
            
            ax.scatter(x_data, y_data, label='Original Data', color='red', zorder=5, s=20)

            if popt is not None:
                x_fit = np.linspace(min(x_data), max(x_data) * 1.1, 200)
                y_fit = inverse_power_law(x_fit, *popt)
                ax.plot(x_fit, y_fit, label='Fitted Curve', color='blue', linewidth=2)
                
                param_text = f'a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}'
                target_f1_score = 0.75
                predicted_size = predict_sample_size(popt, target_f1_score)
                
                prediction_text = f'\nPred. size for F1={target_f1_score}: '
                if predicted_size is not None:
                    prediction_text += f'{predicted_size:.0f}' if isinstance(predicted_size, (int, float)) else predicted_size
                else:
                    prediction_text += 'N/A'

                ax.text(0.05, 0.95, param_text + prediction_text, transform=ax.transAxes,
                         fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.6))

            ax.set_title(f'Question: {question}', fontsize=14)
            ax.set_xlabel('Sample Size', fontsize=10)
            ax.set_ylabel('F1-Score', fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(fontsize=8)

        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # 保存整个图表
        file_name = f"HEA_{class_type}_all_questions.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        print(f"  - Saved consolidated plot to {save_path}")

# --- 主执行函数 ---

def main():
    """
    主函数，协调整个流程：读取 -> 重组 -> 绘图。
    """
    data_folder = 'size_experiment'
    experiment_files = {
        '2class': 'experiment_results_2.json',
        '5class': 'experiment_results_5.json'
    }
    
    all_structured_data = {}

    for class_type, file_name in experiment_files.items():
        file_path = os.path.join(data_folder, file_name)
        try:
            print(f"Reading data for '{class_type}' from {file_path}...")
            raw_results = read_experiment_results(file_path)
            structured_data = restructure_data(raw_results)
            all_structured_data[class_type] = structured_data
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"\nAn error occurred while processing {file_name}: {e}")
            print("Please ensure the file exists and is correctly formatted.")
            continue
    
    if not all_structured_data:
        print("\nNo data was successfully processed. Exiting.")
        return
        
    plot_all_in_one_figure(all_structured_data)
    
    print("\nAll plots have been generated and saved successfully! 🎉")

if __name__ == "__main__":
    main()
