import numpy as np
import pandas as pd
from dawid_skene_model import list2array
from dawid_skene_model import DawidSkeneModel
class_num = 5 # number of classes
max_questions = 16
threshold = 3


file_names = ["1_510.xlsx", "2_510.xlsx", "3_510.xlsx"]
dataframes = [pd.read_excel(file_path) for file_path in file_names]
length = len(dataframes[0])


def DS(max_questions,class_num,dataframes):
# 两层循环，外层循环是每个问题，内层循环是每个rater的回答
    predict = []
    for j in range(max_questions):
        current_question_r1 = dataframes[0].iloc[:, 1+j]
        current_question_r2 = dataframes[1].iloc[:, 1+j]
        current_question_r3 = dataframes[2].iloc[:, 1+j]
        current_question = []
        # 内层循环为了构建针对每个文本的向量
        for i in range(len(current_question_r1)):
            current_text = []
            current_text.append([current_question_r1[i]-1])
            current_text.append([current_question_r2[i]-1])
            current_text.append([current_question_r3[i]-1])
            current_question.append(current_text)


        dataset_tensor = list2array(class_num, current_question)
        model = DawidSkeneModel(class_num, max_iter=45, tolerance=10e-100)
        marginal_predict, error_rates, worker_reliability, predict_label = model.run(dataset_tensor)
        predict.append(predict_label)
    predict_array = np.array(predict)
    return predict_array




# 随机生成预测结果
def generate_random_predict(class_num, max_questions, length):
    predict_array = np.random.randint(0, class_num, size=( length, max_questions))
    predict_array = pd.DataFrame(predict_array)
    return predict_array

# 转换预测结果为0-1的标签
def transform_value(value):
    if value < threshold:
        return 0
    else:
        return 1

if __name__ == '__main__':
    predict_array = DS(max_questions,class_num,dataframes)


    # 找到每个 5 维向量中值最大的索引
    max_indices = np.argmax(predict_array, axis=2)

    # 找到每个 5 维向量中值最大的元素
    max_values = np.max(predict_array, axis=2)

    # 创建一个新的数组来存储结果
    new_predict_array = np.copy(max_indices)

    # # 检测最大元素是否大于 0.75
    # new_predict_array[max_values <= 0.75] = -1

    # 将结果转置
    new_predict_array_transposed = new_predict_array.T

    # 将结果转换为 DataFrame
    new_predict = pd.DataFrame(new_predict_array_transposed)

    # 保存新的 DataFrame 为 Excel 文件
    new_predict.to_excel('DS_predict_output_510.xlsx', index=False)

    #转换预测结果为0-1的标签
    new_predict = new_predict.map(transform_value)

    # 保存新的 DataFrame 为 Excel 文件
    new_predict.to_excel('DS_predict_output_0-1_510_notype_threshold3.xlsx', index=False)