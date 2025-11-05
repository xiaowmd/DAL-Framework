import krippendorff
import numpy as np
import pandas as pd




max_questions = 16

# Load the Excel files
file_names = ["1_510.xlsx", "2_510.xlsx", "3_510.xlsx"]


dataframes = [pd.read_excel(file_path) for file_path in file_names]
data = []
results = np.zeros(max_questions)

for i in range(max_questions):
    current_question_r1 = dataframes[0].iloc[:, i + 1]
    current_question_r2 = dataframes[1].iloc[:, i + 1]
    current_question_r3 = dataframes[2].iloc[:, i + 1]
    current_question = [current_question_r1, current_question_r2, current_question_r3]
    results[i] = krippendorff.alpha(
        reliability_data=current_question, level_of_measurement="ordinal"
    )

df = pd.DataFrame(results)
df.to_excel("krippendorff_alpha.xlsx", index=False)