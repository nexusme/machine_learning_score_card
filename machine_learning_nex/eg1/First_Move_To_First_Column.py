import numpy as np
import pandas as pd

# 读文件
CSV_FILE_PATH = "./cs-training.csv"
csv_data = pd.read_csv(CSV_FILE_PATH, index_col=0)

print(csv_data)
# 找到无效数据少量删除
csv_data_1 = csv_data.dropna(subset=['NumberOfDependents'], inplace=True)
print(csv_data)
# 测试代码
# nan_count = csv_data_1.isnull().sum()
# nan_count .to_csv("nan_count")


# 找中位数平均值并生成csv文件
# describe_csv = csv_data.describe()
# xx.to_csv('xxx.csv')

# MonthlyIncome缺失29731条数据 将其移到首列 不包含索引列
df_MonthlyIncome = csv_data.MonthlyIncome
csv_data = csv_data.drop('MonthlyIncome', axis=1)
csv_data.insert(0, 'MonthlyIncome', df_MonthlyIncome)

print(csv_data)
# nan_count_1 = csv_data.isnull().sum()
# print(nan_count_1)
csv_data.to_csv('cs-first-column.csv')

