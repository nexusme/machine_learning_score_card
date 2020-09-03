import pandas as pd


# 读文件
CSV_FILE_PATH = "cs-data-fill.csv"
csv_data = pd.read_csv(CSV_FILE_PATH, index_col=0)
# print(csv_data)
# 找中位数平均值并生成csv文件
# describe_csv = csv_data.describe()
# describe_csv.to_csv('third_describe.csv')
# print(describe_csv)

# 数据处理
# age>0
csv_data_1 = csv_data[csv_data.age > 0]
# csv_data_1 = csv_data[csv_data.age > 0]

csv_data_1.to_csv('cs-data-fill-select-age.csv')
print(csv_data_1)
