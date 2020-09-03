import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 读文件
CSV_FILE_PATH = "cs-1.csv"
csv_data = pd.read_csv(CSV_FILE_PATH, index_col=0)
# print(csv_data)
'''找中位数平均值并生成csv文件
describe_csv = csv_data.describe()
describe_csv.to_csv('second_describe.csv')
print(describe_csv )'''


# 尝试一 使用随即森林填充缺失值 定义函数

def set_missing_part(df):
    # 把数值型特征放入随机森林里
    missing_part_df = df[['MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                          'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                          'NumberOfDependents', 'RevolvingUtilizationOfUnsecuredLines', 'age',
                          'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio']]
    known_part = missing_part_df[missing_part_df['MonthlyIncome'].notnull()].values
    unknown_part = missing_part_df[missing_part_df['MonthlyIncome'].isnull()].values

    y = known_part[:, 0]  # y，第一列数据
    x = known_part[:, 1:]  # x是特征属性值，后面几列
    rfr = RandomForestRegressor(random_state=0, n_estimators=15, max_depth=10, n_jobs=-1)
    # print('3')
    # 根据已有数据去拟合随机森林模型
    rfr.fit(x, y)
    # print('4')
    # 预测缺失值
    predicted_results = rfr.predict(unknown_part[:, 1:])
    predicted_results = [int(x) for x in predicted_results]
    print(predicted_results)

    # 填补缺失值
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted_results
    # print('5')
    return df


csv_data_new = set_missing_part(csv_data)
csv_data_new.to_csv('cs-data-fill.csv')
print(csv_data_new)
