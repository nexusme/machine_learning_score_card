import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

# 读文件
CSV_FILE_PATH = "cs-data-delete-after-boxplot.csv"
csv_data = pd.read_csv(CSV_FILE_PATH, index_col=0)

cols = list(csv_data)
cols.insert(0, cols.pop(cols.index('SeriousDlqin2yrs')))
csv_data = csv_data.loc[:, cols]
csv_data.rename(columns={'SeriousDlqin2yrs': 'y'}, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(csv_data, csv_data['y'], test_size=0.3, random_state=0)


# print(csv_data)


# 等频分箱
# def mono_bin(Y, X, X_test, column_name, nn=20):
#     # Y: ytrain X: xtrain
#     # X_test:
#     # column_name:
#     # n: 分箱数
#     print("@@ woe coding", column_name)
#     r = 0
#     good = Y.sum()
#     bad = len(Y) - good
#     X = pd.Series(X)
#     # while np.abs(r) < 1 - 1e-7:
#     #     # 进行等频分箱
#     array_bin, bins = pd.qcut(X, nn, duplicates='drop', retbins=True)
#     d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": array_bin})
#     d2 = d1.groupby('Bucket', as_index=True)
#     # r, p = spearmanr(d2.mean().X, d2.mean().Y)
#     # nn = nn - 1
#     d3 = pd.DataFrame()
#     d3['min'] = d2.min().X
#     d3['max'] = d2.max().X
#     d3['num_of_good'] = d2.sum().Y
#     d3['num_of_bad'] = d2.size() - d3['num_of_good']
#     d3['total'] = d2.size()
#     d3['distribution of good'] = d3['num_of_good'] / good
#     d3['distribution of bad'] = d3['num_of_bad'] / bad
#     d3['woe'] = np.log(d3['distribution of good'] / d3['distribution of bad'])
#     d3['cumulative infomation value'] = (d3['distribution of good'] - d3['distribution of bad']) * d3['woe']
#     iv = d3['cumulative infomation value'].sum()
#     print(iv)
#     bins[0] = bins[0] - 1e-4
#     d3['start'] = bins[:-1]
#     d3['end'] = bins[1:]
#     d3 = d3.sort_values('min')
#     # d3.to_csv(column_name + '.csv')
#     print("=" * 60)
#     print(d3)
#
#     X = pd.Series(X)
#     X_test = pd.Series(X_test)
#     X_test[(X_test < bins[0]) & (X_test > bins[-1])] = np.nan
#
#     print("###", len(d3.index))
#     for i in range(len(d3.index)):
#         print("@@@@@@woe:" + str(d3['woe'][i]) + " @@@@@@@")
#         X[(X > d3['start'][i]) & (X <= d3['end'][i])] = d3['woe'][i]
#         X_test[(X_test > d3['start'][i]) & (X_test <= d3['end'][i])] = d3['woe'][i]


def best_bins(x, y, name, n):  # x为待分箱的变量，y为target变量.n为分箱数量
    list = []
    while n > 2:
        total = y.count()  # 计算总样本数
        bad = y.sum()  # 计算坏样本数
        good = total - y.sum()  # 计算好样本数
        array_bin, bins = pd.qcut(x, n, duplicates='drop', retbins=True)
        d1 = pd.DataFrame({'x': x, 'y': y, 'bucket': array_bin})  # 用pd.cut实现等频分箱
        # print(array_bin)
        # print('d1', d1)
        d2 = d1.groupby('bucket', as_index=True)  # 按照分箱结果进行分组聚合
        d3 = pd.DataFrame(d2.x.min(), columns=['min_bin'])
        d3['min_bin'] = d2.x.min()  # 箱体的左边界
        d3['max_bin'] = d2.x.max()  # 箱体的右边界
        d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
        d3['total'] = d2.y.count()  # 每个箱体的总样本数
        d3['bad_rate'] = d3['bad'] / d3['total']  # 每个箱体中坏样本所占总样本数的比例
        d3['badattr'] = d3['bad'] / bad  # 每个箱体中坏样本所占坏样本总数的比例
        d3['goodattr'] = (d3['total'] - d3['bad']) / good  # 每个箱体中好样本所占好样本总数的比例
        d3['woe'] = np.log(d3['goodattr'] / d3['badattr'])  # 计算每个箱体的woe值
        iv = ((d3['goodattr'] - d3['badattr']) * d3['woe']).sum()  # 计算变量的iv值
        d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True)  # 对箱体从大到小进行排序
        # print('分箱结果：')
        # print(d4)
        # print('IV值为：')
        # print(iv)
        # print('n值为：')
        # print(n)
        d5 = {'n': n, 'length of bins': len(bins), 'iv': iv, 'bins': bins}
        list.append(d5)
        n = n - 1
    # print(list)
    d6 = DataFrame(list)
    print(d6)
    d6_group = d6.sort_values(by="iv", ascending=False)
    # print(d6_group)
    d6_group_best = d6_group.iloc[0]
    print('The best result for ' + name + ' is:\n', d6_group_best)
    return d6_group_best['bins']


# cut = [float('-inf')]
# for i in d4.min_bin:
#     cut.append(i)
# cut.append(float('inf'))
# woe = list(d4['woe'].round(3))


#
# train_X, test_X, train_y, test_y =
# train_test_split(csv_data, csv_data['MonthlyIncome'], test_size=0.33, random_state=0)
# print("train_X:", train_X)
# print("train_y:", train_y)
# print("test_X:", test_X)


best_bins(x_train['age'].values, y_train, 'age', 20)  # dict
best_bins(x_train['MonthlyIncome'].values, y_train, 'MonthlyIncome', 20)  # dict
best_bins(x_train['RevolvingUtilizationOfUnsecuredLines'].values, y_train, 'RevolvingUtilizationOfUnsecuredLines',
          20)  # dict
best_bins(x_train['DebtRatio'].values, y_train, 'DebtRatio', 20)  # dict
best_bins(x_train['NumberOfOpenCreditLinesAndLoans'].values, y_train, 'NumberOfOpenCreditLinesAndLoans', 20)  # dict
best_bins(x_train['NumberRealEstateLoansOrLines'].values, y_train, 'NumberRealEstateLoansOrLines', 20)  # dict
best_bins(x_train['NumberOfDependents'].values, y_train, 'NumberOfDependents', 20)  # dict
best_bins(x_train['NumberOfTime60-89DaysPastDueNotWorse'].values, y_train, 'NumberOfTime60-89DaysPastDueNotWorse',
          30)  # dict
best_bins(x_train['NumberOfTime30-59DaysPastDueNotWorse'].values, y_train, 'NumberOfTime30-59DaysPastDueNotWorse',
          20)  # dict
best_bins(x_train['NumberOfTimes90DaysLate'].values, y_train, 'NumberOfTimes90DaysLate', 20)  # dict

