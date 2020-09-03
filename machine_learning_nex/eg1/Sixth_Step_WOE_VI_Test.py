import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 读文件
CSV_FILE_PATH = "cs-data-delete-after-boxplot.csv"
csv_data = pd.read_csv(CSV_FILE_PATH, index_col=0)

cols = list(csv_data)
cols.insert(0, cols.pop(cols.index('SeriousDlqin2yrs')))
csv_data = csv_data.loc[:, cols]
csv_data.rename(columns={'SeriousDlqin2yrs': 'y'}, inplace=True)

# print(csv_data)

# def chi2(arr):
#     # 计算卡方值
#     # arr:频数统计表,二维numpy数组
#     assert (arr.ndim == 2)
#     # 计算每行总频数
#     R_N = arr.sum(axis=1)
#     # 每列总频数
#     C_N = arr.sum(axis=0)
#     # 总频数
#     N = arr.sum()
#     # 计算期望频数 C_i * R_j / N。
#     E = np.ones(arr.shape) * C_N / N
#     E = (E.T * R_N).T
#     square = (arr - E) ** 2 / E
#     # 期望频数为0时，做除数没有意义，不计入卡方值
#     square[E == 0] = 0
#     # 卡方值
#     v = square.sum()
#     return v
#
#
# def chiMerge(df, col, target, max_groups=None, threshold=None):
#     # 卡方分箱
#     # df: pandas dataframe数据集
#     # col: 需要分箱的变量名（数值型）
#     # target: 类标签
#     # max_groups: 最大分组数。
#     # threshold: 卡方阈值，如果未指定max_groups，默认使用置信度95%设置threshold。
#     # return: 包括各组的起始值的列表.
#     freq_tab = pd.crosstab(df[col], df[target])
#     # 转成numpy数组用于计算。
#     freq = freq_tab.values
#     # 初始分组切分点，每个变量值都是切分点。每组中只包含一个变量值.
#     # 分组区间是左闭右开的，如cutoffs = [1,2,3]，则表示区间 [1,2) , [2,3) ,[3,3+)。
#     cutoffs = freq_tab.index.values
#     # 如果没有指定最大分组
#     if max_groups is None:
#         # 如果没有指定卡方阈值，就以95%的置信度（自由度为类数目-1）设定阈值。
#         if threshold is None:
#             # 类数目
#             cls_num = freq.shape[-1]
#             threshold = chi2.isf(0.05, df=cls_num - 1)
#
#     while True:
#         minvalue = None
#         minidx = None
#         # 从第1组开始，依次取两组计算卡方值，并判断是否小于当前最小的卡方
#         for i in range(len(freq) - 1):
#             v = chi2(freq[i:i + 2])
#             if minvalue is None or minvalue > v:
#                 # 小于当前最小卡方，更新最小值
#                 minvalue = v
#                 minidx = i
#         # 如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
#         if (max_groups is not None and max_groups < len(freq)) or (threshold is not None and minvalue < threshold):
#             # minidx后一行合并到minidx
#             tmp = freq[minidx] + freq[minidx + 1]
#             freq[minidx] = tmp
#             # 删除minidx后一行
#             freq = np.delete(freq, minidx + 1, 0)
#             # 删除对应的切分点
#             cutoffs = np.delete(cutoffs, minidx + 1, 0)
#         else:
#             break
#         # 最小卡方值不小于阈值，停止合并。
#     return cutoffs
#
#
# def valueToGroup(x, cutoffs):
#     # 将变量的值转换到相应的组 x:需要转换到分组的值 cutoffs:各组的起始值
#     # return：x对应的组 如group1 从group1开始
#     # 切分点从小到大排序
#     cutoffs = sorted(cutoffs)
#     num_groups = len(cutoffs)
#     if x < cutoffs[0]:
#         return 'group1'
#     for i in range(1, num_groups):
#         if cutoffs[i - 1] <= x < cutoffs[i]:
#             return 'group{}'.format(i)
#     return 'group{}'.format(num_groups)
#
#
# def calWOE(df, var, target):
#     # 计算WOE编码
#     # df:数据集 var:已分组列名 target:响应变量0，1 return:编码字典
#     eps = 0.000001
#     gbi = pd.crosstab(df[var], df[target]) + eps
#     gb = df[target].value_counts() + eps
#     gbri = gbi / gb
#     gbri['woe'] = np.log(gbri[1] / gbri[0])
#     return gbri['woe'].to_dict()
#
#
# def calIV(df, var, target):
#     # 计算IV值
#     # df:数据集 var:已分组列名 target:响应变量0，1 return:IV值
#     eps = 0.000001
#     gbi = pd.crosstab(df[var], df[target]) + eps
#     gb = df[target].value_counts() + eps
#     gbri = gbi / gb
#     gbri['woe'] = np.log(gbri[1] / gbri[0])
#     gbri['iv'] = (gbri[1] - gbri[0]) * gbri['woe']
#     return gbri['iv'].sum()
#
#
# def valueToWOE(x, cutoffs, woe_map):
#     # 变量值转换成相应woe编码
#     # x: 需要转换到分组的值 cutoffs: 各组的起始值
#     # woe_map: woe编码字典 return:woe 编码
#     # 切分点从小到大排序
#     cutoffs = sorted(cutoffs)
#     num_groups = len(cutoffs)
#     group = None
#     if x < cutoffs[0]:
#         return 'group1'
#     for i in range(1, num_groups):
#         if cutoffs[i - 1] <= x < cutoffs[i]:
#             return 'group{}'.format(i)
#     if group is None:
#         group = 'group{}'.format(num_groups)
#
#     if group in woe_map:
#         return woe_map[group]
#
#     return None
#
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.33, random_state=0)

# 打印出原始样本集、训练集和测试集的数目
print("The length of train Data is:", X_train.shape[0])
print("The length of test Data is:", X_test.shape[0])


def mono_bin(Y, X, X_test, column_name, n=20):
    # Y: ytrain
    # X: xtrain

    # X_test:
    # column_name:
    # n: 分箱数
    print("@@ woe coding", column_name)
    r = 0
    good = Y.sum()
    bad = len(Y) - good
    X = pd.Series(X)
    if column_name == 'NumberOfTime30-59DaysPastDueNotWorse' or column_name == 'NumberOfOpenCreditLinesAndLoans' \
            or column_name == 'NumberOfTimes90DaysLate' or column_name == 'NumberRealEstateLoansOrLines' or \
            column_name == 'NumberOfTime60-89DaysPastDueNotWorse' or column_name == 'NumberOfDependents':
        bins = [-1e-3, 1, 2, 3, 5, float('inf')]
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": -1})
        for i in range(len(bins) - 1):
            d1.loc[(d1['X'] > bins[i]) & (d1['X'] <= bins[i + 1]), "Bucket"] = str([bins[i], bins[i + 1]])
        d2 = d1.groupby('Bucket', as_index=True)
    # 如果不是上述这些则
    else:
        while np.abs(r) < 1 - 1e-7:
            # 进行等频分箱
            array_bin, bins = pd.qcut(X, n, duplicates='drop', retbins=True)
            d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": array_bin})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
            # if n <= 2:
            #     print(column_name + " scannot using supviced cut, using non_supviced cut instead!")
            #     quantile_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            #     bins = []
            #     bins.append(float('-inf'))
            #     for i in range(len(quantile_per)):
            #         bins.append(pd.Series(X).quantile(quantile_per[i]))
            #     bins.append(float('inf'))
            #     print(bins)
            #     d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": -1})
            #     for i in range(len(bins)-1):
            #         d1.loc[(d1['X']>bins[i]) & (d1['X']<=bins[i+1]), "Bucket"] = str([bins[i], bins[i+1]])
            #     d2 = d1.groupby('Bucket', as_index=True)
            #     break

    d3 = pd.DataFrame()
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['num_of_good'] = d2.sum().Y
    d3['num_of_bad'] = d2.size() - d3['num_of_good']
    d3['total'] = d2.size()
    d3['distribution of good'] = d3['num_of_good'] / good
    d3['distribution of bad'] = d3['num_of_bad'] / bad
    d3['woe'] = np.log(d3['distribution of good'] / d3['distribution of bad'])
    d3['cumulative infomation value'] = (d3['distribution of good'] - d3['distribution of bad']) * d3['woe']
    iv = d3['cumulative infomation value'].sum()

    bins[0] = bins[0] - 1e-4
    d3['start'] = bins[:-1]
    d3['end'] = bins[1:]
    d3 = d3.sort_values('min')
    d3.to_csv(column_name + '.csv')
    print("=" * 60)
    print(d3)

    X = pd.Series(X)
    X_test = pd.Series(X_test)
    X_test[(X_test < bins[0]) & (X_test > bins[-1])] = np.nan

    print("###", len(d3.index))
    for i in range(len(d3.index)):
        print("@@@@@@woe:" + str(d3['woe'][i]) + " @@@@@@@")
        X[(X > d3['start'][i]) & (X <= d3['end'][i])] = d3['woe'][i]
        X_test[(X_test > d3['start'][i]) & (X_test <= d3['end'][i])] = d3['woe'][i]
    return iv, X.values, X_test
