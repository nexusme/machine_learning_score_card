import numpy as np
import pandas as pd

from pandas import DataFrame as dff

from pandas import DataFrame
from sklearn.model_selection import train_test_split


def change_name_SD2yrs_Y():
    # 读文件
    CSV_FILE_PATH = "cs-data-delete-after-boxplot.csv"
    data = pd.read_csv(CSV_FILE_PATH, index_col=0)
    cols = list(data)
    cols.insert(0, cols.pop(cols.index('SeriousDlqin2yrs')))
    data = data.loc[:, cols]
    data.rename(columns={'SeriousDlqin2yrs': 'y'}, inplace=True)
    return data


csv_data = change_name_SD2yrs_Y()
# print(csv_data)
# print(len(csv_data)) 145802
# print(len(csv_data['y']))

x_train, x_test, y_train, y_test = train_test_split(csv_data, csv_data['y'], test_size=0.3, random_state=0)


# print(len(y_train))
# print(len(x_train))
# print(x_test)


# print(y_test)


def calculate_chi_square(df, bins, index):
    box1 = df[(df['X'] >= bins[index]) & (df['X'] < bins[index + 1])]
    box2 = df[(df['X'] >= bins[index + 1]) & (df['X'] < bins[index + 2])]

    # print(len(box1), len(box2), index)
    a11 = len(box1[box1["Y"] == 0])
    a12 = len(box1[box1['Y'] == 1])
    a21 = len(box2[box2['Y'] == 0])
    a22 = len(box2[box2['Y'] == 1])

    a_list = [a11, a12, a21, a22]

    r1 = a11 + a12
    r2 = a21 + a22
    n = r1 + r2
    c1 = a11 + a21
    c2 = a12 + a22

    e11 = r1 * c1 / n
    e12 = r1 * c2 / n
    e21 = r2 * c1 / n
    e22 = r2 * c2 / n

    e_list = [e11, e12, e21, e22]

    chi_square = 0
    try:
        for k in range(len(a_list)):
            chi_square += (a_list[k] - e_list[k]) ** 2 / e_list[k]
    except:
        print(a_list, e_list, bins[index], r1, r2, c1, c2)
        raise
    return chi_square


def get_chi_square(df, ori_bins):
    chi_square_values_list = []
    for i in range(len(ori_bins) - 2):
        # print(ori_bins[i])
        chi_square_value = calculate_chi_square(df, ori_bins, i)
        chi_square_values_list.append(chi_square_value)
    return chi_square_values_list


def ini_chi_merge(bins, df):
    i = 0
    while i < len(bins) - 2:
        df_adjacent = df[(df['X'] >= bins[i]) & (df['X'] < bins[i + 2])]
        if len(df_adjacent[df_adjacent["Y"] == 0]) == 0 or len(df_adjacent[df_adjacent["Y"] == 1]) == 0:
            del bins[i + 1]
            i -= 1
        i += 1
    return bins


def chi_square_merge_goes(X_test_column, Y_test, Y_train, X_column, col_name):
    print(col_name)
    # Y_train = 1 响应 Y_train = 0 未响应
    # X_column待分箱变量
    # 卡方分箱阈值
    THRESHOLD = 10000
    # 分箱数目限制
    LIMIT_NUM = 6

    df = pd.DataFrame({"Y": Y_train, "X": X_column})
    df = df.sort_values("X")

    df_test = pd.DataFrame({"Y": Y_test, "X": X_test_column})
    df_test = df_test.sort_values("X")
    # print('test数据\n', df_test)

    # print(df)
    original_bins = sorted(set(df["X"].values))

    total = df.Y.count()  # 计算总样本数
    bad = df.Y.sum()  # 计算坏样本总数
    good = total - bad  # 计算好样本总数
    total_bin = df.groupby(['X'])['Y'].count()  # 计算每个箱体总样本数
    total_bin_table = pd.DataFrame({'total': total_bin})  # 创建一个数据框保存结果
    bad_bin = df.groupby(['X'])['Y'].sum()  # 计算每个箱体的坏样本数
    bad_bin_table = pd.DataFrame({'bad': bad_bin})  # 创建一个数据框保存结果
    regroup = pd.merge(total_bin_table, bad_bin_table, left_index=True, right_index=True,
                       how='inner')  # 组合total_bin 和 bad_bin
    regroup.reset_index(inplace=True)
    regroup['good'] = regroup['total'] - regroup['bad']  # 计算每个箱体的好样本数
    regroup = regroup.drop(['total'], axis=1)  # 删除total
    # print(regroup)

    np_regroup = np.array(regroup)  # 将regroup转为numpy
    # print(np_regroup)

    original_bins.append(np.inf)

    # 预处理 合并全为0/1的区间

    original_bins = ini_chi_merge(original_bins, df)
    # print(original_bins)
    # 开始计算最初的卡方值
    chi_square_list = get_chi_square(df, original_bins)
    # print(chi_square_list)
    # 开始合并
    while 1:
        # print('Current chi merge box is: ', len(original_bins) - 1)

        min_chi_square = min(chi_square_list)

        min_chi_square_index = chi_square_list.index(min_chi_square)
        # print('The min index is: ', min_chi_square_index)
        # print('Original bin is: ', original_bins)
        # print('The length of original bins is: ', len(original_bins))
        del original_bins[min_chi_square_index + 1]
        if min_chi_square_index == 0:
            chi_square_list[min_chi_square_index + 1] = calculate_chi_square(df, original_bins, min_chi_square_index)
        elif min_chi_square_index == len(chi_square_list) - 1:
            chi_square_list[min_chi_square_index - 1] = calculate_chi_square(df, original_bins,
                                                                             min_chi_square_index - 1)
        else:
            chi_square_list[min_chi_square_index - 1] = calculate_chi_square(df, original_bins,
                                                                             min_chi_square_index - 1)
            chi_square_list[min_chi_square_index + 1] = calculate_chi_square(df, original_bins, min_chi_square_index)
        del chi_square_list[min_chi_square_index]

        if min_chi_square > THRESHOLD or len(original_bins) <= LIMIT_NUM:
            break

    result_data = pd.DataFrame()
    list_temp = []  # 创建一个空白的分组列
    list_bad_num = []
    list_good_num = []
    # bad_rate = []
    bad_attr = []
    good_attr = []
    woe = []
    i = 0
    iv_fine = 0
    feature_series = pd.Series(df['X'])
    test_feature_series = pd.Series(df_test['X'])

    # print(feature_series)
    while i < len(original_bins) - 1:
        list_temp.append(str(original_bins[i]) + ',' + str(original_bins[i + 1]))
        new_cut = regroup[(regroup['X'] >= original_bins[i]) & (regroup['X'] < original_bins[i + 1])]
        # print('new cut\n', new_cut)
        # list_num.append(num)
        bad_num = new_cut['bad'].sum()
        if bad_num == 0:
            bad_num = 1
        list_bad_num.append(str(bad_num))

        good_num = new_cut['good'].sum()
        if good_num == 0:
            good_num = 1
        list_good_num.append(str(good_num))

        # bad_rate.append(str(bad_num / total))
        bad_attr.append(str(bad_num / bad))
        badatt = bad_num / bad
        good_attr.append(str(good_num / good))
        goodatt = good_num / good
        woe_value = np.log(goodatt / badatt)

        feature_series[(feature_series >= original_bins[i]) & (feature_series < original_bins[i + 1])] = woe_value

        test_feature_series[
            (test_feature_series >= original_bins[i]) & (test_feature_series < original_bins[i + 1])] = woe_value
        woe.append(str(woe_value))
        minus_value = goodatt - badatt
        iv_1 = minus_value * woe_value
        iv_fine = iv_fine + iv_1
        i += 1
    # print(feature_series)
    result_data['bin'] = list_temp
    result_data['bad'] = list_bad_num
    result_data['good'] = list_good_num
    # result_data['bad_rate'] = bad_rate  # 计算每个箱体坏样本所占总样本比例
    result_data['badattr'] = bad_attr  # 计算每个箱体坏样本所占坏样本总数的比例
    result_data['goodattr'] = good_attr  # 计算每个箱体好样本所占好样本总数的比例
    result_data['woe'] = woe  # 计算每个箱体的woe值
    # iv =  # 计算每个变量的iv值
    # print('分箱结果:')
    # print(result_data)
    result_data_woe = result_data[['bin', 'woe']]
    print('woe结果:')
    print(result_data_woe)

    final_pd = pd.DataFrame({"SeriousDlqin2yrs": df['Y'], col_name: feature_series})
    final_test_pd = pd.DataFrame({"SeriousDlqin2yrs": df_test['Y'], col_name: test_feature_series})
    # final_pd = pd.DataFrame({"SeriousDlqin2yrs": df['Y'], col_name: feature_series})
    # final_pd = final_pd.sort_values("SeriousDlqin2yrs")
    final_pd = final_pd.sort_index()
    final_test_pd = final_test_pd.sort_index()
    # final_pd = final_pd.sort_values("SeriousDlqin2yrs")
    print("训练集woe编码结果\n", final_pd[col_name])
    print("测试集woe编码结果\n", final_test_pd[col_name])

    # print('IV值为:')
    # print(iv_fine)
    # print(original_bins)
    list_new_woe = [woe]
    new_cut_woe_result = pd.DataFrame({'feature_name': col_name, 'bins_woe': list_new_woe})
    print(new_cut_woe_result)

    # print('The length of original bins is: ', len(original_bins))
    return final_pd[col_name], final_test_pd[col_name], new_cut_woe_result


# chi_square_merge_goes(x_test['age'].values, y_test, y_train, x_train['age'].values, col_name='age')
# df_new = pd.DataFrame({"Y": y_train})
# df_new = df_new.sort_index()
# df_new['age'] = get_column_value
# print(df_new)


df_new = pd.DataFrame({"SeriousDlqin2yrs": y_train})
df_new = df_new.sort_index()

df_new_test = pd.DataFrame({"SeriousDlqin2yrs": y_test})
df_new_test = df_new_test.sort_index()

df_new_woe_cut = pd.DataFrame()

cols = csv_data.columns[1:]
for column in cols:
    get_column_value, get_test_column, get_new_cut_woe_result = chi_square_merge_goes(x_test[column].values, y_test,
                                                                                      y_train, x_train[column].values,
                                                                                      col_name=column)
    df_new[column] = get_column_value
    df_new_test[column] = get_test_column
    df_new_woe_cut = df_new_woe_cut.append(get_new_cut_woe_result)
df_new.to_csv("train_WOE.csv")
df_new_test.to_csv("test_WOE.csv")
df_new_woe_cut.to_csv("df_new_woe_cut.csv")

print('train\n', df_new)
print('test\n', df_new_test)
print('woe_cut\n', df_new_woe_cut)

# chi_square_merge_goes(x_test['age'].values, y_test,
#                       y_train, x_train['age'].values, col_name='age')

# chi_square_merge_goes(y_train, x_train['age'].values,
#                       col_name='age')

# chi_square_merge_goes(y_train, x_train['NumberOfDependents'].values,
#                       col_name='NumberOfDependents')

# chi_square_merge_goes(y_train, x_train['MonthlyIncome'].values, col_name='MonthlyIncome')

# chi_square_merge_goes(y_train, x_train['RevolvingUtilizationOfUnsecuredLines'].values,
#                       col_name='RevolvingUtilizationOfUnsecuredLines')
#
# chi_square_merge_goes(y_train, x_train['NumberOfTime30-59DaysPastDueNotWorse'].values,
#                       col_name='NumberOfTime30-59DaysPastDueNotWorse')

# chi_square_merge_goes(y_train, x_train['DebtRatio'].values, col_name='DebtRatio')

# chi_square_merge_goes(y_train, x_train['NumberOfOpenCreditLinesAndLoans'].values,
#                       col_name='NumberOfOpenCreditLinesAndLoans')

# chi_square_merge_goes(y_train, x_train['NumberOfTimes90DaysLate'].values, col_name='NumberOfTimes90DaysLate')
#
# chi_square_merge_goes(y_train, x_train['NumberRealEstateLoansOrLines'].values,
#                       col_name='NumberRealEstateLoansOrLines')
#
# chi_square_merge_goes(y_train, x_train['NumberOfTime60-89DaysPastDueNotWorse'].values,
#                       col_name='NumberOfTime60-89DaysPastDueNotWorse')
