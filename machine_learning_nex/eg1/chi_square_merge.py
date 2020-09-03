import numpy as np
import pandas as pd
from numpy.ma import sqrt
# 读文件

from sklearn.model_selection import train_test_split

CSV_FILE_PATH = "cs-data-delete-after-boxplot.csv"
csv_data = pd.read_csv(CSV_FILE_PATH, index_col=0)

cols = list(csv_data)
cols.insert(0, cols.pop(cols.index('SeriousDlqin2yrs')))
csv_data = csv_data.loc[:, cols]
csv_data.rename(columns={'SeriousDlqin2yrs': 'y'}, inplace=True)
x_train, x_test, Y_train, y_test = train_test_split(csv_data, csv_data['y'], test_size=0.33)

print(Y_train)


def chiMerge(y_train, x_column):
    # y_train = 1 good
    # y_train = 0 bad

    # 卡方分箱阈值
    THRESHOLD = 10000
    # 分箱数目限制
    MIN_BOX = 8

    df = pd.DataFrame({"X": x_column, "Y": y_train})

    df = df.sort_values("X")

    ori_bins = sorted(set(df["X"].values))
    ori_bins.append(np.inf)
    # 将相邻区间Y值全为0或者全为1的区间合并
    ori_bins = chi_merge_process(ori_bins, df)

    # 初始化计算相邻区间的卡方值
    chi_values_list = get_chi_square_values_list(df, ori_bins)
    print(chi_values_list)
    # 循环合并
    while True:

        print("chi merge: current box is ", len(ori_bins) - 1)
        min_chi_square_value = min(chi_values_list)
        min_chi_square_index = chi_values_list.index(min_chi_square_value)
        print("min index is {}, len(ori_bins)={}, len(chi_values_list)={}".format(min_chi_square_index, len(ori_bins),
                                                                                  len(chi_values_list)))
        del ori_bins[min_chi_square_index + 1]
        if min_chi_square_index == 0:
            chi_values_list[min_chi_square_index + 1] = cal_chi_square_value(df, ori_bins, min_chi_square_index)
        elif min_chi_square_index == len(chi_values_list) - 1:
            chi_values_list[min_chi_square_index - 1] = cal_chi_square_value(df, ori_bins, min_chi_square_index - 1)
        else:
            chi_values_list[min_chi_square_index - 1] = cal_chi_square_value(df, ori_bins, min_chi_square_index - 1)
            chi_values_list[min_chi_square_index + 1] = cal_chi_square_value(df, ori_bins, min_chi_square_index)
        del chi_values_list[min_chi_square_index]

        if min_chi_square_value > THRESHOLD or len(ori_bins) <= MIN_BOX:
            break
    print(ori_bins)
    return ori_bins


def chi_merge_process(ori_bins, df):
    i = 0
    while i < len(ori_bins) - 2:
        df_adjcent = df[(df['X'] >= ori_bins[i]) & (df['X'] < ori_bins[i + 2])]
        if len(df_adjcent[df_adjcent["Y"] == 0]) == 0 or len(df_adjcent[df_adjcent["Y"] == 1]) == 0:
            del ori_bins[i + 1]
            i -= 1
        i += 1
    return ori_bins


def get_chi_square_values_list(df, ori_bins):
    chi_square_values_list = []
    for i in range(len(ori_bins) - 2):
        chi_square_value = cal_chi_square_value(df, ori_bins, i)
        chi_square_values_list.append(chi_square_value)
    return chi_square_values_list


def cal_chi_square_value(df, bins, index_first):
    box_1 = df[(df['X'] >= bins[index_first]) & (df['X'] < bins[index_first + 1])]
    box_2 = df[(df['X'] >= bins[index_first + 1]) & (df['X'] < bins[index_first + 2])]
    a_11 = len(box_1[box_1["Y"] == 0])
    a_12 = len(box_1[box_1["Y"] == 1])
    a_21 = len(box_2[box_2["Y"] == 0])
    a_22 = len(box_2[box_2["Y"] == 1])
    a_list = [a_11, a_12, a_21, a_22]

    r1 = a_11 + a_12
    r2 = a_21 + a_22
    c1 = a_11 + a_21
    c2 = a_12 + a_22
    N = r1 + r2

    e_11 = r1 * c1 / N
    e_12 = r1 * c2 / N
    e_21 = r2 * c1 / N
    e_22 = r2 * c2 / N
    e_list = [e_11, e_12, e_21, e_22]

    # print("a_11 = {}, a_12 = {}, a_21 = {}, a_22 = {}".format(a_11, a_12, a_21, a_22))
    # print("r1 = {}, r2 = {}, c1 = {}, c2 = {}, N = {}, N = {}".format(r1, r2, c1, c2, r1+r2, c1+c2))
    chi_square_value = 0
    for k in range(len(a_list)):
        chi_square_value += (a_list[k] - e_list[k]) ** 2 / e_list[k]
    return chi_square_value


chiMerge(Y_train, csv_data['MonthlyIncome'])
