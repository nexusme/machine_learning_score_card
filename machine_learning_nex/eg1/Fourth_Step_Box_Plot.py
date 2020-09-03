import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 对monthlyIncome 进行箱形图分析
from matplotlib.backends.backend_pdf import PdfPages

CSV_FILE_PATH = "cs-data-fill-select-age.csv"
data = pd.read_csv(CSV_FILE_PATH, index_col=0)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


#
#
def draw_box_plot(name):
    cols = data.columns[0:]
    plt.figure()
    plt.boxplot(data[name], sym='r*')
    plt.grid(True)
    plt.ylim(1000000, 200000)
    plt.show()


# pp = PdfPages("box_plot.pdf")
# cols = data.columns[0:]
# for column in cols:
#     plt.figure()
#     plt.boxplot(data[column], sym='r*')
#     # plt.ylim(0, 100000)
#     plt.grid(True)
#     plt.title(column)
#     pp.savefig()
# pp.close()


def box_plot_analysis(dt):
    # 上四分位数
    q3 = dt.quantile(q=0.75)
    # 下四分位数
    q1 = dt.quantile(q=0.25)
    # 四分位间距
    iqr = q3 - q1
    # 上限
    up = q3 + 1.5 * iqr

    print("上限:", end='')
    print(up)

    # 下限
    down = q1 - 1.5 * iqr
    print("下限:", end='')
    print(down)
    bool_result = (dt < up) & (dt > down)
    return bool_result


def three_sigma(dt):
    # 上限
    up = dt.mean() + 3 * dt.std()
    # 下线
    low = dt.mean() - 3 * dt.std()
    # 在上限与下限之间的数据是正常的

    bool_result = (dt < up) & (dt > low)
    return bool_result


# 异常值处理
def df_filter():
    df = pd.read_csv("cs-data-fill-select-age.csv", index_col=0)
    print(len(df))
    df_filtered = df[(df["age"] < 100) & (df["MonthlyIncome"] < 1000000)
                     & (df["RevolvingUtilizationOfUnsecuredLines"] < 16000) & (df["DebtRatio"] < 51000) &
                     (df["NumberOfTime30-59DaysPastDueNotWorse"] < 19)
                     & (df["NumberOfOpenCreditLinesAndLoans"] < 60) & (df["NumberOfTimes90DaysLate"] < 20) &
                     (df["NumberRealEstateLoansOrLines"] < 30) &
                     (df["NumberOfTime60-89DaysPastDueNotWorse"] < 17) &
                     (df["NumberOfDependents"] < 10.5)]
    print(len(df_filtered))
    df_filtered.to_csv('cs-data-delete-after-boxplot.csv')


df_filter()
