import time
import np as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, accuracy_score, \
    precision_score, recall_score
from sklearn.svm import SVC

CSV_FILE_PATH = "train_WOE.csv"
train_data = pd.read_csv(CSV_FILE_PATH, index_col=0)

CSV_FILE_PATH_1 = "test_WOE.csv"
test_data = pd.read_csv(CSV_FILE_PATH_1, index_col=0)

# 训练集
# 只选取IV大于0.1的特征
x = train_data.loc[:, ['RevolvingUtilizationOfUnsecuredLines', 'age',
                       'NumberOfTime30-59DaysPastDueNotWorse',
                       'NumberOfTimes90DaysLate',
                       'NumberOfTime60-89DaysPastDueNotWorse']]
y = 1 - train_data['SeriousDlqin2yrs']

# 测试集
# 只选取IV大于0.1的特征
x_test = test_data.loc[:, ['RevolvingUtilizationOfUnsecuredLines', 'age',
                           'NumberOfTime30-59DaysPastDueNotWorse',
                           'NumberOfTimes90DaysLate',
                           'NumberOfTime60-89DaysPastDueNotWorse']]
y_test = 1 - test_data['SeriousDlqin2yrs']
# print(np.array(y_true))
# print(x)
# print(y)
# 建立模型，并使用训练集进行模型训练
# 逻辑回归
clf = SVC(C=1.0, class_weight='balanced', gamma='scale', kernel="linear", probability=True)
# clf = LogisticRegression(C=10, class_weight='balanced')
print('Svm')
start = time.process_time()
clf.fit(x, y)

end = time.process_time()
print('Running time: %s Seconds' % (end - start))

# 特征权值系数，后面转换为打分规则时会用到
coe = clf.coef_

# y_score = clf.oob_score(x_test)
# print('y score: ', y_score)
# print(clf)
# 获得变量权重
# print('变量权重：', clf.coef_)

# 使用测试集进行模型预测
y_predict = clf.predict(x_test)
# print('预测模型：', y_predict)

y_score = clf.predict_proba(x_test)[:, 1]
# probs = clf.decision_function(x_test)
# y_score = (probs - probs.min()) / (probs.max() - probs.min())
# print('y_score: ', y_score)
# print('模型score: ', clf.score(x_test, y_test))

print('Accuracy: ', accuracy_score(y_test, y_predict))

print('Precision: ', precision_score(y_test, y_predict, average=None)[0])

print('Recall: ', recall_score(y_test, y_predict, average=None)[0])

new_pd = pd.DataFrame({'y_test': y_test, 'y_pre': y_predict})


# new_pd.to_csv('pre_and_test.csv')
# print(new_pd)


def compute_score(series, cut, score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list


# 绘制PR图
def draw_PR(Y_test, Y_score):
    plt.figure(1)  # 创建图表1
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label="Luck")  # 画对角线
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    Y_test = np.array(Y_test)
    precision, recall, thresholds = precision_recall_curve(np.array(Y_test), Y_score)
    # print('precision:', precision)
    # print('recall:', recall)
    # print('y_test', Y_test)
    # print('y_predict', Y_score)
    plt.figure(1)
    plt.plot(recall, precision)
    plt.show()


# draw_PR(y_test, y_score)


# 绘制ROC图

def draw_ROC_curve(Y_test, Y_score):
    Y_test = np.array(Y_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_score)
    print('y test:', Y_test)
    print('y_predict', Y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# draw_ROC_curve(y_test, y_score)

# 计算ks值
dff = pd.DataFrame({'score': y_predict, 'label': y_test})


def ks(df, y_true, y_pre, num=10, good=0, bad=1):
    # 1.将数据从小到大平均分成num组
    df_ks = df.sort_values(y_pre).reset_index(drop=True)
    df_ks['rank'] = np.floor((df_ks.index / len(df_ks) * num) + 1)
    df_ks['set_1'] = 1
    # 2.统计结果
    result_ks = pd.DataFrame()
    result_ks['group_sum'] = df_ks.groupby('rank')['set_1'].sum()
    result_ks['group_min'] = df_ks.groupby('rank')[y_pre].min()
    result_ks['group_max'] = df_ks.groupby('rank')[y_pre].max()
    result_ks['group_mean'] = df_ks.groupby('rank')[y_pre].mean()
    # 3.最后一行添加total汇总数据
    result_ks.loc['total', 'group_sum'] = df_ks['set_1'].sum()
    result_ks.loc['total', 'group_min'] = df_ks[y_pre].min()
    result_ks.loc['total', 'group_max'] = df_ks[y_pre].max()
    result_ks.loc['total', 'group_mean'] = df_ks[y_pre].mean()
    # 4.好用户统计
    result_ks['good_sum'] = df_ks[df_ks[y_true] == good].groupby('rank')['set_1'].sum()
    result_ks.good_sum.replace(np.nan, 0, inplace=True)
    result_ks.loc['total', 'good_sum'] = result_ks['good_sum'].sum()
    result_ks['good_percent'] = result_ks['good_sum'] / result_ks.loc['total', 'good_sum']
    result_ks['good_percent_cum'] = result_ks['good_sum'].cumsum() / result_ks.loc['total', 'good_sum']
    # 5.坏用户统计
    result_ks['bad_sum'] = df_ks[df_ks[y_true] == bad].groupby('rank')['set_1'].sum()
    result_ks.bad_sum.replace(np.nan, 0, inplace=True)
    result_ks.loc['total', 'bad_sum'] = result_ks['bad_sum'].sum()
    result_ks['bad_percent'] = result_ks['bad_sum'] / result_ks.loc['total', 'bad_sum']
    result_ks['bad_percent_cum'] = result_ks['bad_sum'].cumsum() / result_ks.loc['total', 'bad_sum']
    # 6.计算ks值
    result_ks['diff'] = result_ks['bad_percent_cum'] - result_ks['good_percent_cum']
    # 7.更新最后一行total的数据
    result_ks.loc['total', 'bad_percent_cum'] = np.nan
    result_ks.loc['total', 'good_percent_cum'] = np.nan
    result_ks.loc['total', 'diff'] = result_ks['diff'].max()

    result_ks = result_ks.reset_index()

    return result_ks


def ks_curve(df, num=10):
    # 防止中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    ks_value = df['diff'].max()

    # 获取绘制曲线所需要的数据
    x_curve = range(num + 1)
    y_curve1 = [0] + list(df['bad_percent_cum'].values[:-1])
    y_curve2 = [0] + list(df['good_percent_cum'].values[:-1])
    y_curve3 = [0] + list(df['diff'].values[:-1])
    # 获取绘制ks点所需要的数据
    df_ks_max = df[df['diff'] == ks_value]
    x_point = [df_ks_max['rank'].values[0], df_ks_max['rank'].values[0]]
    y_point = [df_ks_max['bad_percent_cum'].values[0], df_ks_max['good_percent_cum'].values[0]]
    # 绘制曲线
    plt.title('KS Curve')
    plt.plot(x_curve, y_curve1, label='bad', linewidth=2)
    plt.plot(x_curve, y_curve2, label='good', linewidth=2)
    plt.plot(x_curve, y_curve3, label='diff', linewidth=2)
    # 标记ks
    plt.plot(x_point, y_point, label='ks - {:.2f}'.format(ks_value), color='r', marker='o', markerfacecolor='r',
             markersize=5)
    plt.scatter(x_point, y_point, color='r')
    plt.legend()
    plt.show()

    return ks_value


def ks_calc_auc(true, score):
    # 功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    # score: 一维数组或series，代表模型得分（一般为预测正类的概率）
    # true: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    # 输出值:
    # 'ks': KS值
    fpr, tpr, thresholds = roc_curve(true, score)
    ks_value = max(tpr - fpr)
    print('ks: ', ks_value)
    return ks_value


ks_calc_auc(y_test, y_score)

# 建立评分卡
# print('coe:', coe)

# 假设好坏比为20的时候分数为600分，每高20分好坏比翻一倍
# 现在求每个变量不同woe值对应的分数刻度可得：

factor = 20 / np.log(2)
offset = 600 - 20 * np.log(20) / np.log(2)


def get_score(Coe, woe, Factor):
    scores = []
    for w in woe:
        score = round(Coe * w * Factor, 0)
        scores.append(score)
    return scores


x1 = get_score(coe[0][0], x_test['RevolvingUtilizationOfUnsecuredLines'], factor)
x2 = get_score(coe[0][1], x_test['age'], factor)
x3 = get_score(coe[0][2], x_test['NumberOfTime30-59DaysPastDueNotWorse'], factor)
x4 = get_score(coe[0][3], x_test['NumberOfTimes90DaysLate'], factor)
x5 = get_score(coe[0][4], x_test['NumberOfTime60-89DaysPastDueNotWorse'], factor)

# 每个用户的总分
scores = pd.DataFrame(
    {'RevolvingUtilizationOfUnsecuredLines': x1, 'age': x2, 'NumberOfTime30-59DaysPastDueNotWorse': x3,
     'NumberOfTimes90DaysLate': x4, 'NumberOfTime60-89DaysPastDueNotWorse': x5, 'scores': ''})
#
# print("Rev...UnsecuredLines:", x1)
# print("age:", x2)
# print("30-59...NotWorse:", x3)
# print("90DaysLate:", x4)
# # print("60-89...NotWorse:", x5)

scores['scores'] = scores['RevolvingUtilizationOfUnsecuredLines'] + scores['NumberOfTime30-59DaysPastDueNotWorse'] + \
                   scores['age'] + scores['NumberOfTimes90DaysLate'] + scores[
                       'NumberOfTime60-89DaysPastDueNotWorse'] + 600
scores.to_csv('all_scores.csv')
print(scores)
