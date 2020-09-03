import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

CSV_FILE_PATH = "cs-data-delete-after-boxplot.csv"
data = pd.read_csv(CSV_FILE_PATH, index_col=0)

# 计算协方差
print(data.corr())
# print(data.corr('kendall'))
# print(data.corr('spearman'))

# 计算协方差
corr = data.corr()
sns.heatmap(corr,
            annot=True,
            annot_kws={'size': 5, 'weight': 'bold', 'color': 'blue'},
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='rainbow'
            )  # 画热力图

plt.xticks(rotation=90, size=4)    # 将字体进行旋转
plt.yticks(rotation=360, size=5)
plt.show()  # plt.show()
