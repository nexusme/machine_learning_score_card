import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

CSV_FILE_PATH = "cs-data-delete-after-boxplot.csv"
data = pd.read_csv(CSV_FILE_PATH, index_col=0)
ax = sns.heatmap(data, center=0)
plt.show()


def draw_hist(df, b, m, x, d):
    pp = PdfPages(df + "_hist.pdf")
    plt.figure()
    plt.hist(data[df], bins=b, range=(m, x), density=d)
    plt.grid(True)
    plt.title(df)
    pp.savefig()
    pp.close()

# draw_hist("age", 45, 0, 120, 1)
# draw_hist("MonthlyIncome", 50, 0, 20000, 1)
# draw_hist("NumberOfTime30-59DaysPastDueNotWorse", 3, 0, 3, 1)
# draw_hist("RevolvingUtilizationOfUnsecuredLines", 30, 0, 1, 1)
# draw_hist("SeriousDlqin2yrs", 2, 0, 1, 1)
# draw_hist("DebtRatio", 50, 0, 3, 1)
# draw_hist("NumberOfOpenCreditLinesAndLoans", 30, 0, 60, 1)
# draw_hist("NumberOfTimes90DaysLate", 100, 0, 100, 1)
# draw_hist("NumberRealEstateLoansOrLines", 60, 0, 60, 1)
# draw_hist("NumberOfTime60-89DaysPastDueNotWorse", 100, 0, 100, 1)
