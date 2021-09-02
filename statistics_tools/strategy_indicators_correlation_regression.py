import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt


path = r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out'


def regression_count_floating_profit(zzg_num,ema_num,multiplier):
    file_path = os.path.join(path, f'table_{zzg_num}_{ema_num}_{multiplier}.csv')
    his_table = pd.read_csv(file_path, index_col=0)
    his_table = his_table[~his_table.isin([np.nan, np.inf, -np.inf]).any(1)]
    # 修改下面就可以计算不同变量间的关系
    x = his_table.standardized_dayrange.values.reshape(-1,1)
    y = his_table.net_profit.values
    plt.scatter(his_table.standardized_dayrange.values,his_table.net_profit.values)
    plt.title(f"scatter_{zzg_num}_{ema_num}_{multiplier}")
    plt.show()
    model = LinearRegression()
    model = model.fit(x,y)
    # 验证模型的拟合度
    r_sq = model.score(x,y)
    print('coefficient of determination(𝑅²) :', r_sq)
    print('intercept: ', model.intercept_)
    print('slope: ', model.coef_)

if __name__ == '__main__':
    for zzg_num in [0.382, 0.5, 0.782, 1]:
        for ema_num in range(13, 57, 2):
            for multiplier in np.arange(0.1, 2.7, 0.2):

                regression_count_floating_profit(zzg_num, ema_num, multiplier)