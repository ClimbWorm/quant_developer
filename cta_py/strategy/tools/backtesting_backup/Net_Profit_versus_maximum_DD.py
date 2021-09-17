import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from strategy.functionsforATRRegression import *
from tools.Evaluation_indicators import *


def main(zzg_num):
    params_list = np.arange(0.0001, 0.0052, 0.0002)
    # net_profit_versus_maximum_dd_list = []
    net_profit = []
    # max_DD_ratio = []
    for num in params_list:
        test_table = pd.read_excel(
            "../code_generated_csv/900_zzg_num_and_stoploss_fixed/table_{}_{}.xlsx".format(
                zzg_num, num),
            index_col=0)
        rst = test_table.cumsum_Profits.iloc[-1]
        net_profit.append(rst)

        # rst = calc_drawdown_ratio_list(test_table)
        # max_DD_ratio.append(rst)

    #     rst = Calc_net_profit_versus_maximum_DrawDown(test_table)
    #     net_profit_versus_maximum_dd_list.append(rst)
    # plt.plot(np.arange(0.0001, 0.0101, 0.0001), net_profit_versus_maximum_dd_list)
    plt.plot(np.arange(0.0001, 0.0052, 0.0002), net_profit)
    # plt.plot(np.arange(0.0001, 0.0101, 0.0001), max_DD_ratio)
    plt.savefig(
        '../code_generated_pic/900_zzg_num_and_stoploss_fixed/pnl_{}.png'.format(
            zzg_num))


if __name__ == '__main__':

    zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]

    pool = multiprocessing.Pool(9)
    i = 1

    for zzg_num in zzg_num_list:
        pool.apply_async(main, (zzg_num,))
        print('i: ', i)
        i += 1
    print('.' * 30, '程序正在进行......')
    pool.close()
    pool.join()
    print('.' * 30, '程序运行结束')
