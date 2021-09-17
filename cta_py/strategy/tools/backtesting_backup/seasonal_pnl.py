import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing
from tools.Evaluation_indicators import trade_group_by_daytime_frame


def draw_seasonal_pic(timeperiod_list:list, zzg_num, fixed_stop_loss_num):
    his_table = pd.read_excel(
        '../code_generated_csv/900_zzg_num_and_stoploss_fixed/table_{}_{}.xlsx'.format(zzg_num,fixed_stop_loss_num),
        index_col=0)
    for i in range(1, len(timeperiod_list) + 1):
        # print(i)
        period = timeperiod_list[i - 1]

        sub_table = his_table[(pd.to_datetime(his_table.EntryTime) > pd.to_datetime(period[0])) & (
                pd.to_datetime(his_table.EntryTime) < pd.to_datetime(period[1]))]
        if len(sub_table) > 0:
            start_idx, end_idx = sub_table.index[0], sub_table.index[-1]
            # print(start_idx, end_idx)
            if i % 4 == 1:  # 第一季度
                plt.plot(np.arange(start_idx, end_idx + 1), sub_table.cumsum_Profits, color='green')
                plt.axvline(x=end_idx + 1)
            elif i % 4 == 2:
                plt.plot(np.arange(start_idx, end_idx + 1), sub_table.cumsum_Profits, color='red')
                plt.axvline(x=end_idx + 1)
            elif i % 4 == 3:
                plt.plot(np.arange(start_idx, end_idx + 1), sub_table.cumsum_Profits, color='yellow')
                plt.axvline(x=end_idx + 1)
            else:
                plt.plot(np.arange(start_idx, end_idx + 1), sub_table.cumsum_Profits, color='black')
                plt.axvline(x=end_idx + 1)
    # print("该保存图片了")
    plt.savefig(
        '../code_generated_csv/900_zzg_num_and_stoploss_fixed/seasonal_pnl_{}_{}.png'.format(zzg_num,
                                                                                             fixed_stop_loss_num))
    # print("图片保存完毕")


if __name__ == '__main__':

    # first_season_start = '/1/1'
    # second_season_start = '/4/1'
    # third_season_start = '/7/1'
    # fourth_season_start = '/10/1'
    # season_list = [first_season_start, second_season_start, third_season_start, fourth_season_start]
    # back_testing_year = ['2018', '2019', '2020', '2021']
    # date = []
    # for year in back_testing_year:
    #     for season in season_list:
    #         date.append(year + season)
    # timeperiod_list = []
    # for i, j in zip(date[:-1], date[1:]):
    #     timeperiod_list.append([i, j])
    # # print(timeperiod_list)
    #
    zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
    fixed_percent_list = np.arange(0.0001, 0.0052, 0.0002)
    addressList = []
    for num in zzg_num_list:
        for fixed_loss_percent in fixed_percent_list:
            addressList.append({"zzg_num": num, "loss_mode_4_percent_fixed": fixed_loss_percent})
    print('.' * 30, '开始运行', '.' * 30)

    pool = multiprocessing.Pool(10)
    i = 1
    for params in addressList:
        # 列表等参数要放在元组的最前面才可以，且下面因为不断绘图的原因，需要添加lock
        # pool.apply_async(draw_seasonal_pic, (timeperiod_list, list(params.values())[0], list(params.values())[1]))
        pool.apply_async(trade_group_by_daytime_frame, (list(params.values())[0], list(params.values())[1]))
        print('i: ', i)
        i += 1
    print('.' * 30, '程序正在进行......')
    pool.close()
    pool.join()
    print('.' * 30, '程序运行结束')
    # trade_group_by_daytime_frame(0.5, 0.0001)
