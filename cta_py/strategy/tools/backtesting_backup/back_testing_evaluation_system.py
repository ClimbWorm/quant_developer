import numpy as np
import pandas as pd
import heapq

# 需要记录Idx_start和Idx_low（Idx_low是通过找Idx_start和Idx_end之间的最小值得到的）
# def return_evaluation_record_of_specific_params(zzg_num, ema_num,multiplier):
#     """统计最大回撤的前10个，以及出现最大回撤的位置在总的交易次数中的位置"""
#     # his_table = pd.read_csv(
#     #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
#     #         zzg_num, ema_num,multiplier))
#     his_table = pd.read_csv(
#         r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\best_avg_pnl_table.csv')
#     series = his_table.net_profit.cumsum()  # 是pandas的series类型
#     if series.iloc[-1] < 0:  ##考虑特殊情况，第一个即为最大值，事先写一个判断，如果曲线一直是向下的，就直接赋值一个-1或者别的常数值
#         raise Exception(f'({zzg_num},{ema_num},{multiplier})该参数组合最终亏损')
#
#     max_value = series.min()
#     Idx = []  # 记录出现新高的bar的index
#     margin_call_flag = False
#     for Idx_, item in enumerate(series):
#         if (item > max_value):
#             max_value = item
#             Idx.append(Idx_)
#         if item <= -2000:
#             margin_call_flag = True
#
#     Idx_start = Idx[:]
#     Idx_end = Idx[1:] + [len(series) - 1]
#     drawdown_list = []
#     drawdown_ratio_list = []
#     drawdown_last_times_list = []
#     # lowest_point_entrytime = []
#     location_percent_list = []
#     for start, end in zip(Idx_start, Idx_end):
#         if start < 0.1*len(series):
#             continue
#         if start == end:# 就是最后一个order产生的cumsum是新高
#             continue
#         previous_max = series[start]
#         min_ = series.iloc[start:end].min()
#         min_index_ = series.tolist()[start:end].index(min_) + start  # 获取两个peak之间的最小值的出现的第一个位置
#         location_percent_list.append(min_index_/len(series))# 最大的drawdown出现的位置占所有order的位置的百分比
#         # lowest_point_entrytime.append(his_table.loc[min_index_].EntryTime)
#         drawdown_last_times = min_index_ - start - 1
#         drawdown = previous_max - min_
#         drawdown_ratio = drawdown / previous_max
#         drawdown_list.append(drawdown)
#         drawdown_ratio_list.append(drawdown_ratio)
#         drawdown_last_times_list.append(drawdown_last_times)
#     top10_drawdown = heapq.nlargest(10,drawdown_list)  # 统计drawdown的最大数值，按照降序排列
#     top10_drawdown_index = map(drawdown_list.index,heapq.nlargest(10,drawdown_list))# 统计drawdown绝对数值最大的索引
#     drawdown_output_times = []
#     drawdown_output_percent = []
#     for i in top10_drawdown_index:
#         value_last_times = drawdown_last_times_list[i]
#         value_location_percent = location_percent_list[i]
#         drawdown_output_times.append(round(value_last_times,3))
#         drawdown_output_percent.append(round(value_location_percent,3))
#     top10_drawdown_ratio = heapq.nlargest(10,drawdown_ratio_list) # 统计drawdown ratio的最大数值，按照降序排列
#     top10_drawdown_ratio_index = map(drawdown_ratio_list.index,heapq.nlargest(10,drawdown_ratio_list))# 统计drawdown ratio最大的索引
#     drawdown_ratio_output_times = []
#     drawdown_ratio_output_percent = []
#     for i in top10_drawdown_ratio_index:
#         ratio_last_times = drawdown_last_times_list[i]
#         ratio_location_percent = location_percent_list[i]
#         drawdown_ratio_output_times.append(ratio_last_times)
#         drawdown_ratio_output_percent.append(ratio_location_percent)
#
#     total_pnl_std_month,avg_pnl_std_month = stat_pnl_std_by_month_freq(his_table)
#     total_pnl_std_percent,avg_pnl_std_month_percent = stat_pnl_std_by_percent_freq(his_table, percent=0.02)
#
#     return zzg_num,ema_num, multiplier,len(series),margin_call_flag,round(series.iloc[-1],1),round(series.iloc[-1]/len(series),1),[round(i,1) for i in top10_drawdown],round(top10_drawdown[0],1),drawdown_output_times,drawdown_output_percent,\
#            [round(i,3) for i in top10_drawdown_ratio],\
#            round(top10_drawdown_ratio[0],3),\
#            drawdown_ratio_output_times,\
#            [round(i,3) for i in drawdown_ratio_output_percent],\
#            int(total_pnl_std_month),\
#            int(avg_pnl_std_month),\
#            int(total_pnl_std_percent),\
#            int(avg_pnl_std_month_percent)

# def draw_pnl(zzg_num,multiplier):
#     import matplotlib.pyplot as plt
#     his_table = pd.read_csv(
#         r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\ha_rsi_all_added_ohlc_avg\table_add_rsi_{}_{}.csv'.format(
#             zzg_num, multiplier))
#     plt.plot(his_table.net_profit.cumsum())
#     plt.title(f"pnl_of_{zzg_num}_{multiplier}")
#     return plt


def stat_pnl_std_by_month_freq(his_table):
    his_table["month"] = his_table.EntryTime.apply(lambda x: x.split("-")[0] + x.split("-")[1])
    total_pnl_std = his_table.groupby(['month'])['net_profit'].sum().std()
    avg_pnl_std = his_table.groupby(['month'])['net_profit'].mean().std()
    return total_pnl_std,avg_pnl_std


def stat_pnl_std_by_percent_freq(his_table,percent=0.02):
    chunk_len = int(len(his_table) * percent)
    from more_itertools import chunked
    total_pnl = [sum(x) for x in chunked(his_table.net_profit, chunk_len)]
    avg_pnl = [sum(x)/len(x) for x in chunked(his_table.net_profit,chunk_len)]
    total_pnl_std = np.std(total_pnl)
    avg_pnl_std = np.std(avg_pnl)
    return total_pnl_std,avg_pnl_std


# 生成36行，2列的
def draw_pnl():
    import matplotlib.pyplot as plt
    i = 1
    plt.figure(figsize=(20,150))
    for zzg_num in [0.382, 0.5, 0.618, 0.782, 0.886, 1,1.236, 1.5, 2]:
        for multiplier in [0.382, 0.5, 0.618, 0.782, 0.886, 1,1.236, 1.5, 2]:
            try:
                his_table = pd.read_csv(
                    r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\ha_rsi_all_added_ohlc_avg\table_add_rsi_{}_{}.csv'.format(
                        zzg_num, multiplier))
                plt.subplot(36,2,i)
                plt.plot(his_table.net_profit.cumsum())
                plt.title(f"pnl_of_{zzg_num}_{multiplier}")
                i += 1
            except Exception as e:
                print(e)
    plt.savefig(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\pnl.jpg')


# def return_drawdown(his_table,ordertype):

def return_win_ratio(his_table):
    win_ratio = len(his_table.loc[his_table.net_profit >= 0])/len(his_table.loc[his_table.net_profit < 0]) # 盈亏比
    return win_ratio

# 先按照这个函数跑出来的乘数处理完表，然后把表传到return_evaluation_distinguish_long_short_orders中运行
# 这个函数只适用于对整个表做处理，不适用于需要筛选具体的season和month算乘数
def connection_multiplier(zzg_num,ema_num,multiplier):
    # lots排序：only_ha_without_lots > only_ha_with_lots > rsi_added_with_lots，所以以完全没有优化的为参考基准
    df_only_ha_with_lots = pd.read_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(zzg_num, ema_num, multiplier))
    df_rsi_added_with_lots = pd.read_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(zzg_num, ema_num, multiplier))
    ha_without_lots = len(df_only_ha_with_lots)
    ha_with_lots = df_only_ha_with_lots.Lots.sum()
    rsi_with_lots = df_rsi_added_with_lots.Lots.sum()
    lots_diff_multiplier = ha_without_lots/ha_with_lots # 未优化前和加了手数优化的乘数
    lots_rsi_diff_multiplier = ha_without_lots/rsi_with_lots # 加了rsi和手数优化的乘数
    return lots_diff_multiplier,lots_rsi_diff_multiplier

# # 下面是要写到main函数中运行的
# def generate_multiplier_plot_data(multiplier_type="lots",timeframe="season"):
#     """type:lots画的是没有filter只加了lots的
#     rsi画的是加入了rsi并且有lots的图
#
#     timeframe season按照季度统计
#     month按照月统计"""
#
#     plot_multiplier_between_none_and_lots_added = []
#     plot_multiplier_between_none_and_lots_rsi_added = []
#     #
#     if timeframe == "season":
#         for year in ['2015', '2016', '2017', '2018', '2019', '2020', '2021']:
#             for season in ['1', '2', '3', '4']:
#                 season_ = int(year + season)

def calibrate_his_table_lots_and_profits_by_lots_multiplier(timeframe="season"):
    """timeframe: season month"""
    if timeframe == "season":
        for zzg_num in [0.382, 0.5, 0.782, 1]:
            for ema_num in range(13, 57, 2):
                for multiplier in np.arange(0.1, 2.7, 0.2):
                    df_only_ha_with_lots = pd.read_csv(
                        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                            zzg_num, ema_num, multiplier))
                    df_rsi_added_with_lots = pd.read_csv(
                        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                            zzg_num, ema_num, multiplier))
                    df_season_adjusted = pd.DataFrame()
                    all_season_list = [int(year+season) for year in ['2015','2016','2017','2018','2019','2020','2021'] for season in ['1','2','3','4']]
                    for i in range(len(all_season_list)):
                        append_table_season = all_season_list[i]

                        if i == 0:
                            selected_season = all_season_list[i] # 用于计算乘数的月份

                        else:
                            selected_season = all_season_list[i-1]

                        if len(df_rsi_added_with_lots.loc[df_rsi_added_with_lots.season == append_table_season]) == 0:
                            break

                        lots_multiplier = len(
                            df_only_ha_with_lots.loc[df_only_ha_with_lots.season == selected_season]) / \
                                          df_rsi_added_with_lots.loc[
                                              df_rsi_added_with_lots.season == selected_season].Lots.sum()

                        append_table = df_rsi_added_with_lots.loc[df_rsi_added_with_lots.season == append_table_season]

                        # 需要修改的就只有Lots和net_profit
                        append_table["net_profit"] = append_table.net_profit * lots_multiplier
                        append_table["Lots"] = append_table.Lots * lots_multiplier

                        df_season_adjusted = pd.concat([df_season_adjusted, append_table], axis=0)
                    df_season_adjusted = df_season_adjusted.reset_index(drop=True)
                    df_season_adjusted.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out_season_adjusted\table_{}_{}_{}.csv'.format(
                            zzg_num, ema_num, multiplier))


    else:
        for zzg_num in [0.382, 0.5, 0.782, 1]:
            for ema_num in range(13, 57, 2):
                for multiplier in np.arange(0.1, 2.7, 0.2):
                    df_only_ha_with_lots = pd.read_csv(
                        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                            zzg_num, ema_num, multiplier))
                    df_rsi_added_with_lots = pd.read_csv(
                        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                            zzg_num, ema_num, multiplier))
                    df_month_adjusted = pd.DataFrame()
                    all_month_list = [int(year+month) for year in ['2015','2016','2017','2018','2019','2020','2021'] for month in ['1','2','3','4','5','6','7','8','9','10','11','12']]
                    for i in range(len(all_month_list)):
                        append_table_month = all_month_list[i]

                        if i == 0:
                            selected_month = all_month_list[i]

                        else:
                            selected_month = all_month_list[i-1]
                        if len(df_rsi_added_with_lots.loc[df_rsi_added_with_lots.month == append_table_month]) == 0:
                            break
                        lots_multiplier = len(
                            df_only_ha_with_lots.loc[df_only_ha_with_lots.month == selected_month]) / \
                                          df_rsi_added_with_lots.loc[
                                              df_rsi_added_with_lots.month == selected_month].Lots.sum()

                        append_table = df_rsi_added_with_lots.loc[df_rsi_added_with_lots.month == append_table_month]

                        # 需要修改的就只有Lots和net_profit
                        append_table["net_profit"] = append_table.net_profit * lots_multiplier
                        append_table["Lots"] = append_table.Lots * lots_multiplier

                        df_month_adjusted = pd.concat([df_month_adjusted, append_table], axis=0)
                    df_month_adjusted = df_month_adjusted.reset_index(drop=True)
                    df_month_adjusted.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out_month_adjusted\table_{}_{}_{}.csv'.format(
                            zzg_num, ema_num, multiplier))





# 终极版评估指标输出
def return_evaluation_distinguish_long_short_orders_or_all(zzg_num, ema_num, multiplier, ordertype="all"):
    "ordertype如果是all的话就是不分多空的 TradeDirection.SHORT TradeDirection.LONG"
    # his_table_all = pd.read_csv(
    #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out_month_adjusted\table_{}_{}_{}.csv'.format(
    #         zzg_num, ema_num, multiplier), index_col=0)
    # 参数为000时打开
    # his_table_all = pd.read_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out_season_adjusted\best_total_pnl_table_season.csv',index_col=0)
    # 参数为001时打开
    his_table_all = pd.read_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out_season_adjusted\best_avg_pnl_table_season.csv', index_col=0)

    if ordertype == "all":
        his_table_sub = his_table_all
    else:
        his_table_sub = his_table_all.loc[his_table_all.Direction == ordertype]

    total_lots = his_table_sub.Lots.sum()
    series = his_table_sub.net_profit.cumsum()
    df_cumsum_profits = pd.DataFrame(series)

    idx = []
    drawdown_list = []
    drawdown_ratio_list = []
    drawdown_last_times_list = []
    location_percent_list = []

    max_value = df_cumsum_profits.net_profit.min()
    margin_call_flag = False
    for i in df_cumsum_profits.index.tolist():  # range(len(df_cumsum_profits)):
        # print(df_cumsum_profits)
        # print(df_cumsum_profits.loc[i].net_profit)
        cum_pft, nth = df_cumsum_profits.loc[i].net_profit, i  # df_cumsum_profits.index.values[i]
        # print(cum_pft,nth)
        if cum_pft >= max_value:
            max_value = cum_pft
            idx.append(nth)

        if cum_pft <= -2000:
            margin_call_flag = True

    idx_start = idx[:]
    # print(zzg_num, ema_num, multiplier,df_cumsum_profits.index)
    idx_end = idx[1:] + [df_cumsum_profits.index[-1]]

    for start, end in zip(idx_start, idx_end):
        if start < 0.1 * len(df_cumsum_profits):  # 如果是前10%的交易，不计算drawdown，因为此时profit较小，会计算出较大的drawdown
            continue
        if start >= end:
            continue

        previous_max = df_cumsum_profits.loc[start].values[0]
        # print("previous_max: ",previous_max)
        sub_peak_chunk = df_cumsum_profits.loc[start:(end - 1)]
        # print("sub_peak_chunk: ",sub_peak_chunk)
        min_ = sub_peak_chunk.min().values[0]
        # print("min_: ", min_)

        min_index_ = sub_peak_chunk.loc[sub_peak_chunk.net_profit == min_].index.values[0]
        # print("min_index_: ",min_index_)

        location_percent_list.append(min_index_ / len(his_table_all))  # 在大表中的位置
        drawdown_last_times = min_index_ - start  # drawdown持续的时间

        drawdown = previous_max - min_
        drawdown_ratio = drawdown / previous_max
        drawdown_list.append(drawdown)
        drawdown_ratio_list.append(drawdown_ratio)
        drawdown_last_times_list.append(drawdown_last_times)

    top10_drawdown = heapq.nlargest(10, drawdown_list)
    # print("top10_drawdown: ",top10_drawdown)
    top10_drawdown_index = map(drawdown_list.index, heapq.nlargest(10, drawdown_list))
    # 下面的目的是按照给定顺序输出
    drawdown_output_times = []
    drawdown_output_percent = []
    for i in top10_drawdown_index:
        value_last_times = drawdown_last_times_list[i]
        value_location_percent = location_percent_list[i]
        drawdown_output_times.append(round(value_last_times, 3))
        drawdown_output_percent.append(round(value_location_percent, 3))
    top10_drawdown_ratio = heapq.nlargest(10, drawdown_ratio_list)  # 统计drawdown ratio的最大数值，按照降序排列
    top10_drawdown_ratio_index = map(drawdown_ratio_list.index,
                                     heapq.nlargest(10, drawdown_ratio_list))  # 统计drawdown ratio最大的索引

    drawdown_ratio_output_times = []
    drawdown_ratio_output_percent = []
    for i in top10_drawdown_ratio_index:
        ratio_last_times = drawdown_last_times_list[i]
        ratio_location_percent = location_percent_list[i]
        drawdown_ratio_output_times.append(ratio_last_times)
        drawdown_ratio_output_percent.append(ratio_location_percent)

    total_pnl_std_month, avg_pnl_std_month = stat_pnl_std_by_month_freq(his_table_sub)
    total_pnl_std_percent, avg_pnl_std_month_percent = stat_pnl_std_by_percent_freq(his_table_sub, percent=0.02)
    win_ratio = return_win_ratio(his_table_sub)

    return zzg_num, ema_num, multiplier, len(series), total_lots, margin_call_flag, win_ratio, round(series.iloc[-1], 1), round(
        series.iloc[-1] / total_lots, 1), [round(i, 1) for i in top10_drawdown], round(top10_drawdown[0],1), \
           drawdown_output_times, drawdown_output_percent, \
           [round(i, 3) for i in top10_drawdown_ratio], \
           round(top10_drawdown_ratio[0], 3), \
           drawdown_ratio_output_times, \
           [round(i, 3) for i in drawdown_ratio_output_percent], \
           int(total_pnl_std_month), \
           int(avg_pnl_std_month), \
           int(total_pnl_std_percent), \
           int(avg_pnl_std_month_percent)



if __name__ == '__main__':
    # a:drawdown的绝对数值前10，降序排列
    # b:最小值距离前peak的距离有多少单
    # c:最小值在所有的order中所在位置的百分比
    # d:drawdown ratio的绝对数值前10，降序排列
    # e:最小值距离前peak的距离有多少单
    # f:最小值在所有的order中所在位置的百分比
    # a,b,c,d,e,f = drawdown_amount_ratio_times(0.5, 0.5)
    # print(a,b,c,d,e,f)
    # drawdown_amount_ratio_times(0.5, 53, np.arange(0.1,2.7,0.2)[4])

    import csv
    import codecs
    file_csv = codecs.open(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out_season_adjusted\rsi_all_added_ema_differ_drawdown_stat_using_stopline_out_season_adjusted_short_only.csv','a+','utf-8')
    writer = csv.writer(file_csv)
    # writer.writerow(['zzg_num','ema_num','multiplier','orders','lots','margin_call','win_ratio','total_profit_point',
    #                  'avg_profit_point','drawdown','max_drawdown','times_peak','location_percent',
    #                  'drawdown_ratio','max_drawdown_ratio','times_peak','location_percent',"total_pnl_std_month",
    #                  "avg_pnl_std_month","total_pnl_std_percent","avg_pnl_std_month_percent"])

    rst = return_evaluation_distinguish_long_short_orders_or_all(0,0,1,ordertype="TradeDirection.SHORT")
    writer.writerow(rst)

    # for zzg_num in [0.382, 0.5, 0.782, 1]:
    #     for ema_num in range(13,57,2):
    #         for multiplier in np.arange(0.1,2.7,0.2):
    #             # debug状态
    #             # rst = return_evaluation_distinguish_long_short_orders_or_all(zzg_num, ema_num, multiplier,
    #             #                                                              ordertype="TradeDirection.SHORT")
    #             try:
    #                 rst = return_evaluation_distinguish_long_short_orders_or_all(zzg_num, ema_num, multiplier,
    #                                                                              ordertype="TradeDirection.SHORT")
    #                 writer.writerow(rst)
    #                 # for items in rst:
    #                 #     print(items)
    #                 #     writer.writerow(items)
    #             except Exception as e:
    #                 print("========================================",e,"==============================================")


    # for zzg_num in [0.382, 0.5, 0.782, 1]:
    #     for ema_num in range(13,57,2):
    #         for multiplier in np.arange(0.1,2.7,0.2):
    #             drawdown_amount_ratio_times(zzg_num,ema_num,multiplier)

    # rst = return_evaluation_distinguish_long_short_orders(0.782, 41,0.1, ordertype="TradeDirection.SHORT")
    # print(rst)

    # calibrate_his_table_lots_and_profits_by_lots_multiplier(timeframe="month")
