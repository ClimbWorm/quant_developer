import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 只有ha，没有rsi的不加手数的情况
# 下面的参数组合包含
# zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1]
# multiplier_list = [1.236, 1.5, 2]
# zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1,1.236,1.5,2]
# multiplier_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1]
def pic_compare_pnl(zzg_num, open_threshold_multiplier):
    # 下面这个表是有概率分布的手数的表，没有rsi的filter
    his_table_without_rsi_filter = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\final_version_only_ha_ema21_trailing_stop\table_{}_{}.csv'.format(
            zzg_num, open_threshold_multiplier))

    # 统计出平均的开仓手数
    avg_lots = his_table_without_rsi_filter.Lots.mean()

    # 统计出新的pnl
    without_lots_distribution_profit = [(his_table_without_rsi_filter.loc[i].ExitPrice - his_table_without_rsi_filter.loc[i].EntryPrice) * avg_lots - 3
           if his_table_without_rsi_filter.loc[i].Direction == "TradeDirection.LONG"
           else (his_table_without_rsi_filter.loc[i].EntryPrice - his_table_without_rsi_filter.loc[i].ExitPrice) * avg_lots - 3
           for i in range(len(his_table_without_rsi_filter))]
    without_lots_distribution_pnl = pd.Series(without_lots_distribution_profit).cumsum()

    # 给dataframe添加year和month列,按季度统计出总共的开仓次数，组成一个list
    year_list = [his_table_without_rsi_filter.EntryTime[i].split("-")[0] for i in range(len(his_table_without_rsi_filter))]
    his_table_without_rsi_filter['year'] = year_list
    dropduplicates_year_list = list(set(year_list))
    year_list = sorted(dropduplicates_year_list, key=year_list.index)
    month_list = [his_table_without_rsi_filter.EntryTime[i].split("-")[1] for i in range(len(his_table_without_rsi_filter))]
    his_table_without_rsi_filter['month'] = month_list
    dropduplicates_month_list = list(set(month_list))
    month_list = sorted(dropduplicates_month_list, key=month_list.index)

    season_list = []
    for i in range(len(month_list) // 3):
        season_list.append(month_list[i * 3:i * 3 + 3])

    has_distribution_lots_list = []
    without_rsi_filter_order_num_list = [] # 后续按照从前到后的索引去取要标注在垂直线上的开仓数量
    for y in year_list:
        for s in season_list:
            order_num = len(his_table_without_rsi_filter.loc[his_table_without_rsi_filter.year.isin([y]) & his_table_without_rsi_filter.month.isin(s)])
            has_distribution_lots = his_table_without_rsi_filter.loc[his_table_without_rsi_filter.year.isin([y]) & his_table_without_rsi_filter.month.isin(s)].Lots.sum()
            without_rsi_filter_order_num_list.append(order_num)
            has_distribution_lots_list.append(has_distribution_lots)

    without_distribution_lots_list = (pd.Series(without_rsi_filter_order_num_list)*avg_lots).tolist()

    # 下面这个表是有概率分布的，有rsi filter的表
    his_table_with_rsi_filter = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\ha_rsi_all_added_ohlc_avg\table_add_rsi_{}_{}.csv'.format(
            zzg_num, open_threshold_multiplier))

    rsi_filter_added_lots_compensate_multiplier = his_table_without_rsi_filter.Lots.sum()/his_table_with_rsi_filter.Lots.sum()

    plt.figure(figsize=(200,20))
    plt.plot(without_lots_distribution_pnl,label='pnl_without_lots_distribution_without_rsi_filter',color='green',linewidth=2)
    plt.plot(his_table_without_rsi_filter.net_profit.cumsum(),label='pnl_with_lots_distribution_without_rsi_filter',color='blue',linewidth=2)
    plt.plot(his_table_without_rsi_filter.loc[his_table_without_rsi_filter.entry_idx.isin(his_table_with_rsi_filter.entry_idx)].index,
             his_table_with_rsi_filter.net_profit.cumsum()*rsi_filter_added_lots_compensate_multiplier,
             label='pnl_with_lots_distribution_with_rsi_filter',color='red',linewidth=2)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.0))

    his_table_with_rsi_filter['year'] = [his_table_with_rsi_filter.EntryTime[i].split("-")[0] for i in
                 range(len(his_table_with_rsi_filter))]
    his_table_with_rsi_filter['month'] = [his_table_with_rsi_filter.EntryTime[i].split("-")[1] for i in
                 range(len(his_table_with_rsi_filter))]

    # 统计加入rsi filter的有手数分布的按照季度统计的开仓次数和lots
    has_distribution_add_rsi_lots_list = []
    with_rsi_filter_order_num_list = []  # 后续按照从前到后的索引去取要标注在垂直线上的开仓数量
    for y in year_list:
        for s in season_list:
            order_num = len(his_table_with_rsi_filter.loc[his_table_with_rsi_filter.year.isin([y]) & his_table_with_rsi_filter.month.isin(s)])
            has_distribution_add_rsi_lots = his_table_with_rsi_filter.loc[his_table_with_rsi_filter.year.isin([y]) & his_table_with_rsi_filter.month.isin(s)].Lots.sum()*rsi_filter_added_lots_compensate_multiplier
            with_rsi_filter_order_num_list.append(order_num)
            has_distribution_add_rsi_lots_list.append(has_distribution_add_rsi_lots)

    # 获取到画垂直线的位置
    # loc到三月、六月、九月、十二月的order，然后选取出现的最后一个数的inx
    axvline_idx_list = []
    for y in year_list:
        for m in ['03','06','09','12']:
            axvline_idx = his_table_without_rsi_filter.loc[his_table_without_rsi_filter.year.isin([y]) & his_table_without_rsi_filter.month.isin([m])].index.max()
            axvline_idx_list.append(axvline_idx)


    # 画垂直线
    draw_distance = 1000
    for x,axvline_idx in enumerate(axvline_idx_list):
        if isinstance(axvline_idx,float):
            continue
        else:
            if x%4 == 0:#一季度绿色
                c = "green"
            elif x%4 == 1:#二季度
                c = "red"
            elif x%4 == 2:#三季度
                c = "yellow"
            else:# 四季度
                c = "black"

            plt.axvline(x=axvline_idx,color=c)
            plt.text(axvline_idx+1,-100+draw_distance*x,f"has_distribution_no_rsi_filter:    order:{without_rsi_filter_order_num_list[x]}    lots:{has_distribution_lots_list[x]}\n"
                                        f"no_distribution_no_rsi_filter:    order:{without_rsi_filter_order_num_list[x]}    lots:{without_distribution_lots_list[x]}\n"
                                        f"has_distribution_with_rsi_filter:    order:{with_rsi_filter_order_num_list[x]}    lots:{has_distribution_add_rsi_lots_list[x]}")

    plt.savefig(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\compare_pnl\compare_pnl_{}_{}.jpg'.format(zzg_num, open_threshold_multiplier))

if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool(9)
    zzg_num_list_part1 = [0.382, 0.5, 0.618, 0.782, 0.886, 1]
    multiplier_list_part1 = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
    zzg_num_list_part2 = [1.236,1.5,2]
    multiplier_list_part2 = [0.382, 0.5, 0.618, 0.782, 0.886, 1]

    # i = 1
    # for num in zzg_num_list_part1:
    #     for multiplier in multiplier_list_part1:
    #         pool.apply_async(pic_compare_pnl,(num,multiplier,))
    #         print("i: ",i)
    #         i += 1
    # pool.close()
    # pool.join()

    #
    j = 1
    for num in zzg_num_list_part2:
        for multiplier in multiplier_list_part2:
            pool.apply_async(pic_compare_pnl,(num,multiplier,))
            print("j: ",j)
            j += 1
    pool.close()
    pool.join()
