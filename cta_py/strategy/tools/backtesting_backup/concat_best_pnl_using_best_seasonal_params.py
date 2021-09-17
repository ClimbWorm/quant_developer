import pandas as pd
import numpy as np

def concat_best_pnl_using_best_seasonal_params():

    max_total_pnl_params = []
    corresponding_lots_multiplier_total = []
    max_avg_pnl_params = []
    corresponding_lots_multiplier_avg = []

    for year in ['2015','2016','2017','2018','2019','2020','2021']:
        for season in ['1','2','3','4']:
            season_ = int(year + season)
            season_list = []
            total_pnl_of_different_params = []
            avg_pnl_of_different_params = []
            # 新增lots_multiplier
            lots_multiplier_list = []
            for zzg_num in [0.382, 0.5, 0.782, 1]:
                for ema_num in range(13, 57, 2):
                    for multiplier in np.arange(0.1, 2.7, 0.2):
                        his_table = pd.read_csv(
                            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                                zzg_num, ema_num, multiplier))
                        # 过渡的table，仅做计算乘数用途
                        trans_table = pd.read_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                zzg_num, ema_num, multiplier),index_col=0)
                        if len(his_table.loc[his_table.season == season_]) == 0:
                            break
                        total_pnl = his_table.loc[his_table.season == season_].net_profit.sum()
                        total_pnl_of_different_params.append(total_pnl)

                        lots_multiplier = len(trans_table.loc[trans_table.season == season_]) /his_table.loc[his_table.season == season_].Lots.sum()
                        lots_multiplier_list.append(lots_multiplier)



                        avg_pnl = his_table.loc[his_table.season == season_].net_profit.sum()/his_table.loc[his_table.season == season_].Lots.sum()
                        avg_pnl_of_different_params.append(avg_pnl)

                        season_list.append({'zzg_num': zzg_num, 'ema_num': ema_num, 'multiplier': multiplier})
            total_max_idx = total_pnl_of_different_params.index(max(total_pnl_of_different_params))
            avg_max_idx = avg_pnl_of_different_params.index(max(avg_pnl_of_different_params))
            max_total_pnl_params.append(season_list[total_max_idx])
            corresponding_lots_multiplier_total.append(lots_multiplier_list[total_max_idx])
            max_avg_pnl_params.append(season_list[avg_max_idx])
            corresponding_lots_multiplier_avg.append(lots_multiplier_list[avg_max_idx])

    print(corresponding_lots_multiplier_total,corresponding_lots_multiplier_avg)

    # 生成根据total pnl表现最佳的参数来确定下一个季度的参数拼接成的交易记录表
    best_total_pnl_table = pd.DataFrame()
        # (columns=['entry_idx', 'EntryTime', 'EntryPrice', 'Direction', 'exit_idx', 'ExitTime', 'ExitPrice', 'Lots',
        #              'multiplier', 'Commissions_and_slippage', 'net_profit', 'floating_profit',
        #              'floating_loss', 'high_bar', 'low_bar', 'last_zzp_value','season'])

    all_season_list = [year+season for year in ['2015','2016','2017','2018','2019','2020','2021'] for season in ['1','2','3','4']]

    for i in range(len(max_total_pnl_params)):
        season_to_select = int(all_season_list[i])
        if i == 0:  #开始交易的第一个季度
            params_dict = max_total_pnl_params[i]
            lots_multiplier_ = corresponding_lots_multiplier_total[i]
        else:
            params_dict = max_total_pnl_params[i-1] # 选上一个季度表现最好的参数
            lots_multiplier_ = corresponding_lots_multiplier_total[i-1]

        his_table_sub = pd.read_csv(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{zzg_num}_{ema_num}_{multiplier}.csv'.format(
                **params_dict),index_col=0)

        append_table = his_table_sub.loc[his_table_sub.season == season_to_select]
        # 需要修改的就只有Lots和net_profit
        append_table["net_profit"] = append_table.net_profit * lots_multiplier_
        append_table["Lots"] = append_table.Lots * lots_multiplier_

        # print(append_table)
        best_total_pnl_table = pd.concat([best_total_pnl_table,append_table],axis=0)
        # print(best_total_pnl_table)
    best_total_pnl_table = best_total_pnl_table.reset_index(drop=True)
    best_total_pnl_table.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\best_total_pnl_table_season.csv')


    # 生成根据avg pnl表现最佳的参数来确定下一个季度的参数拼接成的交易记录表
    best_avg_pnl_table = pd.DataFrame()
        # (columns=['entry_idx', 'EntryTime', 'EntryPrice', 'Direction', 'exit_idx', 'ExitTime', 'ExitPrice', 'Lots',
        #              'multiplier', 'Commissions_and_slippage', 'net_profit', 'floating_profit',
        #              'floating_loss', 'high_bar', 'low_bar', 'last_zzp_value','season'])

    for i in range(len(max_avg_pnl_params)):
        season_to_select = int(all_season_list[i])
        if i == 0:  #开始交易的第一个季度
            params_dict = max_avg_pnl_params[i]
            lots_multiplier_ = corresponding_lots_multiplier_avg[i]
        else:
            params_dict = max_avg_pnl_params[i-1] # 选上一个季度表现最好的参数
            lots_multiplier_ = corresponding_lots_multiplier_avg[i-1]

        his_table_sub = pd.read_csv(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{zzg_num}_{ema_num}_{multiplier}.csv'.format(
                **params_dict),index_col=0)
        append_table = his_table_sub.loc[his_table_sub.season == season_to_select]
        # 需要修改的就只有Lots和net_profit
        append_table["net_profit"] = append_table.net_profit * lots_multiplier_
        append_table["Lots"] = append_table.Lots * lots_multiplier_
        best_avg_pnl_table = pd.concat([best_avg_pnl_table,append_table],axis=0)
    best_avg_pnl_table = best_avg_pnl_table.reset_index(drop=True)
    best_avg_pnl_table.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\best_avg_pnl_table_season.csv')

    # return best_total_pnl_table

def concat_best_pnl_using_best_month_params():

    max_total_pnl_params = []
    corresponding_lots_multiplier_total = []
    max_avg_pnl_params = []
    corresponding_lots_multiplier_avg = []

    for year in ['2015','2016','2017','2018','2019','2020','2021']:
        for month in ['1','2','3','4','5','6','7','8','9','10','11','12']:
            month_ = int(year + month)
            month_list = []
            total_pnl_of_different_params = []
            avg_pnl_of_different_params = []
            # 新增lots_multiplier
            lots_multiplier_list = []
            for zzg_num in [0.382, 0.5, 0.782, 1]:
                for ema_num in range(13, 57, 2):
                    for multiplier in np.arange(0.1, 2.7, 0.2):
                        his_table = pd.read_csv(
                            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                                zzg_num, ema_num, multiplier))
                        # 过渡的table，仅做计算乘数用途
                        trans_table = pd.read_csv(
                            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                                zzg_num, ema_num, multiplier), index_col=0)
                        if len(his_table.loc[his_table.month == month_]) == 0:
                            break
                        total_pnl = his_table.loc[his_table.month == month_].net_profit.sum()
                        total_pnl_of_different_params.append(total_pnl)

                        lots_multiplier = len(trans_table.loc[trans_table.month == month_]) / his_table.loc[
                            his_table.month == month_].Lots.sum()
                        lots_multiplier_list.append(lots_multiplier)

                        avg_pnl = his_table.loc[his_table.month == month_].net_profit.sum()/his_table.loc[his_table.month == month_].Lots.sum()
                        avg_pnl_of_different_params.append(avg_pnl)

                        month_list.append({'zzg_num': zzg_num, 'ema_num': ema_num, 'multiplier': multiplier})

            total_max_idx = total_pnl_of_different_params.index(max(total_pnl_of_different_params))
            avg_max_idx = avg_pnl_of_different_params.index(max(avg_pnl_of_different_params))
            max_total_pnl_params.append(month_list[total_max_idx])
            corresponding_lots_multiplier_total.append(lots_multiplier_list[total_max_idx])
            max_avg_pnl_params.append(month_list[avg_max_idx])
            corresponding_lots_multiplier_avg.append(lots_multiplier_list[avg_max_idx])

    print(corresponding_lots_multiplier_total,corresponding_lots_multiplier_avg)

    # 生成根据total pnl表现最佳的参数来确定下一个季度的参数拼接成的交易记录表
    best_total_pnl_table = pd.DataFrame()
        # (columns=['entry_idx', 'EntryTime', 'EntryPrice', 'Direction', 'exit_idx', 'ExitTime', 'ExitPrice', 'Lots',
        #              'multiplier', 'Commissions_and_slippage', 'net_profit', 'floating_profit',
        #              'floating_loss', 'high_bar', 'low_bar', 'last_zzp_value','season'])

    all_month_list = [year+month for year in ['2015','2016','2017','2018','2019','2020','2021'] for month in ['1','2','3','4','5','6','7','8','9','10','11','12']]

    for i in range(len(max_total_pnl_params)):
        month_to_select = int(all_month_list[i])
        if i == 0:  #开始交易的第一个月
            params_dict = max_total_pnl_params[i]
            lots_multiplier_ = corresponding_lots_multiplier_total[i]
        else:
            params_dict = max_total_pnl_params[i-1] # 选上一个月表现最好的参数
            lots_multiplier_ = corresponding_lots_multiplier_total[i-1]

        his_table_sub = pd.read_csv(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{zzg_num}_{ema_num}_{multiplier}.csv'.format(
                **params_dict),index_col=0)
        append_table = his_table_sub.loc[his_table_sub.month == month_to_select]
        # 需要修改的就只有Lots和net_profit
        append_table["net_profit"] = append_table.net_profit * lots_multiplier_
        append_table["Lots"] = append_table.Lots * lots_multiplier_
        best_total_pnl_table = pd.concat([best_total_pnl_table,append_table],axis=0)
        # print(best_total_pnl_table)
    best_total_pnl_table = best_total_pnl_table.reset_index(drop=True)
    best_total_pnl_table.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\best_total_pnl_table_month.csv')


    # 生成根据avg pnl表现最佳的参数来确定下一个季度的参数拼接成的交易记录表
    best_avg_pnl_table = pd.DataFrame()
        # (columns=['entry_idx', 'EntryTime', 'EntryPrice', 'Direction', 'exit_idx', 'ExitTime', 'ExitPrice', 'Lots',
        #              'multiplier', 'Commissions_and_slippage', 'net_profit', 'floating_profit',
        #              'floating_loss', 'high_bar', 'low_bar', 'last_zzp_value','season'])

    for i in range(len(max_avg_pnl_params)):
        month_to_select = int(all_month_list[i])
        if i == 0:  #开始交易的第一个季度
            params_dict = max_avg_pnl_params[i]
            lots_multiplier_ = corresponding_lots_multiplier_avg[i]
        else:
            params_dict = max_avg_pnl_params[i-1] # 选上一个季度表现最好的参数
            lots_multiplier_ = corresponding_lots_multiplier_avg[i-1]

        his_table_sub = pd.read_csv(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\table_{zzg_num}_{ema_num}_{multiplier}.csv'.format(
                **params_dict),index_col=0)
        append_table = his_table_sub.loc[his_table_sub.month == month_to_select]
        # 需要修改的就只有Lots和net_profit
        append_table["net_profit"] = append_table.net_profit * lots_multiplier_
        append_table["Lots"] = append_table.Lots * lots_multiplier_

        best_avg_pnl_table = pd.concat([best_avg_pnl_table,append_table],axis=0)
    best_avg_pnl_table = best_avg_pnl_table.reset_index(drop=True)
    best_avg_pnl_table.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\best_avg_pnl_table_month.csv')



# 在拼表之前要先运行这个函数
def add_month_info_to_his_table():
    for zzg_num in [0.382, 0.5, 0.782, 1]:
        for ema_num in range(13,57,2):
            for multiplier in np.arange(0.1,2.7,0.2):
                his_table = pd.read_csv(
                    r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                        zzg_num, ema_num, multiplier))
                his_table["month"] = his_table.EntryTime.apply(lambda x: str(pd.to_datetime(x).year) + str(pd.to_datetime(x).month))
                his_table.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                        zzg_num, ema_num, multiplier))



# 在拼表之前要先运行这个函数
def add_season_info_to_his_table():
    for zzg_num in [0.382, 0.5, 0.782, 1]:
        for ema_num in range(13,57,2):
            for multiplier in np.arange(0.1,2.7,0.2):
                his_table = pd.read_csv(
                    r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                        zzg_num, ema_num, multiplier))
                his_table["season"] = his_table.EntryTime.apply(lambda x: str(pd.to_datetime(x).year) + str((pd.to_datetime(x).month - 1)//3 + 1))
                his_table.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out\table_{}_{}_{}.csv'.format(
                        zzg_num, ema_num, multiplier))


if __name__ == '__main__':
    add_season_info_to_his_table()
    print("season over......")
    add_month_info_to_his_table()

    # concat_best_pnl_using_best_seasonal_params()
    # print("拼接season的运行完了。。。。")
    # concat_best_pnl_using_best_month_params()


