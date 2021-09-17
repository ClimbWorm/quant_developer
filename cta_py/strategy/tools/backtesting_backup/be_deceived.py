import numpy as np
import pandas as pd

# Todo 这个文件貌似都不需要了
def integrate_long_and_short(zzg_num, trailing_stop_loss):
    # 拼接表格
    his_table_long = pd.read_excel(
        'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/long_order_only_with_trailing_stop_multiplier/mean_adjusted_trailing_stop/table_{}_{}.xlsx'.format(
            zzg_num, trailing_stop_loss), index_col=0)
    his_table_short = pd.read_excel(
        'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/short_order_only_with_trailing_stop_multiplier/mean_adjusted_trailing_stop/table_{}_{}.xlsx'.format(
            zzg_num, trailing_stop_loss), index_col=0)
    his_table = his_table_long.append(his_table_short)
    his_table = his_table.sort_values(by="entry_index", ascending=True)
    his_table = his_table.reset_index(drop=True)
    his_table.to_csv(
        'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/mean_multiplier_added_integrate_csv/table_{}_{}.csv'.format(
            zzg_num, trailing_stop_loss))
    return his_table


def count_be_deceived(zzg_num, trailing_stop_loss):
    his_table = pd.read_csv(
        'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/mean_multiplier_added_integrate_csv/table_{}_{}.csv'.format(
            zzg_num, trailing_stop_loss), index_col=0)
    count_list = []
    count = 0
    current_ordertype = his_table.OrderType[0]
    for i in range(len(his_table)):
        if his_table.OrderType[i] != current_ordertype:
            count = 1
            current_ordertype = his_table.OrderType[i]
        else:
            count += 1
        count_list.append(count)
    his_table['count'] = count_list
    # 这边顺带更新cumsum_profit
    his_table['cumsum_Profits'] = np.cumsum(his_table.Profits)
    his_table.to_csv(
        'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/mean_multiplier_added_integrate_csv/table_{}_{}.csv'.format(
            zzg_num, trailing_stop_loss))


if __name__ == '__main__':
    zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
    hard_loss_num = np.arange(0.0005, 0.01, 0.00025)
    for i in zzg_num_list:
        for j in hard_loss_num:
            # integrate_long_and_short(i, j)
            count_be_deceived(i, j)

