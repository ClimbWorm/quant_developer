文件夹说明：
two_week_adjusted 调整频率为2周
month_adjusted调整频率为1个月
season_adjusted调整频率为一个季度
adjusted的内容为lots和profit

stat_evaluation_result专门存放回测结果的评估表格


文件名称解读：
best_avg_pnl_table_half_month_short.csv 表示拼接完成的做空的订单 ，按照每半个月的最优参数调整，最优的评估标准是avg，也就是平均每笔交易的profit
best_total_pnl_table_half_month_short.csv表示拼接完成的做空的订单 ，按照每半个月的最优参数调整，最优的评估标准是total，也就是每个月的total profit

table_2.0_53.0_2.1.csv 命名中的参数依次为zzg_num,ema_num,multiplier
zzg_num有[0.382, 0.5, 0.618, 0.782, 0.886, 1.0, 1.236, 1.5, 2.0]