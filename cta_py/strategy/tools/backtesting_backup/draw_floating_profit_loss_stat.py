import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import threading


def floating_stat(zzg_num):
    hard_loss_num = np.arange(0.0005, 0.01, 0.00025)
    pnl_list = []
    profit_mean_list = []
    diff_mean_list = []
    std_profit_list = []
    std_diff_list = []
    for num in hard_loss_num:
        # his_table = pd.read_excel(
        #     'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/short_order_only/table_{}_{}.xlsx'.format(zzg_num, num), index_col=0)
        his_table = pd.read_csv(
            'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/mean_multiplier_added_integrate_csv/table_{}_{}.csv'.format(
                zzg_num, num), index_col=0)
        pnl = his_table.cumsum_Profits.iloc[-1]
        profit_mean = np.mean(his_table.max_profit)
        diff_mean = np.mean(his_table.max_profit - his_table.max_loss)
        std_profit = np.std(his_table.max_profit)
        std_diff = np.std(his_table.max_profit - his_table.max_loss)
        pnl_list.append(pnl)
        profit_mean_list.append(profit_mean)
        diff_mean_list.append(diff_mean)
        std_profit_list.append(std_profit)
        std_diff_list.append(std_diff)

    plt.figure(figsize=(20, 60))  # Todo 这句话好像没起作用
    fig, axs = plt.subplots(5)
    fig.suptitle('pnl & profit_mean & diff_mean & std_profit & std_diff')
    axs[0].plot(np.arange(0.0005, 0.01, 0.00025), pnl_list)
    axs[1].plot(np.arange(0.0005, 0.01, 0.00025), profit_mean_list)
    axs[2].plot(np.arange(0.0005, 0.01, 0.00025), diff_mean_list)
    axs[3].plot(np.arange(0.0005, 0.01, 0.00025), std_profit_list)
    axs[4].plot(np.arange(0.0005, 0.01, 0.00025), std_diff_list)
    fig.savefig(
        'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/mean_multiplier_added_integrate_csv/floating_stat_{}.png'.format(
            zzg_num))


# 根据每一个zzg num画出max floating profit、max（floating profit-floating loss）
# 是生成数据的函数
def max_floating_stat_in_daytime_frame(zzg_num, data_type=1):
    """
    type表示绘制的z轴是什么图
    1: 在不同时刻进场的每笔订单的profit的mean
    2：floating profit mean
    3: (floating profit - floating loss)的mean
    4: 在不同时刻进场的每笔订单的profit的std
    5：floating profit std
    6: (floating profit - floating loss)的std
    7: count的mean

    """
    hard_loss_num = np.arange(0.0005, 0.01, 0.00025)  # Todo 这边在用的时候需要改
    # hard_loss_num = pd.Series(hard_loss_num).apply(lambda x: '%.6s' % x)
    factor_value_list = []
    timeseries_list = []
    zzg_num_list = []
    for num in hard_loss_num:
        # his_table = pd.read_excel(
        #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\short_order_only_with_trailing_stop_multiplier\table_{}_{}.xlsx'.format(
        #         zzg_num, num), index_col=0)
        his_table = pd.read_csv(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\mean_multiplier_added_integrate_csv\table_{}_{}.csv'.format(
                zzg_num, num), index_col=0)
        df = pd.DataFrame(columns=['Time', 'Profits'])
        df['Time'] = his_table.EntryTime.apply(lambda x: x.split(" ")[-1])
        if (data_type == 1) | (data_type == 4):
            df['factor'] = his_table.Profits
        elif (data_type == 2) | (data_type == 5):
            df['factor'] = his_table.max_profit
        elif (data_type == 3) | (data_type == 6):
            df['factor'] = his_table.max_profit - his_table.max_loss
        else:  # 求count的均值
            # df['factor'] = his_table['count'] # 这里不能直接.count是因为count本来就是一个方法，会导致后面groupby的时候找不到可以groupby的数值类型的对象
            pass

        if (data_type == 1) | (data_type == 2) | (data_type == 3):  # | (data_type == 7):
            result_temp = df.groupby(['Time'])['factor'].mean()
        elif (data_type == 4) | (data_type == 5) | (data_type == 6):
            result_temp = df.groupby(['Time'])['factor'].std()
        else:
            pass
        result_temp.index = pd.to_datetime(result_temp.index)
        result = result_temp.sort_index()
        factor_value_list.extend(result.tolist())  # z
        timeseries = (np.arange(len(result)) / 4).tolist()
        # timeseries_list.extend(pd.Series(result.index).apply(lambda x: x.strftime("%H:%M:%S")))  # x非数字类型画三维图时会报错
        timeseries_list.extend(timeseries)  # x
        zzg_num_list.extend([num] * len(result))  # y

    return timeseries_list, zzg_num_list, factor_value_list


# 写一个3d绘制，x：trailing stop的步长  y：count取均值 z：profit mean?
def draw_3d_stop_count_profit(zzg_num):
    hard_loss_num = np.arange(0.0005, 0.01, 0.00025)  # Todo 这边在用的时候需要改
    # hard_loss_num = pd.Series(hard_loss_num).apply(lambda x: '%.6s' % x)
    trailing_stop_num_list = []
    count_list = []
    profits_mean_list = []
    for num in hard_loss_num:
        # his_table = pd.read_excel(
        #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\short_order_only_with_trailing_stop_multiplier\table_{}_{}.xlsx'.format(
        #         zzg_num, num), index_col=0)
        his_table = pd.read_csv(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\mean_multiplier_added_integrate_csv\table_{}_{}.csv'.format(
                zzg_num, num), index_col=0)
        df = pd.DataFrame(columns=['count', 'Profits'])

        df['count'] = his_table['count']  # 这里不能直接.count是因为count本来就是一个方法，会导致后面groupby的时候找不到可以groupby的数值类型的对象
        df['Profits'] = his_table.Profits
        z = df.groupby(['count'])['Profits'].mean().tolist()
        y = df.groupby(['count'])['Profits'].mean().index.tolist()  # Todo 顺序问题可能要考虑一下
        x = [num] * len(y)

        trailing_stop_num_list.extend(x)
        count_list.extend(y)
        profits_mean_list.extend(z)

    return trailing_stop_num_list, count_list, profits_mean_list

# x：日内的时间段 y:trailing stop num z:number_of_total_trades_generated
# def draw_3d_number_of_total_trades_generated(zzg_num):
#     hard_loss_num = np.arange(0.0005, 0.01, 0.00025)
#     for num in hard_loss_num:
#         his_table = pd.read_csv()
#         df = pd.DataFrame(columns=['Time','OrderType'])

def draw_3d_pic_about_profit_and_loss(zzg_num, data_type):#
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import MultipleLocator
    # lock = multiprocessing.Lock()
    # x, y, z = max_floating_stat_in_daytime_frame(zzg_num, data_type)
    x, y, z = draw_3d_stop_count_profit(zzg_num)
    fig = plt.figure(figsize=([42, 16]))

    gs = gridspec.GridSpec(2, 5, hspace=0.3)
    ax = {0: plt.subplot(gs[:, :3], projection='3d'),
          1: plt.subplot(gs[0, 3], projection='3d'),
          2: plt.subplot(gs[1, 3], projection='3d'),
          3: plt.subplot(gs[0, 4], projection='3d'),
          4: plt.subplot(gs[1, 4], projection='3d')}
    view = [(30, 80), (0, 90), (90, 0), (45, 45), (0, 0), ]


    # gs = gridspec.GridSpec(3, 3, hspace=0.4)
    # ax = {0: plt.subplot(gs[0:2, :], projection='3d'),
    #       1: plt.subplot(gs[2, 0], projection='3d'),
    #       2: plt.subplot(gs[2, 1], projection='3d'),
    #       3: plt.subplot(gs[2, 2], projection='3d')}
    # view = [(30, 80), (0, 90), (90, 0), (45, 45), ]
    # lock.acquire()
    if data_type != 7:
        xlabel = 'time'
        ylabel = 'zzg_num'
    else:
        xlabel = 'trailing_stop'
        ylabel = 'count_mean'


    if data_type ==1:
        zlabel = 'mean_profit'
    elif data_type == 2:
        zlabel = 'mean_floating_profit'
    elif data_type == 3:
        zlabel = '(floating profit - floating loss)\'s mean'
    elif data_type == 4:
        zlabel = 'profit_std'
    elif data_type == 5:
        zlabel = 'floating_profit_std'
    elif data_type == 6:
        zlabel = '(floating profit - floating loss)\'s std'
    else:
        zlabel = 'mean_profit'


    for i in range(5):
        ax[i].plot_trisurf(x, y, z, cmap=plt.cm.Spectral, linewidth=0.1)
        x_major_locator = MultipleLocator(0.00125)  # 若以时间作为x轴的，这边写1即可，以trailing stop作为x轴取0.00125
        ax[i].xaxis.set_major_locator(x_major_locator)
        ax[i].set_xlabel(f"{xlabel}")
        ax[i].set_ylabel(f"{ylabel}")
        ax[i].set_zlabel(f'{zlabel}')
        ax[i].view_init(*view[i])
    # Todo 下面这个地址在使用时要改
    fig.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\mean_multiplier_added_integrate_csv\3d\3d_{}_count.png'.format(
            zzg_num))
    plt.close('all')
    # lock.release()


def draw_x_ema_num_y_multiplier_z_total_pnl(zzg_num):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import MultipleLocator
    data_source = pd.read_csv(
        r"F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_all_added_ema_differ_drawdown_stat.csv")
    # print(data_source.columns)
    sub_data_source = data_source.loc[data_source.zzg_num == zzg_num]
    x,y,z = sub_data_source.ema_num,sub_data_source.multiplier,sub_data_source.total_profit_point

    fig = plt.figure(figsize=([42, 16]))
    gs = gridspec.GridSpec(2, 5, hspace=0.3)
    ax = {0: plt.subplot(gs[:, :3], projection='3d'),
          1: plt.subplot(gs[0, 3], projection='3d'),
          2: plt.subplot(gs[1, 3], projection='3d'),
          3: plt.subplot(gs[0, 4], projection='3d'),
          4: plt.subplot(gs[1, 4], projection='3d')}
    view = [(30, 80), (0, 90), (90, 0), (45, 45), (0, 0), ]
    for i in range(5):
        ax[i].plot_trisurf(x, y, z, cmap=plt.cm.Spectral, linewidth=0.1)
        x_major_locator = MultipleLocator(2)  # 若以时间作为x轴的，这边写1即可，以trailing stop作为x轴取0.00125
        ax[i].xaxis.set_major_locator(x_major_locator)
        ax[i].set_xlabel("ema_num")
        ax[i].set_ylabel("multiplier")
        ax[i].set_zlabel('total_pnl')
        ax[i].view_init(*view[i])
        if i == 0:
            ax[i].set_title(f"{zzg_num}:x_ema_num_y_multiplier_z_total_pnl")
    # Todo 下面这个地址在使用时要改
    fig.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\3d_rsi_all_added\3d_{}_total_pnl.png'.format(
            zzg_num))
    plt.close('all')


def draw_x_ema_num_y_multiplier_z_avg_pnl(zzg_num):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import MultipleLocator
    data_source = pd.read_csv(
        r"F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_all_added_ema_differ_drawdown_stat.csv")
    sub_data_source = data_source.loc[data_source.zzg_num == zzg_num]
    x, y, z = sub_data_source.ema_num, sub_data_source.multiplier, sub_data_source.avg_profit_point

    fig = plt.figure(figsize=([42, 16]))
    gs = gridspec.GridSpec(2, 5, hspace=0.3)
    ax = {0: plt.subplot(gs[:, :3], projection='3d'),
          1: plt.subplot(gs[0, 3], projection='3d'),
          2: plt.subplot(gs[1, 3], projection='3d'),
          3: plt.subplot(gs[0, 4], projection='3d'),
          4: plt.subplot(gs[1, 4], projection='3d')}
    view = [(30, 80), (0, 90), (90, 0), (45, 45), (0, 0), ]
    for i in range(5):
        ax[i].plot_trisurf(x, y, z, cmap=plt.cm.Spectral, linewidth=0.1)
        x_major_locator = MultipleLocator(2)  # 若以时间作为x轴的，这边写1即可，以trailing stop作为x轴取0.00125
        ax[i].xaxis.set_major_locator(x_major_locator)
        ax[i].set_xlabel("ema_num")
        ax[i].set_ylabel("multiplier")
        ax[i].set_zlabel('avg_pnl')
        ax[i].view_init(*view[i])
        if i == 0:
            ax[i].set_title(f"{zzg_num}:x_ema_num_y_multiplier_z_avg_pnl")
    # Todo 下面这个地址在使用时要改
    fig.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\3d_rsi_all_added\3d_{}_avg_pnl.png'.format(
            zzg_num))
    plt.close('all')


def draw_x_ema_num_y_multiplier_z_drawdown(zzg_num):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import MultipleLocator
    data_source = pd.read_csv(
        r"F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_all_added_ema_differ_drawdown_stat.csv")
    sub_data_source = data_source.loc[data_source.zzg_num == zzg_num]
    x, y, z = sub_data_source.ema_num, sub_data_source.multiplier, sub_data_source.max_drawdown

    fig = plt.figure(figsize=([42, 16]))
    gs = gridspec.GridSpec(2, 5, hspace=0.3)
    ax = {0: plt.subplot(gs[:, :3], projection='3d'),
          1: plt.subplot(gs[0, 3], projection='3d'),
          2: plt.subplot(gs[1, 3], projection='3d'),
          3: plt.subplot(gs[0, 4], projection='3d'),
          4: plt.subplot(gs[1, 4], projection='3d')}
    view = [(30, 80), (0, 90), (90, 0), (45, 45), (0, 0), ]
    for i in range(5):
        ax[i].plot_trisurf(x, y, z, cmap=plt.cm.Spectral, linewidth=0.1)
        x_major_locator = MultipleLocator(2)  # 若以时间作为x轴的，这边写1即可，以trailing stop作为x轴取0.00125
        ax[i].xaxis.set_major_locator(x_major_locator)
        ax[i].set_xlabel("ema_num")
        ax[i].set_ylabel("multiplier")
        ax[i].set_zlabel('max_drawdown')
        ax[i].view_init(*view[i])
        if i == 0:
            ax[i].set_title(f"{zzg_num}:x_ema_num_y_multiplier_z_max_drawdown")
    # Todo 下面这个地址在使用时要改
    fig.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\3d_rsi_all_added\3d_{}_max_drawdown.png'.format(
            zzg_num))
    plt.close('all')

def draw_x_ema_num_y_multiplier_z_drawdown_ratio(zzg_num):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import MultipleLocator
    data_source = pd.read_csv(
        r"F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_all_added_ema_differ_drawdown_stat.csv")
    sub_data_source = data_source.loc[data_source.zzg_num == zzg_num]
    x, y, z = sub_data_source.ema_num, sub_data_source.multiplier, sub_data_source.max_drawdown_ratio

    fig = plt.figure(figsize=([42, 16]))
    gs = gridspec.GridSpec(2, 5, hspace=0.3)
    ax = {0: plt.subplot(gs[:, :3], projection='3d'),
          1: plt.subplot(gs[0, 3], projection='3d'),
          2: plt.subplot(gs[1, 3], projection='3d'),
          3: plt.subplot(gs[0, 4], projection='3d'),
          4: plt.subplot(gs[1, 4], projection='3d')}
    view = [(30, 80), (0, 90), (90, 0), (45, 45), (0, 0), ]
    for i in range(5):
        ax[i].plot_trisurf(x, y, z, cmap=plt.cm.Spectral, linewidth=0.1)
        x_major_locator = MultipleLocator(2)  # 若以时间作为x轴的，这边写1即可，以trailing stop作为x轴取0.00125
        ax[i].xaxis.set_major_locator(x_major_locator)
        ax[i].set_xlabel("ema_num")
        ax[i].set_ylabel("multiplier")
        ax[i].set_zlabel('max_drawdown_ratio')
        ax[i].view_init(*view[i])
        if i == 0:
            ax[i].set_title(f"{zzg_num}:x_ema_num_y_multiplier_z_max_drawdown_ratio")
    # Todo 下面这个地址在使用时要改
    fig.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\3d_rsi_all_added\3d_{}_max_drawdown_ratio.png'.format(
            zzg_num))
    plt.close('all')


if __name__ == '__main__':
    # zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
    # 绘制三维图
    # for num in zzg_num_list:
    #     for tp in [1, 2, 3, 4, 5, 6]:
    #         draw_3d_pic_about_profit_and_loss(num, tp)

    # 绘制count的三维图
    # for num in zzg_num_list:
    #     for tp in [7]:
    #         draw_3d_pic_about_profit_and_loss(num, tp)

    # 绘制五张图
    # print("*" * 20, "程序开始执行", "*" * 20)
    # pool = multiprocessing.Pool(10)
    # i = 1
    # for item in zzg_num_list:
    #     rst = pool.apply_async(floating_stat, (item,))#
    #     rst.get()
    #     print('i: ', i)
    #     i += 1

    # 画三维图

    # for num in zzg_num_list:
    #     for tp in [1, 2, 3, 4, 5, 6]:
    #         rst = pool.apply_async(draw_3d_pic_about_profit_and_loss, (num, tp,))
    #         rst.get()
    #         print('i: ', i)
    #         i += 1
    # print('.' * 30, '程序正在进行......')
    # pool.close()
    # pool.join()
    # print('.' * 30, '程序运行结束')



    for zzg_num in [0.382, 0.5, 0.782, 1]:
        draw_x_ema_num_y_multiplier_z_total_pnl(zzg_num)
        draw_x_ema_num_y_multiplier_z_avg_pnl(zzg_num)
        draw_x_ema_num_y_multiplier_z_drawdown(zzg_num)
        draw_x_ema_num_y_multiplier_z_drawdown_ratio(zzg_num)

