#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 回测 画图
#
# Author: Tayii
# Data : 2021/2/8
# ----------------------------------------------------
import datetime
import time
from os import path
import logging
from typing import List, Union, Optional

import matplotlib
import mplfinance as mpf  # pip install --upgrade mplfinance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl  # 用于设置曲线参数
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import matplotlib.dates as mdates
from scipy.stats import norm

logging.getLogger('matplotlib.font_manager').disabled = True

from datahub.generate import Performance

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #解决负号“-”显示为方块的问题

def plot_candle(
        df: pd.DataFrame,  # K线数据
        ax=None,  # 画在哪个区域
):
    """绘制K线图/蜡烛图"""
    if ax is None:
        f, ax = plt.subplots(figsize=(16, 12))

    # 设置基本参数
    kwargs = dict(
        type='candle',  # 绘制图形的类型，有candle, renko, ohlc, line等
        # mav=(7, 30, 60),  # 均线类型,此处设置7,30,60日线
        # volume=True,  # 布尔类型，设置是否显示成交量，默认False
        title=f'\nA_stock candle_line', ylabel='OHLC Candles',
        ylabel_lower='Shares\nTraded Volume',  # 设置成交量图一栏的标题
        figratio=(15, 10),  # 设置图形纵横比
        figscale=3)  # 设置图形尺寸(数值越大图像质量越高)

    # 设置线宽
    mpl.rcParams['lines.linewidth'] = .5

    # 设置marketcolors
    mc = mpf.make_marketcolors(
        up='green',  # up:设置K线线柱颜色，up意为收盘价大于等于开盘价
        down='red',  # down:与up相反，这样设置与国内K线颜色标准相符
        edge='i',  # edge:K线线柱边缘颜色(i代表继承自up和down的颜色)，下同。详见官方文档)
        wick='i',  # wick:灯芯(上下影线)颜色
        volume='in',  # volume:成交量直方图的颜色
        inherit=True)  # inherit:是否继承，选填

    # 设置图形风格
    s = mpf.make_mpf_style(
        gridaxis='both',  # gridaxis:设置网格线位置
        gridstyle='-.',  # gridstyle:设置网格线线型
        y_on_right=False,  # y_on_right:设置y轴位置是否在右
        marketcolors=mc)

    # 设置均线颜色，配色表可见下图
    # 建议设置较深的颜色且与红色、绿色形成对比
    # 此处设置七条均线的颜色，也可应用默认设置
    # mpl.rcParams['axes.prop_cycle'] = cycler(
    #     color=['dodgerblue', 'deeppink',
    #            'navy', 'teal', 'maroon', 'darkorange',
    #            'indigo'])

    # print(123, df)

    mpf.plot(df, **kwargs, style=s,
             # savefig=,  # 保存图像
             )

    if ax is None:
        plt.show()


class PlotBtTotalChart:
    """
    绘图 总的回测结果
    """

    def __init__(self,
                 df: pd.DataFrame,  # 总的结果
                 dir: str  # 回测数据存放的文件夹
                 ):
        self.df = df
        self.dir = dir  # 回测数据存放的文件夹

    def plot_3d(self,
                x: str,  # z轴数据
                y: str,  # y轴数据
                z: str,  # z轴数据
                **kwargs,
                ):
        """
        绘制3D图
        多子图，一个数据一个图
        """
        fig = plt.figure(figsize=([42, 16]))
        title = kwargs.get('title', z)
        fig.suptitle(title, size=kwargs.get('title_size', 20))  # 全局标题

        gs = gridspec.GridSpec(2, 5, hspace=0.3)
        ax = {0: plt.subplot(gs[:, :3], projection='3d'),
              1: plt.subplot(gs[0, 3], projection='3d'),
              2: plt.subplot(gs[1, 3], projection='3d'),
              3: plt.subplot(gs[0, 4], projection='3d'),
              4: plt.subplot(gs[1, 4], projection='3d')}

        # ax = fig.gca(projection='3d')
        view = [(30, 80), (0, 90), (90, 0), (45, 45), (0, 0), ]
        for i in range(5):
            ax[i].plot_trisurf(self.df[x], self.df[y], self.df[z],
                               cmap=plt.cm.Spectral, linewidth=0.1,
                               )
            ax[i].set_xlabel(x)
            ax[i].set_ylabel(y)
            plt.setp(ax[i].get_yticklabels(), rotation=25, ha="right", rotation_mode="anchor")
            ax[i].set_zlabel(z)  # 给三个坐标轴注
            # 调整角度，第一个数字为上下，第二个数字为左右。
            ax[i].view_init(*view[i])

        if kwargs.get('need_show', None):
            plt.show()

        if kwargs.get('need_save', None):
            filename = f'{title}_3d.jpg'
            fig.savefig(path.join(self.dir, filename), dpi=200, bbox_inches='tight')

    def plot_multi_face_grid(self,
                             grid: List[str],  # 用于分类的列
                             x: str,  # x轴数据
                             y: Union[str, List[str]],  # y轴数据
                             **kwargs,
                             ) -> None:
        """
        打印 基于grid分类的 多个子图
        """
        if isinstance(y, str):
            y = [y]

        if len(grid) >= 2:
            g = sns.FacetGrid(self.df, col=grid[0], row=grid[1],
                              height=3.5, aspect=1.25, )
        elif len(grid) == 1:
            g = sns.FacetGrid(self.df, col=grid[0], height=4.5, aspect=1.1)
        else:
            print(f'没有grid1')
            return

        # 画子图
        markers = ['d', '.', '*', '+']
        for i in range(len(y)):
            # self.df[f'y_{i}'] = self.df[y[i]].apply(lambda a: '+' if a > 0 else '-')
            g.map_dataframe(sns.scatterplot,
                            x, y[i],)  #  hue=f'y_{i}', marker=markers[i], label=f'y_{i}'

        g.set_axis_labels(f'{x}', ' & '.join(y))

        if kwargs.get('need_show', None):
            plt.show()

        if kwargs.get('need_save', None):
            g.savefig(path.join(self.dir, f'grid_{x}_{"_".join(y)}.jpg'), dpi=200, bbox_inches='tight')

    def plot_multi_scatter(self,
                           x: str,  # x轴数据列
                           y_types: Union[List[str], str],  # y轴数据列 各子图类别
                           data: pd.DataFrame = None,  # 原始数据
                           **kwargs,
                           ) -> None:
        """
        打印 多个子图的 散点图
        """
        fig = plt.figure(figsize=([46, 12]))
        title: str = kwargs.get('title', None)
        if title:
            fig.suptitle(title, size=kwargs.get('title_size', 18))  # 全局标题

        # 图片包含几块 分别的尺寸以及比例
        size = len(y_types)
        subplot_h_num: int = kwargs.get('subplot_h_num', size)  # 纵向几列
        subplot_rows_num: int = (size // subplot_h_num if size % subplot_h_num == 0
                                 else size // subplot_h_num + 1)
        gs = gridspec.GridSpec(subplot_rows_num, subplot_h_num, wspace=0.2, hspace=0.2, )

        df = data or self.df

        for i in range(len(y_types)):
            # 数据处理
            df[f'y_{i}'] = df[y_types[i]].apply(lambda a: '+' if a > 0 else '-')

            ax = plt.subplot(gs[i // subplot_h_num, i % subplot_h_num])
            ax.set_title(y_types[i], fontsize=kwargs.get('title_size', 12))  # 子图标题
            ax.tick_params(which='major', axis='x', labelrotation=45, labelsize=9, length=5, pad=10)
            sns.scatterplot(x=x, y=y_types[i], data=df,
                            ax=ax,
                            label=y_types[i],
                            hue=f'y_{i}',
                            )

        if kwargs.get('need_show', None):
            plt.show()

        if kwargs.get('need_save', None):
            fig.savefig(path.join(self.dir, f'{title}_multi_scatter.jpg'),
                        dpi=200, bbox_inches='tight')

    def plot_multi_dis(self,
                       x: List[str],  # x轴数据
                       decide_open_type: str,  # 选择的类
                       show: bool = True,  # 是否显示
                       save: bool = False,  # 是否保存
                       ) -> None:
        """打印 多个子图的 分布图"""
        fig = plt.figure(figsize=([28, 55]))
        fig.suptitle(f'Result distplot : decide_open_type={decide_open_type}', size=15)  # 全局标题
        # 图片包含几块 分别的尺寸以及比例
        gs = gridspec.GridSpec(4, 6, hspace=0.4)

        df = self.df[self.df['decide_open_type'] == decide_open_type]
        for i in range(len(x)):
            ax = plt.subplot(gs[i // 6, i % 6])
            # ax.set_title(f'{x[i]}', fontsize=9)
            sns.distplot(df[x[i]], bins=100, fit=norm,
                         kde=True, ax=ax)

        if show:
            plt.show()

        # if save:
        #     filename = f'{y}_multi_scatter.jpg'
        #     fig.savefig(path.join(self.dir, filename), dpi=200, bbox_inches='tight')

    def plot_earnings(self,
                      grid: List[str],  #
                      x: str,  # x轴数据
                      y: Union[str, List[str]],  # y轴数据
                      show: bool = True,  # 是否显示
                      save: bool = False,  # 是否保存
                      ) -> None:
        """
        打印 多因子对 y数据（比如earnings）的影响
        """
        if isinstance(y, str):
            y = [y]

        for i in range(len(y)):
            self.df[f'y_{i}'] = self.df[y[i]].apply(lambda a: '+' if a > 0 else '-')

        # fig = plt.figure(figsize=([28, 56]))
        if len(grid) >= 2:
            g = sns.FacetGrid(self.df, col=grid[0], row=grid[1],
                              height=3.5, aspect=1.25, )
        elif len(grid) == 1:
            g = sns.FacetGrid(self.df, col=grid[0], height=4.5, aspect=1.1)
        else:
            print(f'没有grid1')
            return

        markers = ['d', '.', '*', '+']
        for i in range(len(y)):
            g.map_dataframe(sns.scatterplot,
                            x, y[i], hue=f'y{i}', marker=markers[i], label=f'y{i}')

        g.set_axis_labels(f'{x}', ' & '.join(y))

        if show:
            plt.show()

        if save:
            filename = f'grid_{"_".join(grid)}_x_{x}_y_{"_".join(y)}.jpg'
            g.savefig(path.join(self.dir, filename), dpi=200, bbox_inches='tight')


class PlotBtSingleChart():
    """
    绘图 单个回测结果
    """

    def __init__(self,
                 perf: Performance,  # 盈利情况，统计指标等
                 bt_dir: str,  # 回测结果存放目录
                 kline_data: pd.DataFrame = None,  # kline数据
                 ) -> None:
        self.perf = perf
        self.df_kline: pd.DataFrame = kline_data  # kline数据
        self.dir: str = bt_dir  # 回测结果存放目录

    def multi_subplot_heatmap(self,
                              sub_types: list,  # 各子图类别
                              **kwargs,
                              ):
        """多子图 画热力图"""
        fig = plt.figure(figsize=([30, 8]))
        title: str = kwargs.get('title', None)
        if title:
            fig.suptitle(title, size=kwargs.get('title_size', 18))  # 全局标题
        subplot_h_num: int = kwargs.get('subplot_h_num', 3)  # 纵向几列
        gs = gridspec.GridSpec(1, subplot_h_num, wspace=0.2, hspace=0.5, )

        for i in range(len(sub_types)):
            # 数据处理
            df = self.perf.df_all[[sub_types[i], 'open_datetime']]
            proceed_df = self.process_df_by_minites(data=df, gap=60)
            # 额外特殊处理 防止过多
            if sub_types[i] == 'earnings':
                proceed_df[sub_types[i]] = proceed_df[sub_types[i]].apply(
                    lambda x: int(x / 50) * 50 if abs(x) < 400 else int(x / 400) * 400)
            else:
                proceed_df[sub_types[i]] = proceed_df[sub_types[i]].apply(
                    lambda x: int(x / 50) * 50 if abs(x) < 300 else int(x / 250) * 250)

            self.plot_heatmap(data=proceed_df,
                              need_show=False,  # 子图不用
                              need_save=False,  # 子图不用
                              columns=[sub_types[i], 't'],
                              fig=fig,
                              ax=plt.subplot(gs[i // subplot_h_num, i % subplot_h_num]),
                              title=sub_types[i],
                              )

        if kwargs.get('need_show', None):
            plt.show()

        if kwargs.get('need_save', None):
            filename = f'{plot_what}_{file_num}_multi_heatmap.jpg'
            fig.savefig(path.join(self.dir, filename),
                        dpi=200, bbox_inches='tight')

    def plot_heatmap(self,
                     data: pd.DataFrame,
                     columns: List[str],
                     **kwargs,  # 其他参数
                     ):
        """画热力图"""
        fig = kwargs.get('fig', None)
        if fig is None:
            fig = plt.figure(figsize=([22, 18]))
        ax = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots(figsize=(22, 18))

        title: str = kwargs.get('title', None)
        if title:
            ax.set_title(title, fontsize=kwargs.get('title_size', 15))  # 子图标题

        grouped = data.groupby(columns)
        ok_data = grouped.count().unstack()
        ok_data.columns = [x[1] for x in ok_data.columns]

        cmap = sns.cubehelix_palette(12, start=.5, rot=-.75, as_cmap=True)  # start=1, rot=2, gamma=0.8,
        # x轴标签旋转
        ax.tick_params(which='major', axis='x', labelrotation=65, labelsize=9, length=5, pad=10)
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        sns.heatmap(ok_data, cmap=cmap, linewidths=0.05, vmin=0, ax=ax)

        if kwargs.get('need_show', None):
            plt.show()

        if kwargs.get('need_save', None):
            filename = f'{title}_heatmap.jpg'
            fig.savefig(path.join(self.dir, filename),
                        dpi=200, bbox_inches='tight')

    def plot_pnl_and_floating(self,
                              **kwargs,  # 其他参数
                              ):
        """打印pnl和价格曲线"""
        fig = kwargs.get('fig', None)
        if fig is None:
            fig = plt.figure(figsize=([46, 18]))
        ax = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots(figsize=(28, 18))

        title: str = kwargs.get('title', None)
        if title:
            fig.suptitle(title, size=kwargs.get('title_size', 18))  # 全局标题

        # 图片包含几块 分别的尺寸以及比例
        gs = gridspec.GridSpec(3, 3, hspace=0.6)
        ax0 = plt.subplot(gs[:, :2])
        ax1 = plt.subplot(gs[0, 2])
        ax2 = plt.subplot(gs[1, 2])
        ax3 = plt.subplot(gs[2, 2])

        # 这行 调用 打印pnl
        self.plot_day_pnl(ax=ax0, color='b')
        self.plot_day_pnl(ax=ax0, type_='long', color='g', )
        self.plot_day_pnl(ax=ax0, type_='short', color='r', )  # dashes=[(2, 2), (2, 2)]

        # # 这行 画布林带
        # mean_df = pd.DataFrame(self.perf.day_max_floating_diff_mean)
        # for i in [2, 1, -1, -2, 0]:
        #     df = mean_df.copy()
        #     df.columns = ['dev']
        #     df['dev'] = df['dev'] + i * self.perf.day_max_floating_diff_std
        #
        #     # for t in ['day_max_floating_diff_mean', ] + [f'day_max_floating_diff_bl{i}'
        #     #                                              for i in [2, 1, -1, -2]]:
        #     self.plot_as_line_everyday(data=df, ax=ax1,  # label=t,
        #                                title='Day Average Max floating (profit - loss)',
        #                                color='b' if i == 0 else ('r' if i > 0 else 'g'),
        #                                dashes=False if i == 0 else [(2, 2), (2, 2)],
        #                                markers=False,
        #                                label=False,
        #                                )

        # 这行 打印 每天最大浮盈浮亏 平均值 std
        for t in ['day_max_floating_profit_mean', 'day_max_floating_profit_std', ]:
            data = pd.DataFrame(eval(f'self.perf.{t}'))
            self.plot_as_line_everyday(data=data, ax=ax1, label=t,
                                       title='Day Average Max floating profit',
                                       color='g' if 'mean' in t else 'b',
                                       )
        for t in ['day_max_floating_loss_mean', 'day_max_floating_loss_std', ]:
            data = pd.DataFrame(eval(f'self.perf.{t}'))
            # data['max_floating_loss'] *= -1
            self.plot_as_line_everyday(data=data, ax=ax2, label=t,
                                       title='Day Average Max floating loss',
                                       color='r' if 'mean' in t else 'b',
                                       )
        for t in ['day_max_floating_diff_mean', 'day_max_floating_diff_std', ]:
            data = pd.DataFrame(eval(f'self.perf.{t}'))
            self.plot_as_line_everyday(data=data, ax=ax3, label=t,
                                       title='Day Average Max floating diff',
                                       color='y' if 'mean' in t else 'b',
                                       # dashes=False if 'std' in t else [(2, 2), (2, 2)],
                                       )
        # # 打印 每天成交量
        # for t in ['all', 'short']:
        #     df_ = pd.DataFrame(self.perf.day_trades[t])
        #     self.plot_bar('Trades', df_, type_=t, ax=ax1)

        # 画K线图
        # plot_candle(self.df_kline, ax=ax1)

        if kwargs.get('need_show', None):
            plt.show()

        if kwargs.get('need_save', None):
            filename = f'{kwargs.get("save_name", time.time())}_pnl.jpg'
            fig.savefig(path.join(self.dir, filename),
                        dpi=200, bbox_inches='tight')

    def plot_bar(self,
                 title: str,
                 df: pd.DataFrame,
                 type_: str = 'all',  # 类型 in ['all', 'long', 'short']
                 ax=None,  # 画在哪个区域
                 ):
        ax.set_title(title, fontsize=15)
        # 时间刻度处理 主刻度
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))  # 设置主刻度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(which='major', axis='x', labelrotation=25, labelsize=9, length=5, pad=10)

        sns.barplot(x=df.index, y='earnings', data=df, ax=ax,
                    label=type_, palette=['r' if type_ == 'short' else 'g'], )

    def plot_multi(self, params: dict,
                   bt_dir: str,  # 回测结果存放目录
                   show: bool = True,  # 是否显示
                   save: bool = False,  # 是否保存
                   ):
        """打印 多图表"""
        fig = plt.figure(figsize=([32, 28]))
        hp = params['hyper_parameter_name']  # 超参数列名
        title = f'{params["symbol"]}: {hp["ma_type"]}{hp["bolling_len"]}' \
                f'  bolling_{hp["bl_for_long"]}_line_for_long' \
                f'  stop_loss_{hp["stop_loss_type"]} ' \
                f'  {params["description"]}' \
                f'  open-times={len(self.perf.df_all)}'
        fig.suptitle(title, size=17)  # 全局标题

        # 图片包含几块 分别的尺寸以及比例
        gs = gridspec.GridSpec(2, 3, hspace=0.4)
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[0, 1])
        ax2 = plt.subplot(gs[0, 2])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[1, 2])
        self.plot_day_pnl(ax=ax0)
        self.plot_day_earnings(ax=ax1, )
        self.plot_max_float_profit_dist(ax=ax2)
        self.plot_max_float_profit(ax3, ax4, ax5)

        if show:
            plt.show()

        if save:
            filename = f'{params["symbol"]}_{hp["ma_type"]}' \
                       f'_{hp["bolling_len"]}_{hp["bl_for_long"]}' \
                       f'_{hp["stop_loss_type"]}_{params["description"]}.jpg'
            fig.savefig(path.join(bt_dir, filename), dpi=200, bbox_inches='tight')

    def plot_max_float_profit(self,
                              ax_I,
                              ax_II,
                              ax_III,
                              gap: int = 30,  # 间隔 分钟
                              para: str = ''):
        """ plot 开仓频率 按日内时间 """

        # 盈利
        # ax.set_title(f'open cum / time {para}', fontsize=15)
        # df = self.perf.df.copy()
        # 最大浮盈
        ax_I.set_title(f'max float profit {para}', fontsize=15)
        ax_II.set_title(f'open times {para}', fontsize=15)
        ax_III.set_title(f'max float profit {para}', fontsize=15)
        df = self.perf.max_floating_profit(add_open_datetime=True)
        # df['open_datetime'] = df['open_datetime'].apply(
        #     lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        # 按半小时
        df['t'] = df['open_datetime'].apply(
            lambda x: (x.time().hour * 60 + x.time().minute) // gap)
        df = df[['t', 'max_floating_profit']]
        df = df.sort_values('t')

        grouped = df.groupby(['t'])
        grouped_result: pd.DataFrame = grouped.describe().iloc[:, :3]
        grouped_result.columns = ['times', 'mean', 'std']
        grouped_result = grouped_result.sort_index()
        grouped_result['t'] = grouped_result.index
        grouped_result['t'] = grouped_result['t'].apply(lambda x: f'{int(x) * gap // 60}:{int(x) * gap % 60}')
        grouped_result['t'] = grouped_result['t'].apply(lambda x: f'{x}0' if x.split(':')[1] == '0' else x)
        # grouped_result.set_index('t', drop=True, inplace=True)
        # print(grouped_result, type(grouped_result))

        df['t'] = df['t'].apply(lambda x: f'{int(x) * gap // 60}:{int(x) * gap % 60}')
        df['t'] = df['t'].apply(lambda x: f'{x}0' if x.split(':')[1] == '0' else x)

        # ok_data =grouped.count()
        # ok_data.index = [f'{int(x)*gap // 60}:{int(x)*gap % 60}' for x in ok_data.index]
        # print(ok_data, ok_data.columns)
        ax_I.tick_params(which='major', axis='x', labelrotation=60, labelsize=9, length=5, pad=10)
        ax_II.tick_params(which='major', axis='x', labelrotation=60, labelsize=9, length=5, pad=10)
        ax_III.tick_params(which='major', axis='x', labelrotation=60, labelsize=9, length=5, pad=10)
        sns.boxplot(x='t', y='max_floating_profit', data=df, ax=ax_I)
        # sns.countplot(x='t', data=df, palette="Greens_d", ax=ax_II)
        # sns.violinplot(x='t', y='max_floating_profit', data=df, ax=ax_I)
        # sns.relplot(x='t', y='max_floating_profit', data=df, ax=ax_I)
        sns.barplot(x='t', y='times', data=grouped_result, ax=ax_II, label='times')
        sns.barplot(x='t', y='std', data=grouped_result, ax=ax_III, label='std')
        sns.scatterplot(x='t', y='mean', data=grouped_result, ax=ax_III, label='mean')

    def plot_max_float_profit_dist(self, ax=None, para: str = ''):
        """"""
        if ax is None:
            f, ax = plt.subplots(figsize=(16, 12))
        ax.set_title(f'max float profit usd/lot  {para} total dist ', fontsize=15)
        df_mfp = pd.DataFrame(self.perf.max_floating_profit())
        sns.distplot(df_mfp['max_floating_profit'], bins=150, fit=norm,
                     kde=True, ax=ax)

    def plot_as_line_everyday(self,
                              data,  # 数据
                              type_: str = 'all',  # 类型 in ['all', 'long', 'short']
                              title: str = ' ',
                              ax=None,  # 画在哪个区域
                              color: str = 'blue',  #
                              label=None,
                              dashes: Union[list, bool] = False,  # 虚线
                              markers: bool = True,
                              ):
        """plot 画线 按天"""
        if ax is None:
            f, ax = plt.subplots(figsize=(16, 12))
        ax.set_title(title, fontsize=15)

        # 时间刻度处理 主刻度
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=len(data) // 15))  # 设置主刻度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(which='major', axis='x', labelrotation=35, labelsize=9, length=5, pad=10)
        # 副刻度 可选
        # hoursLoc = mdates.HourLocator(interval=20)  # 为20小时为1副刻度
        # ax.xaxis.set_minor_locator(hoursLoc)
        # ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
        # ax.tick_params(which='minor', axis='x', labelsize=8, length=3)

        sns.lineplot(data=data, markers=True,  # dashes=dashes,
                     ax=ax, label=label, palette=[color], )

        if ax is None:
            plt.show()
        # f.savefig(f'{t}_day_pnl.jpg', dpi=100, bbox_inches='tight')

    def plot_day_pnl(self,
                     type_: str = 'all',  # 类型 in ['all', 'long', 'short']
                     ax=None,  # 画在哪个区域
                     para: str = '',  # 辅助显示信息
                     color: str = 'blue',  #
                     dashes: Union[list, bool] = False,  # 虚线
                     ):
        """plot"""
        if ax is None:
            f, ax = plt.subplots(figsize=(16, 12))
        ax.set_title(f'day pnl {para}', fontsize=15)

        df_day_pnl = pd.DataFrame(self.perf.day_pnl[type_])

        # 时间刻度处理 主刻度
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=len(df_day_pnl) // 15))  # 设置主刻度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(which='major', axis='x', labelrotation=35, labelsize=9, length=5, pad=10)
        # 副刻度 可选
        # hoursLoc = mdates.HourLocator(interval=20)  # 为20小时为1副刻度
        # ax.xaxis.set_minor_locator(hoursLoc)
        # ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
        # ax.tick_params(which='minor', axis='x', labelsize=8, length=3)

        sns.lineplot(data=df_day_pnl, markers=False,  # dashes=dashes,
                     ax=ax, label=type_, palette=[color], )

        if ax is None:
            plt.show()
        # f.savefig(f'{t}_day_pnl.jpg', dpi=100, bbox_inches='tight')

    def plot_day_earnings(self, ax=None, para: str = ''):
        """plot"""
        if ax is None:
            f, ax = plt.subplots(figsize=(16, 12))
        ax.set_title(f'day earnings {para}', fontsize=15)

        # 时间刻度处理 主刻度
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))  # 设置主刻度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(which='major', axis='x', labelrotation=35, labelsize=9, length=5, pad=10)

        df_day_earnings = pd.DataFrame(self.perf.day_earnings)
        df_day_earnings['dt'] = df_day_earnings.index
        df_day_earnings['pn'] = df_day_earnings['earnings'].apply(lambda x: '+' if x > 0 else '-')  # 正负
        sns.barplot(x='dt', y='earnings', data=df_day_earnings,
                    hue='pn', ax=ax, )
        if ax is None:
            plt.show()
        # f.savefig(f'{t}_day_earnings.jpg', dpi=100, bbox_inches='tight')

    @classmethod
    def process_df_by_minites(cls,
                              data: pd.DataFrame,
                              gap: int = 30,  # 间隔 分钟
                              ) -> pd.DataFrame:
        # 把数据 按小时和分钟，时间间隔，分开
        df: pd.DataFrame = data.copy()

        df['t'] = df['open_datetime'].apply(
            lambda x: (x.time().hour * 60 + x.time().minute) // gap)
        # df.drop(columns=['open_datetime'], inplace=True)  # 删除列
        df = df.sort_values('t')
        df.reset_index(drop=True, inplace=True)
        df['t'] = df['t'].apply(lambda x: f'{int(x) * gap // 60}:{int(x) * gap % 60}')
        # 把时间格式 字符串补漂亮
        df['t'] = df['t'].apply(lambda x: f'{x}0' if x.split(':')[1] == '0' else x)
        df['t'] = df['t'].apply(lambda x: f'0{x}' if len(x.split(':')[0]) == 1 else x)

        return df.reset_index(drop=True)
