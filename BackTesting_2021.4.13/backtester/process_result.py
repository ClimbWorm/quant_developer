#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 处理回测结果
# Author: Tayii
# Data : 2021/2/1
# ----------------------------------------------------
import json
from os import path
from typing import Optional, Union

import pandas as pd

from config import BACK_TESTING_RESULT_DIR
from datahub.fmt import DataBase
from datahub.generate import Performance
from dc.sc import get_bar_data_from_txt
from dc.source_data import get_back_testing_result_data
from plot.back_testing_plot import PlotBtSingleChart, PlotBtTotalChart


def get_df_by_one_file(filepath: str,
                       ) -> Optional[pd.DataFrame]:
    """
    根据单个回测结果文件，获取其数据
    """
    if not filepath or not path.exists(filepath):
        print(f'回测文件 {filepath}不存在')
        return

    df: pd.DataFrame = get_back_testing_result_data(filepath)
    # print('df ', df)

    # 临时策略 剔除异常值 todo del
    df = df.drop(df[(1e5 < df['max_floating_profit'])
                    | (df['max_floating_profit'] < -1e5)
                    | (1e5 < df['max_floating_loss'])
                    | (df['max_floating_loss'] < -1e5)].index)

    return df


def get_performance_by_one_file(filepath: str,
                                init_asset: float,  # 初始金额
                                need_calc_day_earnings: bool = False,
                                ) -> Optional[Performance]:
    """
    根据单个回测结果文件，获取其performance
    """
    df: pd.DataFrame = get_df_by_one_file(filepath)
    if df is None or len(df) == 0:
        return

    # 计算回测文件结果的基本指标值
    per = Performance('Performance', init_asset=init_asset, df_all=df)
    # 如果有必要 再计算额外指标值
    if need_calc_day_earnings:
        per.calc_day_earnings(calc_max_floating=True)
    return per


class ProcessBtResult(DataBase):
    """分析/处理 回测数据"""

    def __init__(self,
                 plan_name: str,  # 本次回测计划 名
                 kline_data_path: str = None,  # K线数据 地址， None不带
                 **kwargs
                 ):
        DataBase.__init__(self, plan_name)
        self.plan_name: str = plan_name
        self.df_kline: pd.DataFrame = (self.__get_kline(kline_data_path)
                                       if kline_data_path else None)  # K线数据
        if kline_data_path:
            print(f'带 kline_data: \n', self.df_kline)

        # 回测数据存放的文件夹
        self.dir = path.join(BACK_TESTING_RESULT_DIR, f'{plan_name}')

        # outline
        self.outline = _outline = self.__get_outline()
        self.description: str = _outline['description']  # 回测计划描述
        self.symbol: str = _outline['symbol']  #
        self.hyper_parameter_name: list = _outline['hyper_parameter_name']  # 超参数名
        self.init_money: float = _outline['init_money']  # 初始资产

        # 回测 total数据 csv文件地址
        self.total_csv_filepath = path.join(self.dir, 'total_result.csv')
        if not path.exists(self.total_csv_filepath):
            self.__generate_total_csv()

        # 获取total数据
        self.df_total: pd.DataFrame = self.__get_df_total(**kwargs)

    def __get_df_total(self,
                       need_filter: bool = False,  # 是否筛选一下
                       **kwargs,  # in_ma_patten,
                       ) -> pd.DataFrame:
        """
        获取total数据，有时候还需要再筛选一下 盈利的，或其他的
        """
        df_total: pd.DataFrame = pd.read_csv(self.total_csv_filepath)
        if not need_filter:
            return df_total

        # 再筛选一下 盈利的，或其他的 =======================
        # df_total = df_total[df_total['total_net_profit.all'] > 0]
        domp = kwargs.get('in_ma_patten', None)
        if domp:
            df_total = df_total[df_total['in_ma_patten'] == domp]  #

        if df_total is None or len(df_total) == 0:  # 筛选完后判断一下
            return pd.DataFrame()

        # df_total = df_total[df_total['min1_bolling_len'] >= 10]
        #
        # def select_dev(x):
        #     x = int(eval(x)[0])
        #     return 1 <= x <= 2
        #     # return -0.5 <= x <= 1.5  # neg
        #
        # df_total = df_total[df_total['min1_bolling_dev'].apply(select_dev)]
        # if df_total is None or len(df_total) == 0:  # 筛选完后判断一下
        #     return pd.DataFrame()
        #
        # def select_p(x):
        #     return 0.15 <= x <= 0.21
        #     # return 0.2 <= x <= 0.25
        #     # return 0.3 <= x <= 0.5  # neg
        #
        # df_total = df_total[df_total['take_profit_trigger_p'].apply(select_p)]

        # def select_fix(x):
        #     return 0.15 <= x <= 0.5
        #     # return 0.2 <= x <= 0.25
        #     # return 0.3 <= x <= 0.5  # neg
        #
        # df_total = df_total[df_total['take_profit_trigger_p'].apply(select_p)]
        # 去重
        df_total.drop_duplicates(subset=['std_net.max_floating_loss.all', 'avg_net_profit.all.all'],
                                 keep='first', inplace=True)
        # 按利润排序
        # df_total = df_total[df_total['total_net_profit.all']>100000]
        df_total.sort_values('total_net_profit.all', ascending=False, inplace=True, )
        return df_total  # [:50]

    @staticmethod
    def __get_kline(kline_data_path: str = None  # K线数据 地址
                    ) -> pd.DataFrame:
        """"""
        df = get_bar_data_from_txt(filepath=kline_data_path, header='infer')
        df['Date'] = df['datetime'].apply(lambda x: x.date())
        # df_day = df_day[(df_day['Date'] > '2020/9') & (df_day['Date'] < '2021/2/01')]
        # 筛选列
        df = df[['Date', 'Open', 'Last', 'High', 'Low']]
        return df

    def select_all_detail_together(self, ):
        """
        特殊统计，从各文件 读取全部数据 混合一起
        """
        # 读取各子文件 汇聚一起
        all_detail_df = None
        for index, row in self.df_total[:].iterrows():
            filepath = path.join(self.dir, f'{index}.csv')
            df: pd.DataFrame = get_df_by_one_file(filepath)
            if df is None or len(df) == 0:
                return

            # 保留这些参数
            # df['min1_bolling_dev_long'] = row.min1_bolling_dev_long
            # df['min1_bolling_dev'] = row.min1_bolling_dev
            df['take_profit_trigger_p'] = row.take_profit_trigger_p

            if all_detail_df is None:
                all_detail_df = df
            else:
                all_detail_df = all_detail_df.append(df, ignore_index=True)

        return all_detail_df

    def plot_3d_by_hours(self,
                         df: pd.DataFrame,  # 原始数据
                         y: str,  # y轴
                         **kwargs,
                         ):
        """ 按时间序列画 """

        df = PlotBtSingleChart.process_df_by_minites(df, gap=60)
        grouped_df = df.groupby(['t', y, ])
        for z in ['max_floating_profit', 'max_floating_loss', 'earnings', ]:
            mean_df = pd.DataFrame(grouped_df[z].mean())

            mean_df.columns = [z]
            mean_df['t'] = [int(x[0][:2]) for x in mean_df.index]
            mean_df[y] = [x[1] for x in mean_df.index]

            # print(type(mean_df), )
            btt = PlotBtTotalChart(mean_df, dir=self.dir)
            btt.plot_3d(x='t',
                        y=y,
                        z=z,
                        title=f'/hours [{y}], [{z}]',
                        save=True,
                        )

    def show_total_result(self):
        """显示整体的数据"""

        # for dop in ['positive_correlation', 'negative_correlation']:
        #     calc_all_detail_together(dop)

        def polt_():
            for dop in ['positive_correlation', 'negative_correlation']:
                df = self.df_total[self.df_total['decide_open_type'] == dop]
                df['avg_net.max_floating_diff.all'] = (df['avg_net.max_floating_profit.all']
                                                       + df['avg_net.max_floating_loss.all'])
                total_plot = PlotBtTotalChart(df, dir=self.dir)
                for z in ['avg_net.max_floating_profit.all', 'avg_net_profit.all.all',
                          'avg_net.max_floating_loss.all', 'avg_net.max_floating_diff.all']:
                    total_plot.plot_3d(x='min1_bolling_len',
                                       y='min1_bolling_dev_long',
                                       z=z,
                                       title=f'[{dop}]  {z}',
                                       save=True,
                                       )

        # 按某些列 打印图表
        # for y in ['avg_net.max_floating_profit.all', 'avg_net.max_floating_profit.long',
        #           'avg_net.max_floating_profit.short', 'avg_net_profit.all.all']:
        #     btt.plot_earnings(
        #         grid=['decide_open_type'],
        #         x='stop_loss_fix_p',
        #         y=y,
        #         save=True,
        #     )

        # 按某些列 打印散点图
        # for y in ['avg_net.max_floating_profit.all', 'avg_net.max_floating_profit.long',
        #           'avg_net.max_floating_profit.short', 'avg_net_profit.all.all']:
        #     btt.plot_multi_scatter(
        #         x=['min1_bolling_len', 'min1_bolling_dev_long', 'min1_bolling_dev_short'],
        #         y=y,
        #         save=True,
        #     )

    def plot_total_multi_3d(self):
        """整体回测的数据 多子图 打印3d图 """
        # for dop in ['positive_correlation', 'negative_correlation']:
        #     # self.df_total['in_ma_patten'] = self.df_total['in_ma_patten'].apply(lambda x: x[:20])
        #     df = self.df_total[self.df_total['in_ma_patten'] == dop]
        #     # df['min1_bolling_dev'] = df['min1_bolling_dev'].apply(lambda x: eval(x)[0])
        #     if df is None or len(df) == 0:
        #         continue
        total_plot = PlotBtTotalChart(self.df_total, dir=self.dir)
        for y in ['main_len', ]:
            for z in ['avg_net.max_floating_profit.all', 'std_net.max_floating_profit.all',
                      'avg_net.max_floating_loss.all', 'std_net.max_floating_loss.all',
                      'avg_net.max_floating_diff.all', 'std_net.max_floating_diff.all',
                      'avg_net_profit.all.all', 'std_net_profit.all.all',
                      'total_net_profit.all', ]:
                x = 'min1_bl_dev'
                total_plot.plot_3d(x=x,
                                   y=y,
                                   z=z,
                                   title=f'[{x}], [{y}], [{z}]',
                                   need_show=False,
                                   need_save=True,
                                   )

    def plot_total_multi_dis_scatter(self, **kwargs):
        """整体回测的数据 多子图 打印分布图 散点图 """
        grid = ['main_len', 'min1_ma_type', ]  # min1_bl_dev
        dis_y = ['trade_num.all.all', 'trade_num.winning.all', 'trade_num.losing.all']
        for t in [f'avg_net_profit.all', 'ratio_winning_losing',
                  'avg_net.max_floating_profit', 'std_net.max_floating_profit',
                  'avg_net.max_floating_loss', 'std_net.max_floating_loss',
                  'avg_duration', ]:
            dis_y += [f'{t}.{x}' for x in ['all', 'long', 'short'][:1]]
            total_plot = PlotBtTotalChart(self.df_total, dir=self.dir)
            total_plot.plot_multi_face_grid(grid=grid,
                                            x='min1_bl_dev',  #
                                            y=dis_y,
                                            **kwargs)

        # for dop in ['positive_correlation', 'negative_correlation']:
        #     self.df_total['in_ma_patten'] = self.df_total['in_ma_patten'].apply(lambda x: x[:20])
        #     df = self.df_total[self.df_total['in_ma_patten'] == dop]
        #     df['avg_net.max_floating_diff.all'] = (df['avg_net.max_floating_profit.all']
        #                                            + df['avg_net.max_floating_loss.all'])
        #
        #     total_plot = PlotBtTotalChart(df, dir=self.dir)
        #     total_plot.plot_multi_scatter(
        #         x='min1_bolling_dev',  # take_profit_trigger_p
        #         y_types=['avg_net.max_floating_profit.all', 'avg_net.max_floating_loss.all',
        #                  'avg_net.max_floating_diff.all', 'avg_net_profit.all.all',
        #                  'total_net_profit.all', ],
        #         # subplot_h_num=4,
        #         title=dop,
        #         title_size=25,
        #         need_show=True,
        #         need_save=True,
        #     )

    def show_some_single_result(self,
                                index: Union[int, list],  # 第几个文件/回测
                                plot_what: str = None,  # 打印什么
                                **kwargs,
                                ):
        """显示一些 单回测的数据"""

        # kline数据 （如果需要）

        processed = 0  # 已处理
        index = index if isinstance(index, list) else [index, ]
        for i in index:
            processed += 1
            if processed % 100 == 19:
                print(f'已处理{processed / len(index):.1%}...')
            self.__show_single_result(i, plot_what=plot_what,
                                      **kwargs)

    def __show_single_result(self,
                             file_num: int,  # 第几个文件/回测
                             plot_what: str = None,  # 打印什么
                             **kwargs,
                             ):
        """显示单回测的数据"""
        # 回测文件对应的信息
        row = self.df_total.loc[file_num]
        # 绘图的title
        title = f'{self.symbol} {self.description} {file_num} \n'
        for j, t in enumerate(self.hyper_parameter_name):
            title += f' {t}={row[t]}'  # 拼接
            if j % 5 == 4:
                title += '\n'

        # 获取单回测的具体数据
        # filepath_ = row.filepath
        filepath_ = f'{self.dir}\\{file_num}.csv'

        # plotChart实例
        perf = get_performance_by_one_file(filepath_, self.outline['init_money'],
                                           need_calc_day_earnings=True, )
        if not perf:
            return

        s_plot = PlotBtSingleChart(perf, bt_dir=self.dir, kline_data=None)

        # 画图
        if plot_what == 'tri_ma':
            # 单回测 最大浮盈浮亏 单笔盈利的热力图 基于每小时分布 开仓时间
            s_plot.multi_subplot_heatmap(
                sub_types=['max_floating_profit', 'max_floating_loss', 'earnings'],

            )
        elif plot_what == 'pnl':
            s_plot.plot_pnl_and_floating(title=title,
                                         save_name=f'{file_num}',
                                         **kwargs)
        else:
            self.log(f'show_some_single_result：{plot_what}还未实现')

        # def plot_day_pnl_close():
        #     """日pnl和价格曲线"""
        # 先计算出day earnings

        # # Kline处理
        # start = per.df_all.iloc[0]['open_date']
        # end = per.df_all.iloc[-1]['open_date']
        # select_kline_df = (self.df_kline[(self.df_kline['Date'] >= start)
        #                                  & (self.df_kline['Date'] <= end)])
        # # 格式化列名，用于之后的绘制
        # select_kline_df.rename(columns={'Last': 'Close'}, inplace=True)
        # # 转换为日期格式
        # select_kline_df['Date'] = pd.to_datetime(select_kline_df['Date'])
        # # 将日期列作为行索引
        # select_kline_df.set_index(['Date'], inplace=True)

        # btc.plot_multi(params=outline, bt_dir=bt_dir, show=False, save=True)

    def __get_outline(self) -> dict:
        """获取outline数据"""
        with open(path.join(self.dir, f'outline_filepath.txt'), 'r') as f:
            outline = json.loads(f.read())
        return outline

    def __generate_total_csv(self,
                             show_each_result: bool = False,  # 显示每个表的统计结果
                             ) -> None:
        """生成total数据 csv文件"""
        filepath_ = f'{self.total_csv_filepath[:-3]}txt'
        total_df: pd.DataFrame = pd.read_csv(filepath_, sep='\\s+', header=None)
        total_df = total_df.sort_values(by=0)
        print(total_df.head(2))

        def _process_columns_0404(df: pd.DataFrame):
            """处理列"""
            df[1] = df[1] + df[2] + df[3] + df[4] + df[5] + df[6]
            df[11] = df[11] + df[12]
            df[14] = df[14] + df[15] + df[16] + df[17]
            df = df.drop(columns=[2, 3, 4, 5, 6, 12, 15, 16, 17, 18])
            return df

        def _process_columns_0406(df: pd.DataFrame):
            """处理列"""
            df[1] = df[1] + df[2] + df[3] + df[4] + df[5] + df[6]
            df[11] = df[11] + df[12]
            df[14] = df[14] + df[15] + df[16] + df[17]
            df = df.drop(columns=[2, 3, 4, 5, 6, 12, 15, 16, 17, 18])
            return df

        def _process_columns_0408(df: pd.DataFrame):
            """处理列"""
            # df[1] = df[1] + df[2]
            df[7] = df[7] + df[8]
            df[9] = df[9] + df[10]
            df[11] = df[11] + df[12] + df[13] + df[14]
            df = df.drop(columns=[ 8, 10, 12, 13, 14, 15])
            return df

        # 特殊处理  合并列参数（跟outline里一样）
        df_to_csv = _process_columns_0408(total_df).copy()  # 保存用的数据
        print(df_to_csv.head(3))
        df_to_csv.columns = ['id'] + self.hyper_parameter_name  # + ['filepath']
        df_to_csv.set_index('id', drop=True, inplace=True)
        df_to_csv.sort_index(inplace=True)

        size = len(df_to_csv)
        for index, row in df_to_csv[:].iterrows():
            # 处理每一个单独文件，即每一个回测参数组
            if index % 120 == 33:
                print(f'已处理{index / size:.1%}...')

            # filepath_ = row.filepath
            filepath_ = path.join(self.dir, f'{index}.csv')
            perf = get_performance_by_one_file(filepath_, self.outline['init_money'])
            if perf:
                if show_each_result:
                    perf.show_result()

                # 保存计算的指标值
                df_to_csv.loc[index, 'days'] = perf.days.days
                df_to_csv.loc[index, 'init_asset'] = perf.init_asset
                # df_to_csv.loc[index, 'sharpe_ratio'] = perf.sharpe_ratio(rf=0)
                for k, v in perf.calc_trades().items():
                    df_to_csv.loc[index, k] = v
                for k, v in perf.calc_earnings().items():
                    df_to_csv.loc[index, k] = v
                for k, v in perf.calc_duration().items():
                    df_to_csv.loc[index, k] = v

            else:
                print(f'{filepath_} not exists')

        # 保存
        df_to_csv.to_csv(self.total_csv_filepath)
        print(f'总的回测结果已计算好，并保存到：{self.total_csv_filepath}')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    plan_name_ = 'strategy_tri_ma_min1_0408_YMH21'
    # for domp in ['positive_correlation', 'negative_correlation'][:1]:
    #     # 处理汇总的结果（可以加一些筛选条件）
    #     pbr = ProcessBtResult(plan_name_, kline_data_path=None,
    #                           in_ma_patten=domp, )
    #
    #     all_detail: pd.DataFrame = pbr.select_all_detail_together()
    #     if all_detail is not None:
    #         pbr.plot_3d_by_hours(all_detail, y='take_profit_trigger_p',)

    # 日线数据
    # kline_data_path = 'E:\workspace\ScData\YM-Day-CBOT.scid_BarData.txt'

    # 处理汇总的结果（可以加一些筛选条件）
    pbr = ProcessBtResult(plan_name_,
                          kline_data_path=None,
                          need_filter=True,
                          # decide_open_ma_patten=domp,
                          )
    # 打印 总的结果
    # pbr.show_total_result()
    # pbr.plot_total_multi_dis_scatter(need_show=False,
    #                                  need_save=True, )
    # pbr.plot_total_multi_3d()

    # 打印一些 回测参数的 回测结果
    for index, row in pbr.df_total[:].iterrows():
        # if index < 2050:  # 断点续传
        #     continue
        try:
            pbr.show_some_single_result(index=index,
                                        plot_what='pnl',
                                        need_show=False,
                                        need_save=True,
                                        )
        except Exception as e:
            print(e)
            continue
