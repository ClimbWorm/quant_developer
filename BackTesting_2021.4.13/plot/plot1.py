#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
#  
# Author: Tayii
# Data : 2021/2/4
# ----------------------------------------------------
import datetime
import pandas as pd
pd.set_option("display.max_columns", None)
from dc.sc import get_bar_data_from_txt
from os import path
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm


def get_df(size: float = 1e6):
    paras = {
        'filepath': 'C://Users//sc//Downloads//F.US.EPH21_GraphData.txt',
        'columns': ['Date', 'Time', 'Open', 'High', 'Low', 'Last', 'Volume',
                    'ofTrades', 'OHLC_Avg', 'HLC_Avg', 'HL_Avg', 'Bid_Volume', 'Ask_Volume',
                    'ZigZag', 'TextLabels', 'ZigZagLineLength', 'ExtensionLines',
                    'BarNoInTrend', 'AskVol', 'BidVol', 'Trades', 'cpAskVol', 'cpBidVol',
                    'Duration', 'AskVol/T', 'BidVol/T', 'DeltaP', 'UpDownDelta',
                    'ATR', 'ATRSameTrend', 'ATR/T', 'WeightedVOI', 'PressRatio', 'DeltaSign']
    }

    return get_bar_data_from_txt(paras, size=size)


class Generate(object):
    def __init__(self):
        self.df_year1 = None
        self.first_day = None
        self.quarter = {}
        self.month = {}
        self.latest_2_weeks = None
        self.day = {}

    def gen_all(self):
        self.gen_year1('year1.csv')
        if self.df_year1 is not None:
            self.first_day = self.df_year1.index[0].date()

        self.gen_quarter()
        self.gen_month()
        self.gen_latest_2_weeks()
        self.gen_day()

    def gen_year1(self, test_filename):
        if path.exists(test_filename):
            df_year1 = pd.read_csv(test_filename)
            df_year1['datetime'] = df_year1['datetime'].apply(
                lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            df_year1.set_index('datetime', drop=True, inplace=True)
        else:
            df_year1 = get_df(1e8)
            df_year1 = df_year1[['datetime', 'ZigZagLineLength', 'ATRSameTrend', ]]
            df_year1 = df_year1[df_year1['ZigZagLineLength'] != 0]
            df_year1['datetime'] += datetime.timedelta(hours=7)
            df_year1.set_index('datetime', drop=True, inplace=True)
            df_year1.to_csv(test_filename)

        self.df_year1 = df_year1

    # 季度
    def gen_quarter(self):
        for i in range(4):
            start = self.first_day + datetime.timedelta(days=i * 90)
            end = start + datetime.timedelta(days=90)  # 约一季度
            self.quarter[i + 1] = self.df_year1.truncate(before=start, after=end)
            filename = f'quarter{i}.csv'
            if not path.exists(filename):
                self.quarter[i + 1].to_csv(filename)

    # 月
    def gen_month(self):
        for i in range(12):
            start = self.first_day + datetime.timedelta(days=i * 30)
            end = start + datetime.timedelta(days=30)
            self.month[i + 1] = self.df_year1.truncate(before=start, after=end)
            filename = f'month{i}.csv'
            if not path.exists(filename):
                self.month[i + 1].to_csv(filename)

    # 最近2周
    def gen_latest_2_weeks(self):
        end = self.df_year1.index[-1]
        start = end - datetime.timedelta(days=15)
        self.latest_2_weeks = self.df_year1.truncate(before=start, after=end)
        filename = f'latest_2_week.csv'
        if not path.exists(filename):
            self.latest_2_weeks.to_csv(filename)

    # 最近5天
    def gen_day(self):
        end = len(self.df_year1) - 1
        start = end
        find_days = 0
        for i in range(len(self.df_year1) - 1, 0, -1):
            if find_days >= 5:
                return

            prior_: datetime = self.df_year1.index[i - 1]
            curr_: datetime = self.df_year1.index[i]
            if prior_.time() < datetime.time(hour=17) <= curr_.time():
                self.day[find_days] = self.df_year1[start:end]

                filename = f'prior_{find_days}_days.csv'
                if not path.exists(filename):
                    self.day[find_days].to_csv(filename)
                # print(self.day[find_days])
                start -= 1
                end = start
                find_days += 1
            else:
                start -= 1


def distplot(t, df):
    f, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(f'{t}', fontsize=25)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
    ax.tick_params(labelsize=20)  # y轴 axis='y',
    # sns.distplot(df.ATRst, bins=50, fit=norm, kde=True)
    sns.countplot(x="ATRst", data=df, palette="Greens_d")
    plt.show()
    f.savefig(f'{t}_distplot.jpg', dpi=100, bbox_inches='tight')


def heatmap(t, df):
    # 设置颜色
    df['datetime'] = df['datetime'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['datetime'] -= datetime.timedelta(hours=7)  # 恢复到实际时间
    df['t'] = df['datetime'].apply(lambda x: x.time().hour * 6 + x.time().minute // 10)
    df = df[['t', 'ATRst', 'ATRSameTrend']]
    print(df)

    grouped = df.groupby(['ATRst', 't'])
    ok_data = grouped.count().unstack()
    ok_data.columns = [f'{x[1] // 6}:{x[1] % 6}0' for x in ok_data.columns]
    # print(f'{t}\n', ok_data)
    # print(ok_data.columns, ok_data.shape)

    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    f, ax = plt.subplots(figsize=(46, 12))
    ax.set_title(f'{t}', fontsize=25)
    ax.tick_params(labelsize=20)  # y轴 axis='y',
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    sns.heatmap(ok_data, cmap=cmap, linewidths=0.05, vmin=0, ax=ax)
    plt.show()
    f.savefig(f'{t}_heatmap.jpg', dpi=100, bbox_inches='tight')


def process():
    total = ['year1'] + [f'quarter{i}' for i in range(4)] \
            + [f'month{i}' for i in range(12)] + ['latest_2_week'] \
            + [f'prior_{i}_days' for i in range(5)]

    for t in total[:]:
        print(f'开始处理{t}')
        df = pd.read_csv(f'{t}.csv')
        df['ATRst'] = df['ATRSameTrend'].apply(lambda x: int(x * 10) / 10)
        # print((df))
        distplot(t, df)
        heatmap(t, df)


if __name__ == '__main__':
    gen = Generate()
    gen.gen_all()

    process()
