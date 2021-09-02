# 布林带止盈
def profit_reach_sigma(df, i, direction, sigma=1):
    """df为拼接好的有一堆指标的数据，如table_15_min,
    i表示当前bar的index  在ATRStrategy中要输入i-1
    sigma可取1,2
    direction表示long或者short"""
    if direction == "long":
        return df.iloc[i].High > df['upper_{}'.format(float(sigma))].iloc[i], df['upper_{}'.format(float(sigma))].iloc[i]
    else:
        return df.iloc[i].Low < df['lower_{}'.format(float(sigma))].iloc[i], df['lower_{}'.format(float(sigma))].iloc[i]

# 绝对数值止盈
def absnum_stopprofit(entrypoints, currentpoints, amount=95, unit=5):
    """unit表示1point对应的美金"""
    points = amount / unit
    return abs(entrypoints - currentpoints) > points

# 与止盈相关的止损,返回一个点位值
def stoploss_related_to_profits(df, EntryTime, Entrypoints, direction, sigma, multiple):
    """multiple为止盈的倍数系数"""
    # 获取入场点位和入场点的布林带数值
    if direction == "long":
        entrybolling = df[df.Time == EntryTime]['upper_{}'.format(float(sigma))].values[0]
    else:
        entrybolling = df[df.Time == EntryTime]['lower_{}'.format(float(sigma))].values[0]

    return Entrypoints - multiple * (entrybolling - Entrypoints)

def stoploss_related_to_profits_dynamic(df, i, direction, sigma=1,multiple = 0.5):
    """multiple为止盈的倍数系数
    sigma表示设置的止盈的倍数"""
    if direction == "long":
        return df['lower_{}'.format(float(sigma * multiple))].loc[i]
    else:
        return df['upper_{}'.format(float(sigma * multiple))].loc[i]

# 与atr相关的止损,返回止损线的值
def stoploss_related_to_atr(df, i, EntryTime, direction, atr_n, percentage=0.05):

    if direction == "long":
        return df[df.Time == EntryTime].Low.values[0] - percentage * df.iloc[i]["atr_{}".format(atr_n)]  # 当日实时走出来的atr
    else:
        return df[df.Time == EntryTime].High.values[0] + percentage * df.iloc[i]["atr_{}".format(atr_n)]
# 与atr相关的移动止损
def stoploss_related_to_atr_trailing(df, i, EntryTime, direction, atr_n, percentage=0.05):
    """EntryTime取最后进入的order的入场时间"""

    if direction == "long":
        return df[df.Time == EntryTime].Low.values[0] - percentage * df.iloc[i]["atr_{}".format(atr_n)]  # 当日实时走出来的atr
    else:
        return df[df.Time == EntryTime].High.values[0] + percentage * df.iloc[i]["atr_{}".format(atr_n)]

# low止损,返回止损线的值
def stoploss_related_to_low_or_high(df, EntryTime, direction):
    if direction == "long":
        return df[df.Time == EntryTime].Low.values[0]
    else:
        return df[df.Time == EntryTime].High.values[0]


# high或low止损，移动止损
def stoploss_related_to_low_or_high_trailing(df, EntryTime, direction):
    if direction == "long":
        return df[df.Time == EntryTime].Low.values[0]
    else:
        return df[df.Time == EntryTime].High.values[0]