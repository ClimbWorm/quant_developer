
import talib
import numpy as np
import pandas as pd

epsilon = 1e-6
rsiPeriod = 14

def IdentifyRsi(df):

	RSI = talib.RSI(df.Close.values, timeperiod=rsiPeriod).tolist()
	df['RSI'] = RSI

	return df

def IdentifyPinBar(df):

	'''
	UpPinBar 看涨PinBar
		
		取到Open与Close之间的更小值，赋值给lowerOC
		
		1. (lowerOC - Low)/(High - lowerOC) > 3
		
		若Bar符合上述规则，则Bar属于DownPinBar，DownPinBar特征的参数为 (lowerOC - Low)/(High - lowerOC)
		若Bar不符合上述规则，则Bar不属于DownPinBar，DownPinBar特征的参数为 0

	DownPinBar 看跌PinBar
		
		取到Open与Close之间的更大值，赋值给higherOC
		
		1. (High - higherOC)/(higherOC - Low) > 3
		
		若Bar符合上述规则，则Bar属于UpPinBar，UpPinBar特征的参数为 (High - higherOC)/(higherOC - Low)
		若Bar不符合上述规则，则Bar不属于UpPinBar，UpPinBar特征的参数为 0
	'''

	UpPinBar = []
	DownPinBar = []

	for o,h,l,c in zip(df.Open, df.High, df.Low, df.Close):
		
		if(o > c):
			higherOC = o
			lowerOC = c
		else:
			higherOC = c
			lowerOC = o
			
		if(h-lowerOC == 0): h += epsilon
		if(higherOC-l == 0): l -= epsilon
		
		if((lowerOC - l)/(h - lowerOC) > 3):
			
			UpPinBar.append((lowerOC - l)/(h - lowerOC))
		
		else:
			
			UpPinBar.append(0)
		
		if((h - higherOC)/(higherOC - l) > 3):
			
			DownPinBar.append((h - higherOC)/(higherOC - l))
		
		else:
			
			DownPinBar.append(0)
	df['UpPinBar'] = UpPinBar
	df['DownPinBar'] = DownPinBar

	return df

def IdentifyTopBottomType(df):

	'''
	TopType (当前时刻t，左一时刻t-1，左二时刻t-2) 顶分型
		
		1. t-1时刻对应的高点是t, t-1, t-2 三个时刻中最高的
		2. t-1时刻对应的低点是t, t-1, t-2 三个时刻中最高的
		3. (t时刻Bar振幅 > t-2时刻Bar振幅) or (t时刻最低价 < t-2时刻最低价)
		
		若t时刻Bar符合上述规则，则t时刻出现了顶分型，TopType特征的参数为 1
		若t时刻Bar不符合上述规则，则t时刻没有出现顶分型，TopType特征的参数为 0
		
	BottomType (当前时刻t，左一时刻t-1，左二时刻t-2) 底分型
		
		1. t-1时刻对应的高点是t, t-1, t-2 三个时刻中最低的
		2. t-1时刻对应的低点是t, t-1, t-2 三个时刻中最低的
		3. (t时刻Bar振幅 > t-2时刻Bar振幅) or (t时刻最高价 > t-2时刻最高价)
		
		若t时刻Bar符合上述规则，则t时刻出现了底分型，BottomType特征的参数为 1
		若t时刻Bar不符合上述规则，则t时刻没有出现底分型，BottomType特征的参数为 0
	'''

	TopType = [0,0]
	BottomType = [0,0]

	# t   bar --> idx = 1
	# t-1 bar --> idx = 2
	# t-2 bar --> idx = 3

	high1 = df.High.values[2:]
	high2 = df.High.values[1:-1]
	high3 = df.High.values[0:-2]
	low1 = df.Low.values[2:]
	low2 = df.Low.values[1:-1]
	low3 = df.Low.values[0:-2]

	for h1,h2,h3,l1,l2,l3 in zip(high1, high2, high3, low1, low2, low3):
		
		if(h2 > h1 and h2 > h3 and l2 > l1 and l2 > l3 and ( (h1-l1) > (h3-l3) or l1 < l3)):
			
			TopType.append(1)
		
		else:
			
			TopType.append(0)

		if(h2 < h1 and h2 < h3 and l2 < l1 and l2 < l3 and ((h1-l1) > (h3-l3) or h1 > h2 )):
			
			BottomType.append(1)
		
		else:
			
			BottomType.append(0)
	
	df['TopType'] = TopType
	df['BottomType'] = BottomType

	return df

def IdentifyPregnantType(df):

	'''
	UpPregnantType (当前时刻t, 左一时刻t-1) 看涨孕线

		1. t时刻，Bar为阴线
		2. t-1时刻，Bar为阳线
		3. t-1时刻的开盘价 > t时刻的收盘价
		4. t-1时刻的收盘价 < t时刻的开盘价
		
		若t时刻Bar符合上述规则，则t时刻出现了看涨孕线结构，UpPregnantType特征的参数为 1
		若t时刻Bar不符合上述规则，则t时刻没有出现看涨孕线结构，UpPregnantType特征的参数为 0

	DownPregnantType (当前时刻t, 左一时刻t-1) 看跌孕线
		
		1. t时刻，Bar为阳线
		2. t-1时刻，Bar为阴线
		3. t-1时刻的收盘价 > t时刻的开盘价
		4. t-1时刻的开盘价 < t时刻的收盘价
		
		若t时刻Bar符合上述规则，则t时刻出现了看跌孕线结构，DownPregnantType特征的参数为 1
		若t时刻Bar不符合上述规则，则t时刻没有出现看跌孕线结构，DownPregnantType特征的参数为 0
	'''

	UpPregnantType = [0]
	DownPregnantType = [0]

	# t   bar --> idx = 1
	# t-1 bar --> idx = 2

	open1 = df.Open.values[1:]
	close1 = df.Close.values[1:]
	high1 = df.High.values[1:]
	low1 = df.Low.values[1:]
	open2 = df.Open.values[:-1]
	close2 = df.Close.values[:-1]
	high2 = df.High.values[:-1]
	low2 = df.Low.values[:-1]

	for o1,c1,o2,c2 in zip(open1,close1,open2,close2):
		
		if(c1 < o1 and c2 > o2 and o2 > c1 and c2 < o1):
			
			UpPregnantType.append(1)
		
		else:
			
			UpPregnantType.append(0)
			
		if(c1 > o1 and c2 < o2 and c2 > o1 and o2 < c1):
		
			DownPregnantType.append(1)
		
		else:
			
			DownPregnantType.append(0)

	df['UpPregnantType'] = UpPregnantType
	df['DownPregnantType'] = DownPregnantType

	return df

def IdentifySwallowType(df):

	'''
	UpSwallowType (当前时刻t, 左一时刻t-1) 看涨吞没形态

		1. t-1时刻，Bar为阴线
		2. t时刻，Bar为阳线
		3. t-1时刻开盘价 < t时刻收盘价
		4. t-1时刻收盘价 > t时刻开盘价

	DownSwallowType (当前时刻t, 左一时刻t-1) 看跌吞没形态

		1. t-1时刻，Bar为阳线
		2. t时刻，Bar为阴线
		3. t-1时刻收盘价 < t时刻开盘价
		4. t-1时刻开盘价 > t时刻收盘价
	'''

	UpSwallowType = [0]
	DownSwallowType = [0]

	# t   bar --> idx = 1
	# t-1 bar --> idx = 2

	open1 = df.Open.values[1:]
	close1 = df.Close.values[1:]
	open2 = df.Open.values[:-1]
	close2 = df.Close.values[:-1]

	for o1,c1,o2,c2 in zip(open1,close1,open2,close2):
		
		if(c2 < o2 and c1 > o1 and o2 < c1 and c2 > o1):
			
			UpSwallowType.append(1)
		
		else:
			
			UpSwallowType.append(0)
			
		if(c2 > o2 and c1 < o1 and c2 < o1 and o2 > c1):
		
			DownSwallowType.append(1)
		
		else:
			
			DownSwallowType.append(0)
	
	df['UpSwallowType'] = UpSwallowType
	df['DownSwallowType'] = DownSwallowType

	return df

def NewOpenBar(df):
	
	summer = ((df.GMT < '2018-11-04 02:00:00') | \
			 ((df.GMT >= '2019-03-10 02:00:00') & (df.GMT < '2019-11-03 02:00:00')) | \
			 ((df.GMT >= '2020-03-08 02:00:00') & (df.GMT < '2020-11-01 02:00:00'))).values
	
	df['SW'] = summer
	df['DayOfWeek'] = df.GMT.dt.dayofweek
	
	isOpenBar = []
	
	for gmt, sw, dow in zip(df.GMT, df.SW, df.DayOfWeek):
		
		if((sw == 1 and gmt.strftime('%H:%M:%S') == '13:30:00') or (sw == 0 and gmt.strftime('%H:%M:%S') == '14:30:00')):
			
			if(dow != 5 and dow != 6):
			
				isOpenBar.append('Open')
			
			else:
				
				isOpenBar.append(' ')
		
		else:
			
			isOpenBar.append(' ')
	
	df['IsOpenBar'] = isOpenBar
	
	return df

def RunStrategyOnIntradayData(df, df_D_60, lastDayRange, ratioMean, ratioStd, order_form, history_form, RBH=40, RSL=60, RangeL=0.4, RangeH=1.6, n2=20):
	
	'''
	step1 检查当前是否存在订单，若存在订单识别是否出现离场或止损机会，若不存在订单则识别是否出现进场机会
	step2 current range落入指定区间
	step3 当前价格距离日内最低价更近 or 当前价格距离日内最高价更近
	step4 rsi + current bar阴阳 过滤
	step5 高低点bar形态 过滤
	step6 zscore概率矩阵确定仓位和手数
	step7 建仓
	'''
	
	orderNumber = 0

	for row in range(1,len(df)-1): # 遍历日内行情，检索进场离场情况
		
		'''
		变量缩写
			gmt: GMT
			o: open; h: high; l: low; c: close
			rsi_l: rsi last; rsi_c: rsi current
			range_c: range current
			zscore_c: zscore current
		'''
		
		gmt = df.loc[row].GMT
		o = df.loc[row].Open
		h = df.loc[row].High 
		l = df.loc[row].Low 
		c = df.loc[row].Close 
		rsi_l = df.loc[row-1].RSI 
		rsi_c = df.loc[row].RSI 
		
		if(len(order_form) == 0): # 当前不存在存续订单，识别新的进场机会
			
			dayHigh = np.max(df.loc[:row].High)
			dayHighIdx = np.argmax(df.loc[:row].High)
			dayLow = np.min(df.loc[:row].Low)
			dayLowIdx = np.argmin(df.loc[:row].Low)

			range_c = dayHigh - dayLow

			zscore_l = df_D_60.iloc[-1].Z_score # 获取到前一交易日的zscore
			zscore_c = (range_c/lastDayRange - ratioMean)/ratioStd

			isCurrentRangeRatioInBoundary = range_c/lastDayRange > RangeL and range_c/lastDayRange < RangeH # 实时range ratio在区间内的判断
			
			#isLastDayZscoreInBoundary = zscore_l > -0.5 and zscore_l < 1 # 昨日zscore在(-0.5, 1)之间的判断
			isCurrentZscoreInBoundary = zscore_c > -0.5 and zscore_c < 1 # 昨日zscore在(-0.5, 1)之间的判断

			if((isCurrentRangeRatioInBoundary and isCurrentZscoreInBoundary) or (isCurrentZscoreInBoundary == 0)): # 实时zscore落入均值区间，且实时range落入到(0.4, 1.6)范围内才存在交易机会
				
				if(c - dayLow < dayHigh - c): # 当前价格距离日内最低价更近，存在做多机会
					
					if(rsi_c < RBH and rsi_l < rsi_c and c > o): # rsi过滤
						
						isUpPinBar = df.loc[dayLowIdx].UpPinBar > 0
						isBottomType = df.loc[dayLowIdx+1].BottomType == 1 # 可能存在错误
						isUpPregnantType = df.loc[dayLowIdx+1].UpPregnantType == 1 # 可能存在错误
						isUpSwallowType = df.loc[dayLowIdx].UpSwallowType == 1
						
						if(isUpPinBar or isBottomType or isUpPregnantType or isUpSwallowType): # bar形态过滤

							isGoodZscore = IsCurrentZscoreEnterHistoryMeanBoundary(GetTag(zscore_l), GetTag(zscore_c), zscore_c, df_D_60, 0.2)
							# print(isGoodZscore)
							# print(isGoodZscore == 1)
							
							if(isGoodZscore):

								stoploss = SetStopLossPrice_Long_PercentageLine(dayHigh, dayLow)

								#ZnH = GetZnH(zscore_c)
								#stoploss = SetStopLossPrice_Long(dayHigh, lastDayRange, ratioMean, ratioStd, ZnH, n2)

								# print("即将多单建仓，当前实时价格: ", str(c))
								# print("当前设定的止损价格为: ", str(stoploss))
								# print("c - stoploss = ", str(c - stoploss))

								lots = SetLots(zscore_l, zscore_c, df_D_60)
								#lots = 1.0
								takeprofit = None
								
								order_form.append([orderNumber, gmt, 'Buy', lots, 'USA30', c, stoploss, takeprofit, df.iloc[0].GMT])
								orderNumber += 1

				else: # 当前价格距离日内最高价更近，存在做空机会
					
					if(rsi_c > RSL and rsi_l > rsi_c and c < o): # rsi过滤
						
						isDownPinBar = df.loc[dayHighIdx].DownPinBar > 0
						isTopType = df.loc[dayHighIdx+1].TopType == 1 # 可能存在错误
						isDownPregnantType = df.loc[dayHighIdx+1].DownPregnantType == 1 # 可能存在错误
						isDownSwallowType = df.loc[dayHighIdx].DownSwallowType == 1 
						
						if(isDownPinBar or isTopType or isDownPregnantType or isDownSwallowType): # bar形态过滤

							isGoodZscore = IsCurrentZscoreEnterHistoryMeanBoundary(GetTag(zscore_l), GetTag(zscore_c), zscore_c, df_D_60, 0.2)
							# print(isGoodZscore)
							# print(isGoodZscore == 1)

							if(isGoodZscore):

								stoploss = SetStopLossPrice_Short_PercentageLine(dayHigh, dayLow)

								#ZnH = GetZnH(zscore_c)
								#stoploss = SetStopLossPrice_Short(dayLow, lastDayRange, ratioMean, ratioStd, ZnH, n2)

								# print("即将空单建仓，当前实时价格: ", str(c))
								# print("当前设定的止损价格为: ", str(stoploss))
								# print("stoploss - c = ", str(stoploss - c))

								lots = SetLots(zscore_l, zscore_c, df_D_60)
								#lots = 1.0
								takeprofit = None
							
								order_form.append([orderNumber, gmt, 'Sell', lots, 'USA30', c, stoploss, takeprofit, df.iloc[0].GMT])
								orderNumber += 1
							
		else: # 当前存在存续订单，识别是否出现离场机会

			for i in range(len(order_form)): # 遍历订单表中所有订单(当前版本其实不需要这么操作，因为只会存在一单)

				order_i = order_form[i]
				
				if(order_i[2] == 'Buy'):
					
					if(rsi_c > RSL and rsi_l > rsi_c and c < o): # 止盈离场(99.9%): rsi进入超买区间，前一rsi数值高于当前rsi数值，当前bar为阴线
				
						order_i.append(gmt) # 当前时间为离场时间
						order_i.append(c) # 当前收盘价格为离场价格
						order_i.append(0.0) # 库存费规则暂不明确
						order_i.append((c - order_i[5]) * order_i[3] * 1.0) # 利润 = 价差 * 手数 * 点数转换单位
					
						history_form.append(order_i) # 在历史表中添加完整的交易记录
						order_form.remove(order_i) # 从订单表中将已经完成离场的订单移除
					
					elif(l <= order_i[6]): #止损离场: 当前bar的low低于止损价
						
						order_i.append(gmt) # 当前时间为离场时间
						order_i.append(order_i[6]) # 止损价为离场价格
						order_i.append(0.0) # 库存费规则暂不明确
						order_i.append((order_i[6] - order_i[5]) * order_i[3] * 1.0) # 利润 = 价差 * 手数 * 点数转换单位
					
						history_form.append(order_i) # 在历史表中添加完整的交易记录
						order_form.remove(order_i) # 从订单表中将已经完成离场的订单移除
					
					else: # 当前时刻没有离场机会
						pass
						
				elif(order_i[2] == 'Sell'):
					
					if(rsi_c < RBH and rsi_l < rsi_c and c > o): # 止盈离场(99.9%): rsi进入超卖区间，前一rsi数值低于当前rsi数值，当前bar为阳线
				
						order_i.append(gmt) # 当前时间为离场时间
						order_i.append(c) # 当前收盘价格为离场价格
						order_i.append(0.0) # 库存费规则暂不明确
						order_i.append((-1) * (c - order_i[5]) * order_i[3] * 1.0) # 利润 = 价差 * 手数 * 点数转换单位
					
						history_form.append(order_i) # 在历史表中添加完整的交易记录
						order_form.remove(order_i) # 从订单表中将已经完成离场的订单移除
					
					elif(h >= order_i[6]): #止损离场: 当前bar的high高于止损价
						
						order_i.append(gmt) # 当前时间为离场时间
						order_i.append(order_i[6]) # 止损价为离场价格
						order_i.append(0.0) # 库存费规则暂不明确
						order_i.append((-1) * (order_i[6] - order_i[5]) * order_i[3] * 1.0) # 利润 = 价差 * 手数 * 点数转换单位
					
						history_form.append(order_i) # 在历史表中添加完整的交易记录
						order_form.remove(order_i) # 从订单表中将已经完成离场的订单移除
					
					else: # 当前时刻没有离场机会
						pass

def GenerateDayLevelDF(df_r):

	df_r = NewOpenBar(df_r)

	newOpenBarList = df_r.loc[df_r.IsOpenBar == 'Open'].index.tolist()

	rowInfo = []

	for s, e in zip(newOpenBarList[:-1], newOpenBarList[1:]):

		data = df_r.loc[s:(e-1)]
		
		GMT = data.iloc[0].GMT
		Open = data.iloc[0].Open
		High = np.max(data.High)
		Low = np.min(data.Low)
		Close = data.iloc[-1].Close
		Volume = np.sum(data.Volume)

		rowInfo.append([GMT, Open, High, Low, Close, Volume])

	df_D = pd.DataFrame(data=rowInfo, columns=['GMT','Open','High','Low','Close','Volume'])

	df_D = df_D.loc[df_D.Volume > 1e-6].reset_index(drop=True)

	range1 = df_D.High.values - df_D.Low.values
	range2 = df_D.High.values[1:] - df_D.Close.values[:-1]
	range3 = df_D.Close.values[:-1] - df_D.Low.values[1:]

	rangeReal = [range1[0]]

	for r1, r2, r3 in zip(range1[1:], range2, range3):
		
		rangeReal.append(np.max([r1, r2, r3]))

	df_D['Range'] = rangeReal

	Date = []
	for i in range(df_D.GMT.size):
		Date.append(df_D.loc[i].GMT.strftime('%Y-%m-%d'))

	df_D['Date'] = Date

	Ratio = [1]
	for i in range(1,len(df_D.Range)):
		temp = df_D.Range[i]/df_D.Range[i - 1]
		Ratio.append(temp)
	df_D['Ratio'] = Ratio

	zscore = [1] * 20
	ratioMean = [1] * 20
	ratioStd = [1] * 20

	for i in range(20, len(Ratio)):
		ratio_mean_sub = np.array(Ratio[(i-20):(i+1)]).mean()#向前滚20
		ratio_std_sub = np.array(Ratio[(i-20):(i+1)]).std()
		zscore_sub = (Ratio[i] - ratio_mean_sub)/ratio_std_sub
		zscore.append(zscore_sub)
		ratioMean.append(ratio_mean_sub)
		ratioStd.append(ratio_std_sub)

	df_D['RatioMean'] = ratioMean
	df_D['RatioStd'] = ratioStd
	df_D['Z_score'] = zscore
	df_D['Z_score_tomorrow'] = zscore[1:] + [1]

	tag = [1] * 20     
	for i in range(20,len(df_D)):
		if df_D.iloc[i]['Z_score'] >= float(2):
			tag_sub = 'G'
		elif df_D.iloc[i]['Z_score'] >= float(1):
			tag_sub = 'F'
		elif df_D.iloc[i]['Z_score'] >= float(0.5):
			tag_sub = 'E'
		elif df_D.iloc[i]['Z_score'] >= float(0):
			tag_sub = 'D'
		elif df_D.iloc[i]['Z_score'] >= float(-0.5):
			tag_sub = 'C'
		elif df_D.iloc[i]['Z_score'] >= float(-1):
			tag_sub = 'B'
		else:
			tag_sub = 'A'
		tag.append(tag_sub)
		
	tag_tomorrow = tag[1:] + [1]

	df_D['Tag'] = tag
	df_D['Tag_tomorrow'] = tag_tomorrow

	df_D = df_D[20:len(df_D)-1]

	return df_D


def SetStopLossPrice_Long(dayHigh, lastDayRange, ratioMean, ratioStd, ZnH, n):

	if ZnH < 0:

		x = dayHigh - lastDayRange * ((ZnH * (1-n*0.01)) * ratioStd + ratioMean)

	elif ZnH > 0:

		x = dayHigh - lastDayRange * ((ZnH * (1+n*0.01)) * ratioStd + ratioMean)

	else:

		x = dayHigh - lastDayRange * ((1 * (1+n*0.01)) * ratioStd + ratioMean)

	return x

def SetStopLossPrice_Short(dayLow, lastDayRange, ratioMean, ratioStd, ZnH, n):

	if ZnH < 0:

		x = dayLow + (ratioMean + (ZnH * (1-n*0.01)) * ratioStd) * lastDayRange

	elif ZnH > 0:

		x = dayLow + (ratioMean + (ZnH * (1+n*0.01)) * ratioStd) * lastDayRange

	else:

		x = dayLow + (ratioMean + (1 * (1-n*0.01)) * ratioStd) * lastDayRange

	return x

def SetStopLossPrice_Long_PercentageLine(dayHigh, dayLow, p = 0.05):

	H = dayHigh - dayLow
	loss_long_line = dayLow - p * H

	return loss_long_line

def SetStopLossPrice_Short_PercentageLine(dayHigh, dayLow, p = 0.05):

	H = dayHigh - dayLow
	loss_short_line = dayHigh + p * H

	return loss_short_line

def GetZnH(zscore):

	if (zscore < -1):

		znh = -1

	elif (zscore >= -1 and zscore < -0.5):

		znh = -0.5

	elif (zscore >= -0.5 and zscore < 0):

		znh = 0

	elif (zscore >=0 and zscore < 0.5):

		znh = 0.5

	elif(zscore >= 0.5 and zscore < 1):

		znh = 1

	elif(zscore >= 1 and zscore < 2):

		znh = 2

	else:

		znh = 2

	return znh

def GetTag(zscore):

	if (zscore < -1):

		tag = 'A'

	elif (zscore >= -1 and zscore < -0.5):

		tag = 'B'

	elif (zscore >= -0.5 and zscore < 0):

		tag = 'C'

	elif (zscore >=0 and zscore < 0.5):

		tag = 'D'

	elif(zscore >= 0.5 and zscore < 1):

		tag = 'E'

	elif(zscore >= 1 and zscore < 2):

		tag = 'F'

	else:

		tag = 'G'

	return tag


def SetLots(zscore_l, zscore_c, df_D, standardLots = 1.0):

	'''
	若当前时刻存在进场机会，此函数返回一个手数数值，赋值给即将建仓的订单的Lots参数

	参数:

		1. zscore_l: 前一交易日，日级别zscore
		2. zscore_c: 当前交易日，实时zscore
		3. df_D: 日级别行情数据 + 特征列
		4. standardLots: 标准建仓手数

	规则:
		1. 提取到前一交易日zscore的tag_i(范围)
		2. 提取到当前时刻zscore current的tag_j(范围)

			基于1,2获取到当前时刻对应的tag状态(tag_i, tag_j)

			N = df的总长度

		3. 获取到历史N个交易日中，tag_i发生的次数 M
		4. 获取到历史N个交易日中，tag_i发生后，下一交易日发生tag_j的次数 K

			基于3,4计算出 K/M 的数值

		5. 建仓手数为 标准建仓手数 * K/M * M/N
	'''
	lots = 0

	tag_l = GetTag(zscore_l)
	tag_c = GetTag(zscore_c)

	totalTagNumer = len(df_D.Tag)

	numberM = len(df_D.loc[df_D.Tag == tag_l])
	numberK = len(df_D.loc[df_D.Tag == tag_l].loc[df_D.Tag_tomorrow == tag_c])

	if(numberM > 0):

		lots = standardLots * (numberK/numberM) * (numberM/totalTagNumer)

	else:

		pass

	return lots

def IsCurrentZscoreEnterHistoryMeanBoundary(tag_yesterday, current_tag, zscore_c, df, const = 0.2):

	df_filtered = df[['Tag','Tag_tomorrow','Z_score_tomorrow']].groupby(['Tag','Tag_tomorrow']).mean()

	if((tag_yesterday, current_tag) in df_filtered.index):

		mean = df_filtered.loc[tag_yesterday].loc[current_tag].values[0]

		sub, sup = mean - const, mean + const

		# print('mean: '+ str(mean))
		# print('boundary: ('+str(sub)+', '+str(sup)+')')
		# print('zscore_c: '+str(zscore_c))
		# print('=================')
		
		if ((zscore_c >= sub) and (zscore_c <= sup)):
			return True
		else:
			return False
	else:
		# print('近期历史交易日中不存在: '+tag_yesterday+ ' '+current_tag)
		# print('=================')
		return False




























