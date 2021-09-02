import numpy as np
import pandas as pd

epsilon = 1e-6

def IdentifyPinBar(df):

	'''
	UpPinBar 看涨PinBar
		
		取到Open与Close之间的更小值，赋值给lowerOC
		
		1. (lowerOC - Low)/(High - lowerOC) > 2
		
		若Bar符合上述规则，则Bar属于DownPinBar，DownPinBar特征的参数为 (lowerOC - Low)/(High - lowerOC)
		若Bar不符合上述规则，则Bar不属于DownPinBar，DownPinBar特征的参数为 0

	DownPinBar 看跌PinBar
		
		取到Open与Close之间的更大值，赋值给higherOC
		
		1. (High - higherOC)/(higherOC - Low) > 2
		
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
			
			# UpPinBar.append((lowerOC - l)/(h - lowerOC))
			UpPinBar.append(1)
		
		else:
			
			UpPinBar.append(0)
		
		if((h - higherOC)/(higherOC - l) > 3):
			
			# DownPinBar.append((h - higherOC)/(higherOC - l))
			DownPinBar.append(1)
		
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
	UpPregnantType (当前时刻t, 左一时刻t-1, 左二时刻t-2) 看涨孕线

		1. t-2时刻柱体包住t-1时刻的柱体
        2. t时刻的最高价 > t-1时刻的最高价
		3. t时刻为阳线
		4. t时刻的最低价 > t - 2时刻的最低价
		
		若t时刻Bar符合上述规则，则t时刻出现了看涨孕线结构，UpPregnantType特征的参数为 1
		若t时刻Bar不符合上述规则，则t时刻没有出现看涨孕线结构，UpPregnantType特征的参数为 0

	DownPregnantType (当前时刻t, 左一时刻t-1, 左二时刻t-2) 看跌孕线
		
		1. t-2时刻柱体包住t-1时刻的柱体
        2. t时刻的最低价 < t-1时刻的最低价
		3. t时刻为阴线
		4. t时刻的最高价 < t - 2时刻的最高价
		
		若t时刻Bar符合上述规则，则t时刻出现了看跌孕线结构，DownPregnantType特征的参数为 1
		若t时刻Bar不符合上述规则，则t时刻没有出现看跌孕线结构，DownPregnantType特征的参数为 0
	'''

	UpPregnantType = [0,0]
	DownPregnantType = [0,0]

	# t   bar --> 1
	# t-1 bar --> 2
	# t-2 bar --> 3

	high1 = df.High.values[2:]
	low1 = df.Low.values[2:]
	open1 = df.Open.values[2:]
	close1 = df.Close.values[2:]
	open2 = df.Open.values[1:-1]
	close2 = df.Close.values[1:-1]
	high2 = df.High.values[1:-1]
	low2 = df.Low.values[1:-1]
	high3 = df.High.values[:-2]
	low3 = df.Low.values[:-2]
	open3 = df.Open.values[:-2]
	close3 = df.Close.values[:-2]

	for h1,l1,o1,c1,o2,c2,h2,l2,h3,l3,o3,c3 in zip(high1,low1,open1,close1,open2,close2,high2,low2,high3,low3,open3,close3):

		if((h3 > h2) and (l3 < l2) and (c1 > o1) and (h1 > h2) and (l1 > l3)):
			
			UpPregnantType.append(1)
		
		else:
			
			UpPregnantType.append(0)
			
		if((h3 < h2) and (l3 > l2) and (c1 < o1) and (l1 < l2) and (h1 < h3)):
		
			DownPregnantType.append(1)
		
		else:
			
			DownPregnantType.append(0)

	df['UpPregnantType'] = UpPregnantType
	df['DownPregnantType'] = DownPregnantType

	return df

def IdentifyTriplePregnantType(df):

	'''
	UpTriplePregnantType (当前时刻t, 左一时刻t-1, 左二时刻t-2, 左三时刻t-3) 看涨三重孕线

		1. t-3时刻柱体包住t-2时刻的柱体
		2. t-2时刻柱体包住t-1时刻的柱体
        3. t时刻的最高价 > t-2时刻的最高价
		4. t时刻的最低价 > t - 3时刻的最低价
		5. t时刻为阳线
		
		若t时刻Bar符合上述规则，则t时刻出现了看涨孕线结构，UpTriplePregnantType特征的参数为 1
		若t时刻Bar不符合上述规则，则t时刻没有出现看涨孕线结构，UpTriplePregnantType特征的参数为 0

	DownTriplePregnantType (当前时刻t, 左一时刻t-1, 左二时刻t-2, 左三时刻t-3) 看跌三重孕线
		
		1. t-3时刻柱体包住t-2时刻的柱体
		2. t-2时刻柱体包住t-1时刻的柱体
        3. t时刻的最低价 < t-2时刻的最低价
		4. t时刻的最高价 < t -3时刻的最高价
		5. t时刻为阴线
		
		若t时刻Bar符合上述规则，则t时刻出现了看跌孕线结构，DownTriplePregnantType特征的参数为 1
		若t时刻Bar不符合上述规则，则t时刻没有出现看跌孕线结构，DownTriplePregnantType特征的参数为 0
	'''

	UpTriplePregnantType = [0,0,0]
	DownTriplePregnantType = [0,0,0]

	# t   bar --> 1
	# t-1 bar --> 2
	# t-2 bar --> 3
	# t-3 bar --> 4



	open1 = df.Open.values[3:]
	high1 = df.High.values[3:]
	low1 = df.Low.values[3:]
	close1 = df.Close.values[3:]
	open2 = df.Open.values[2:-1]
	close2 = df.Close.values[2:-1]
	high2 = df.High.values[2:-1]
	low2 = df.Low.values[2:-1]
	open3 = df.Open.values[1:-2]
	close3 = df.Close.values[1:-2]
	low3 = df.Low.values[1:-2]
	high3 = df.High.values[1:-2]
	open4 = df.Open.values[:-3]
	close4 = df.Close.values[:-3]
	low4 = df.Low.values[:-3]
	high4 = df.High.values[:-3]

	for o1,h1,l1,c1,o2,c2,h2,l2,o3,c3,l3,h3,o4,c4,l4,h4 in zip(open1,high1,low1,close1,open2,close2,high2,low2,open3,close3,low3,high3,open4,close4,low4,high4):
		

		if((l4 < l3) and (h4 > h3) and (l3 < l2) and (h3 > h2) and (c1 > o1) and (h1 > h3) and (l1 > l4)):
			
			UpTriplePregnantType.append(1)
		
		else:
			
			UpTriplePregnantType.append(0)
			
		if((l4 < l3) and (h4 > h3) and (l3 < l2) and (h3 > h2) and (c1 < o1) and (l1 < l3) and (h1 < h4)):
		
			DownTriplePregnantType.append(1)
		
		else:
			
			DownTriplePregnantType.append(0)

	df['UpTriplePregnantType'] = UpTriplePregnantType
	df['DownTriplePregnantType'] = DownTriplePregnantType

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