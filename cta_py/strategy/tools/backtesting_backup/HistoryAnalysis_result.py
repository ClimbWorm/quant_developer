import pandas as pd
import numpy as np
import datetime

#按照日groupby来计算sharpratio
def SharpRatioDay(history,rf = 0):
	'''
	history为历史交易记录表，rf为无风险利率，默认为0
	'''
	sharpdata = history.copy()
	sharpdata['date'] = [sharpdata.EP_GMT.loc[i].strftime('%Y-%m-%d') for i in range(len(sharpdata))]
	sharpdata
	table = pd.pivot_table(sharpdata, values=['CP_Price', 'Profits'], index=['date'],
						aggfunc={'CP_Price': np.mean,
								 'Profits': np.sum})
	rate_of_return = table.Profits/table.CP_Price

	sharpratio = (rate_of_return.mean() - rf)/rate_of_return.std()
	return sharpratio

#按照周groupby来计算sharpratio
def SharpRatioWeek(history,rf = 0):
	'''
	history为历史交易记录表，rf为无风险利率，默认为0
	'''
	sharpdata = history.copy()
	sharpdata['date'] = [sharpdata.EP_GMT.loc[i].strftime('%Y-%m-%d') for i in range(len(sharpdata))]
	sharpdata['date'] = pd.to_datetime(sharpdata.date)
	start = sharpdata.EP_GMT.loc[0].strftime('%Y-%m-%d')
	end = (sharpdata.EP_GMT.loc[len(sharpdata)-1] + datetime.timedelta(days=7)).strftime('%Y-%m-%d')#这边可以再加7天
	d = pd.date_range(start = start, end = end, freq='W')
#     print(d)
	DateTime1 = 0
	DateTime2 = 1
	weektag = 1
	week = []
	for i in range(len(sharpdata)):
#         print(week)
	
		if sharpdata.loc[i].date < d[DateTime1]:
			week.append(0)
			continue
		elif (sharpdata.loc[i].date >= d[DateTime1]) & (sharpdata.loc[i].date < d[DateTime2]):
			week.append(weektag)
		else:
			week.append(weektag + 1)
			DateTime1 += 1
			DateTime2 += 1
			weektag += 1
	sharpdata['week'] = week
	#去掉标记为0的和weektag的（因为可能数据不全）
	sharpdata = sharpdata[(sharpdata.week != int(0)) & (sharpdata.week != int(weektag))]
	table = pd.pivot_table(sharpdata, values=['CP_Price', 'Profits'], index=['week'],
						aggfunc={'CP_Price': np.mean,
								 'Profits': np.sum})
	rate_of_return = table.Profits/table.CP_Price

	sharpratio = (rate_of_return.mean() - rf)/rate_of_return.std()
	return sharpratio
		
def GenerateStrategyReport(history):

	if(len(history) > 0):
	
		TotalNetProfit = np.sum(history.Profits)
		ProfitFactor = np.sum(history.loc[history.Profits > 0].Profits)/(-np.sum(history.loc[history.Profits < 0].Profits)+1e-6)
		PercentProfitable = len(history.loc[history.Profits > 0])/len(history)
		AverageTradeNetProfit = np.sum(history.Profits)/len(history)
		sharpRatioDay = SharpRatioDay(history)
		#sharpRatioWeek = SharpRatioWeek(history)
		
		def GetR_squared(history):
			from scipy import stats
			Y = [0.0]
			X = [0.0]
			cumulativeProfits = 0
			i = 1.0
			for p in history.Profits.values:
				cumulativeProfits += p
				Y.append(cumulativeProfits)
				X.append(i)
				i+=1    
			return stats.linregress(X,Y)[2] ** 2

		R_squared = GetR_squared(history)
		
		return {'TotalNetProfit':TotalNetProfit, 'ProfitFactor':ProfitFactor, 'PercentProfitable':PercentProfitable,\
					'AverageTradeNetProfit':AverageTradeNetProfit, 'R_squared':R_squared, 'sharpRatioDay':sharpRatioDay}
	else:

		return {'TotalNetProfit':None, 'ProfitFactor':None, 'PercentProfitable':None,\
					'AverageTradeNetProfit':None, 'R_squared':None, 'sharpRatioDay':None} 





		

