import numpy as np
import pandas as pd

def SetFeaturesForDayLevelDF(df_D):

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