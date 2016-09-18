import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
def TypeMapper(type):
  if type == 'A':
    return 1
  elif type == 'B':
    return 2
  elif type == 'C':
    return 3

#Getting DataSets from the Input
dfStore = pd.read_csv('data/stores.csv')
dfTrain = pd.read_csv('data/train.csv')
dfTest = pd.read_csv('data/test.csv')
dfFeatures = pd.read_csv('data/features.csv')

submission = pd.read_csv('data/sampleSubmission.csv')

#Merging Store with the Input 
dfTrainTmp = pd.merge(dfTrain, dfStore)
dfTestTmp = pd.merge(dfTest, dfStore)

#Merging Features with the Input
Train = pd.merge(dfTrainTmp, dfFeatures)
Test = pd.merge(dfTestTmp, dfFeatures)


#Murging Train Data
Train['year'] = pd.to_numeric(Train['Date'].str[0:4])
Train['month'] = pd.to_numeric(Train['Date'].str[5:7])
Train['day'] = pd.to_numeric(Train['Date'].str[8:10])
Train['days'] = (Train['month'] -1) * 30 + Train['day']

Train['Type'] = Train['Type'].apply(TypeMapper)
Train['IsHoliday'] = Train['IsHoliday'].apply(lambda x: 1 if x == True else 0)

Train['dayHoliday'] = Train['IsHoliday'] * Train['days']

Train['logsales'] = np.log(4990 + Train['Weekly_Sales'])
Train['tDays'] = 360 * (Train['year'] - 2010 ) + (Train['month'] - 1) * 30 + Train['day']

Train['days30'] = (Train['month'] -1 ) * 30 + Train['day']

#Murging Test Data
Test['year'] = pd.to_numeric(Test['Date'].str[0:4])
Test['month'] = pd.to_numeric(Test['Date'].str[5:7])
Test['day'] = pd.to_numeric(Test['Date'].str[8:10])

Test['days'] = (Test['month'] - 1) * 30 + Test['day']

Test['Type'] = Test['Type'].apply(TypeMapper)
Test['IsHoliday'] = Test['IsHoliday'].apply(lambda x: 1 if x == True else 0)

Test['dayHoliday'] = Test['IsHoliday'] * Test['days']
Test['tDays'] = 360 * (Test['year'] - 2010) + (Test['month'] - 1) * 30 + Test['day']

Test['days30'] = (Test['month'] - 1 ) * 30 + Test['day']

tmpR0 = len(submission.index)

j = 0

while(j < tmpR0):
  print(j / tmpR0)

  tmpId = submission['Id'][j]
  tmpStr = tmpId.split('_')
  tmpStore = int(tmpStr[0])
  tmpDept = int(tmpStr[1])
  dataF1 = Train[Train['Dept'] == tmpDept]
  tmpL = len(dataF1[dataF1['Store'] == tmpStore].index)
  tmpF = dataF1[dataF1['IsHoliday'] == 1]
  dataF1 = dataF1.append(pd.concat([tmpF] * 4))
  dataF2 = dataF1[dataF1['Store'] == tmpStore]

  testF1 = Test[Test['Dept'] == tmpDept]
  testF1 = testF1[testF1['Store'] == tmpStore]
 
  testRows = len(testF1.index)
  clf = RandomForestRegressor(n_estimators = 4800,max_features = 4, bootstrap = True)
  if(tmpL < 10):
    columns = list(dataF1.columns[i] for i in [5, 6,16, 17, 18, 19, 20, 22, 23])

    y = np.array(dataF1['logsales'], dtype= np.float64)
    clf.fit(dataF1[columns],y )
  else:
    columns = list(dataF2.columns[i] for i in [6,16, 17, 18, 19, 20, 22, 23])
    y = np.array(dataF2['logsales'], dtype= np.float64)
    clf.fit(dataF2[columns],y )
  
  tmpP = np.exp(clf.predict(testF1[columns]))-4990 
  
  k = j + testRows
  submission['Weekly_Sales'][j:k] = tmpP
  j = k

submission.to_csv('data/outputPy.csv')
