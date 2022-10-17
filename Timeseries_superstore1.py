# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:10:14 2022

@author: tarun
"""

from statsmodels.tsa.api import SimpleExpSmoothing,Holt, ExponentialSmoothing
import pandas as pd
import os


os.chdir("D:\\Data science PDF\\Time series\\SuperStore.csv")
# FullRaw = pd.read_csv("super Store.csv")
FullRaw = pd.read_csv("SuperStore.csv",encoding="ISO-8859-1")

FullRaw.head()
#############
# check for NA values

FullRaw.isnull().sum()

#########

FullRaw2 = FullRaw.groupby('Order Date')['Sales'].sum().reset_index().copy()
FullRaw2.head()
FullRaw2["Order Date"].dtypes
FullRaw2.shape
#Ideally we should have 1460 rows or days[4years*365days]. But we don,t
# so everyday's sales data is Not present. That means we need to roll-up further
# Time series canot have NA values.

#############
# convert date column to "datetime" column
#############

#Before Further roll-up lets first convert order date column
# into a "Date" format column and we will set it s the index
# as well, as that helps in plotting/ conducting time series analysis

FullRaw2['Order Date']= pd.to_datetime(FullRaw2['Order Date'])
FullRaw2.sort_values('Order Date', inplace = True)
FullRaw2.set_index('Order Date', inplace=True)


# setting the index to "datetime" helps in plotting time series


FullRaw2.index.min()
FullRaw2.index.max()

##############
#Further date roll up 
###########

# The current date is at a daily level(with lot of 'Missing days').
#In time series analysis, it s usually advisible
# to roll up/ aggregate the data at a higher , like week or month
# We will consider averaging(rollup) the sales at monthly level
#(but, you can also try weekly level roll up in the same manner)


FullRaw3 = FullRaw2['Sales'].resample('MS').mean() #ms stand for for month start
#you should get 48 rows(4 years*) 
#####
#plot the data
####

import seaborn as sns
sns.lineplot(data= FullRaw3)

###2####
#Time  Series decomposition
#######

from statsmodels.tsa.seasonal import seasonal_decompose
Decomposed_series = seasonal_decompose(FullRaw3)
Decomposed_series.plot()

##########
#sampling: Train and Test split
##########

Train = FullRaw3[:36].copy() # First 3 years for training(2014-1016)
Test = FullRaw3[36:].copy() #Last 1 year for testing(2017)

##########
# SES Model
##########

## Model
SES = SimpleExpSmoothing(Train).fit(smoothing_level=0.01) # Model building
SES.summary()
Forecast = SES.forecast(12).rename('Forecast') # Model Forecasting
Actual_Forecast_DF = pd.concat([FullRaw3, Forecast], axis=1)

# Act, Forecast combining

## plot
import seaborn as sns
sns.lineplot(data= Actual_Forecast_DF)

#validation
import numpy as np
Validation_DF = Actual_Forecast_DF[-12:].copy()
np.mean(abs(Validation_DF['Sales']-Validation_DF['Forecast'])/Validation_DF['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_DF['Sales'] - Validation_DF['Forecast'])**2)) # RMSE

########
#DES Model(Double smoothining model)
########

##Model
DES = Holt(Train).fit(smoothing_level=0.01, smoothing_slope=0.6)
DES.summary()
Forecast = DES.forecast(12).rename('Forecast')
Actual_Forecast_DF = pd.concat([FullRaw3, Forecast],axis= 1)

## plot
sns.lineplot(data = Actual_Forecast_DF)
Validation_DF = Actual_Forecast_DF[-12:].copy()
np.mean(abs(Validation_DF['Sales']-Validation_DF['Forecast'])/Validation_DF['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_DF['Sales'] - Validation_DF['Forecast'])**2)) # RMSE


#########
# TES Model(Triple exponential smoothing )
#########

## Model
TES = ExponentialSmoothing(Train,
                           seasonal_periods= 12,
                           seasonal='add',
                           trend= 'add').fit(smoothing_level=0.01,
                                             smoothing_slope=0.1,
                                             smoothing_seasonal = 0.3)

                
TES.summary()
# trend = 'add' means additive trend. use this when trend is NOT eponentially increasing/decreasing,
# like a steep increase or decrease
# seasonal = 'add' means additive seasonality. Use this when seasonality is NOT increasing/decreasing in magnitude
Forecast = TES.forecast(12).rename('Forecast')
Actual_Forecast_DF = pd.concat([FullRaw3, Forecast], axis=1)


## plot
sns.lineplot(data = Actual_Forecast_DF)

# Validation
Validation_DF = Actual_Forecast_DF[-12:].copy()
np.mean(abs(Validation_DF['Sales']-Validation_DF['Forecast'])/Validation_DF['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_DF['Sales'] - Validation_DF['Forecast'])**2)) # RMSE

#########
#TES: Automatic selection of level,trend and seasonal smoothing components (alpha, beta,gamma)
########

# Model
TES2 = ExponentialSmoothing(Train,
                            seasonal_periods = 12,
                            seasonal='add',
                            trend='add').fit() #Leave fit() empty, then it will automatically find alpha, beta, gamma

Forecast = TES2.forecast(12).rename('Forecast')
Actual_Forecast_DF = pd.concat([FullRaw3, Forecast],axis= 1)

## plot
sns.lineplot(data = Actual_Forecast_DF)

## Validation
Validation_DF = Actual_Forecast_DF[-12:].copy()
np.mean(abs(Validation_DF['Sales']-Validation_DF['Forecast'])/Validation_DF['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_DF['Sales'] - Validation_DF['Forecast'])**2)) # RMSE

##########
#TES Grid Search
#########

myAlpha = np.round(np.arange(0,1.1,0.1),2)
myBeta = np.round(np.arange(0,1.1,0.1),2)
myGamma = np.round(np.arange(0,1.1,0.1),2)

alphaList = []
betaList = []
gammaList = []
mapeList = []

for alpha in myAlpha:
    for beta in myBeta:
        for gamma in myGamma:
            
            print(alpha,beta,gamma)
            
            TES = ExponentialSmoothing(Train,
                                       seasonal_periods=12,
                                       seasonal='mul',
                                       trend= 'add').fit(smoothing_level = alpha,
                                                         smoothing_slope = beta,
                                                         smoothing_seasonal = gamma)
            Forecast = TES.forecast(12).rename('Forecast')                        
            Actual_Forecast_DF = pd.concat([FullRaw3, Forecast], axis=1)
            Validation_DF = Actual_Forecast_DF[-12:].copy()
            tempMAPE = np.mean(abs(Validation_DF['Sales']- Validation_DF['Forecast'])/Validation_DF['Sales'])*100

            alphaList.append(alpha)
            betaList.append(beta)
            gammaList.append(gamma)
            mapeList.append(tempMAPE)

evaluationDf = pd.DataFrame({'alpha': alphaList,
                             'beta': betaList,
                             'gamma': gammaList,
                             'Mape': mapeList})
#evaluationDf.to_csv("EvaluationDf.csv")
###########
###########
# ARIMA Manual Model
###########
# pip install pmdarima
#pip install pmdarima
from pmdarima.arima import ARIMA

# Model
arimaModel = ARIMA((2,1,0),(1,0,0,12)).fit(Train)
Forecast = pd.Series(arimaModel.predict(12)).rename('Forecast')
Forecast
Forecast.index = Test.index # Needed for the pd.concat to work correctly in the next line
Actual_Forecast_DF = pd.concat([FullRaw3, Forecast],axis= 1)

## plot
sns.lineplot(data = Actual_Forecast_DF)

## Validation
Validation_DF = Actual_Forecast_DF[-12:].copy()
np.mean(abs(Validation_DF['Sales']-Validation_DF['Forecast'])/Validation_DF['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_DF['Sales'] - Validation_DF['Forecast'])**2)) # RMSE


#############
# ARIMA Automated Model
############

from pmdarima import auto_arima
arimaModel2 = auto_arima(Train, m=12)

## Get the order of p,d,q & P,D, Q
arimaModel2.get_params()['order'] # p,d,q
arimaModel2.get_params()['seasonal_order'] # P,D,Q


## Forecasting
Forecast = pd.Series(arimaModel2.predict(12).rename('Forecast'))
Forecast.index = Test.index
Actual_Forecast_DF = pd.concat([FullRaw3,Forecast],axis=1)


##plot
sns.lineplot(data= Actual_Forecast_DF)

## Validation
Validation_DF = Actual_Forecast_DF[-12:].copy()
np.mean(abs(Validation_DF['Sales']-Validation_DF['Forecast'])/Validation_DF['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_DF['Sales'] - Validation_DF['Forecast'])**2)) # RMSE


#######
# Adding one parameters
####
# TrainCopy = Train.copy()
sns.lineplot(data = Train)
trainLog = np.log(Train) # take log of series

sns.lineplot(data= trainLog)

## Model 
# if you believe that the series is stationary, then its worth a shot
arimaModel3 = auto_arima(trainLog, m = 12, stationary= True)

## Get the order of p, d, q, & P, D, Q
arimaModel3.get_params()['order']
arimaModel3.get_params()['seasonal_order']


## Forecasting
Forecast = pd.Series(arimaModel3.predict(12)).rename('Forecast')
Forecast = np.exp(Forecast) # De - log the forecasted vlues (back to original scale)
Forecast.index = Test.index # De-log the forecasted values (back to original scale)
Actual_Forecast_DF = pd.concat([FullRaw3,Forecast],axis=1)


##plot
sns.lineplot(data= Actual_Forecast_DF)

## Validation
Validation_DF = Actual_Forecast_DF[-12:].copy()
np.mean(abs(Validation_DF['Sales'] - Validation_DF['Forecast'])/Validation_DF['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_DF['Sales'] - Validation_DF['Forecast'])**2)) # RMSE


#########
# ARIMA Grid Search
#############

p = range(0,2)
d = range(0,2)
q = range(0,2)
P = range(2)
D = range(2)
Q = range(2)


pList = []
dList = []
qList = []
PList = []
DList = []
QList = []
mapeList = []
seasonalPDQList = []

for i in p:
    for j in d:
        for k in q:
            for I in P:
                for J in D:
                    for K in Q:
                        print(i,j,k,I,J,K)
                        tempArimaModel = ARIMA((i,j,k), (I,J,K,12)).fit(Train)
                        
                        Forecast = pd.Series(tempArimaModel.predict(12).rename('Forecast'))
                        Forecast.index = Test.index
                        Actual_Forecast_DF = pd.concat([FullRaw3, Forecast], axis= 1)
                        Validation_DF = Actual_Forecast_DF[-12:].copy()
                        tempMAPE = np.mean(abs(Validation_DF['Sales'] - Validation_DF['Forecast'])/Validation_DF['Sales'])*100 # MAPE
                        
                        pList.append(i)
                        dList.append(j)
                        qList.append(k)
                        PList.append(I)
                        DList.append(J)
                        QList.append(K)
                        mapeList.append(tempMAPE)

arimaEvaluationDF = pd.DataFrame({'p': pList,
                                  'd': dList,
                                  'q': qList,
                                  'P': PList,
                                  'D': DList,
                                  'Q': QList,
                                  'MAPE': mapeList})  

arimaEvaluationDF.to_csv("EvaluationDf.csv")

# Finalize arima full model (This is very important)

arimaFinalModel = ARIMA((0,1,1), (0,1,0,12)).fit(FullRaw3)

# Forecasting
Forecast = pd.Series(arimaFinalModel.predict(12)).rename('Forecast')

# set the correct dates as index of the forecast obtained in the previous line
start = '2018-01-01' # Check the order/ date format in FullRaw3.index. Its year-Month-Day
end = '2018-12-01'
futureDateRange = pd.date_range(start, end, freq='MS')
futureDateRange
Forecast.index = futureDateRange

Actual_Forecast_DF = pd.concat([FullRaw3, Forecast],axis=1) # Column wise binding

## plot
sns.lineplot(data=Actual_Forecast_DF)


