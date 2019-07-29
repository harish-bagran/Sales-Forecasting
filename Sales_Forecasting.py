# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:32:15 2019

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:09:52 2019

@author: Harish Bagran
"""


# Importing Dataset
import pandas as pd
data = pd.read_csv("C:\\Users\\dell\\Desktop\\project7\\data\\sales_project7_modified.csv")
data.iloc[:,2].value_counts()

''' Order Date is not in one format. 
Saperating Order Day, Month and Year in different columns, then again combine them in a
single column in one single format. '''

data['day'] = data['Order Date'].str[0:2]
data['month'] = data['Order Date'].str[3:5]
data['year'] = data['Order Date'].str[6:]

data.loc[data['year'].str.len()==2, 'year'] = '20'+ data['year']
data['year'] = data['year'].str[0:5]


# dd-mm-yyyy string format in one column
data['odatestring'] =  data['day'] +'/'+ data['month'] + '/' + data['year'] 
data.dtypes

# filling NaN values by zero
data.dropna(inplace = True)

# Droping extra columns
sales_dates = data.drop(['Order Date','Order Type','Type','State','day','month','year'], axis=1)

'''
# don't keep the file open in background
sales_dates.to_excel(r'C:\Users\dell\Desktop\project7\data\sales_dates.xlsx')
'''
# check missing values
sales_dates.isnull().sum()
 

#converting string to Date and time object; sales to integer
sales_dates['odatestring'] = pd.to_datetime(sales_dates['odatestring'], dayfirst = True)
sales_dates['Sales'] = sales_dates['Sales'].astype(float)
sales_dates.dtypes

#renaming column name
sales_dates = sales_dates.rename(columns={'odatestring':'order_date'})

'''
checking sales data before regression
sales_dates.to_csv(r'C:\Users\dell\Desktop\project7\data\sales_bf_aggre.csv')
'''

# creating index on date column to work resample
sales_dates = sales_dates.set_index('order_date')

# aggregating sales Day wise using resample method
sales_day_wise = sales_dates.resample('D').sum()


# 21-Aug-2018 to 31-Aug-2018 sales are 0. Replacing them by 1. B'coz taking log of 0 is infinite 
sales_day_wise.loc[sales_day_wise['Sales']==0, 'Sales'] = 1
sales_day_wise.dtypes

'''
checking sales data after regression
sales_day_wise.to_csv(r'C:\Users\dell\Desktop\project7\data\sales_af_aggre.csv')

sales_day_wise.to_excel(r'C:\Users\dell\Desktop\project7\data\sales_af_aggre.xlsx')
'''

### End Of EDA Part ###

##################################
#####   Exploring Series   #######
##################################

# seasonal_decompose expects timeseries column as index
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_data = seasonal_decompose(sales_day_wise['Sales'], model = 'additive')
decompose_data.plot()
decompose_data = seasonal_decompose(sales_day_wise['Sales'], model = 'multiplicative')
decompose_data.plot()
'''
Inferences from above plot:
    Trend : Linear
    Seasonality: Additive (M=6 or M=7)
'''

#ACF Plots and PACF Plots on orignal Data
import statsmodels.graphics.tsaplots as tsa_plts
tsa_plts.plot_acf(sales_day_wise['Sales'], lags=10)
tsa_plts.plot_pacf(sales_day_wise['Sales'], lags=10)

'''
Inferences from above plot:
    ACF lag = 1
    Moving average window = 5
'''
# Splitting Train and Test Data
Train = sales_day_wise.head(406)
Test = sales_day_wise.tail(30)


# Creating a MAPE function
import numpy as np

def MAPE(pred, orig):
    temp = np.abs((pred-orig)/orig)*100
    return np.mean(temp)

##########################
## Smoothing Techniques ##
##########################
    
# Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model = SimpleExpSmoothing(Train['Sales']).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
mape_ses = MAPE(pred_ses, Test['Sales']) #52.92 %

# Holt method only trend
from statsmodels.tsa.holtwinters import Holt
holt_model = Holt(Train['Sales']).fit()
pred_holt = holt_model.predict(start = Test.index[0], end = Test.index[-1])
mape_holt = MAPE(pred_holt, Test['Sales']) #57.83%

# Winter method with both trend and Seasonality
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#winter model with additive seasonality = 7
winter_model_7 = ExponentialSmoothing(Train['Sales'], seasonal = 'add', trend = 'add', seasonal_periods = 7).fit()
pred_winter_model_7 = winter_model_7.predict(start = Test.index[0], end = Test.index[-1])
mape_winter_model_7 = MAPE(pred_winter_model_7, Test['Sales']) #29.71% 


#winter model with additive seasonality = 6
winter_model_6 = ExponentialSmoothing(Train['Sales'], seasonal = 'add', trend = 'add', seasonal_periods = 6).fit()
pred_winter_model_6 = winter_model_6.predict(start = Test.index[0], end = Test.index[-1])
mape_winter_model_6 = MAPE(pred_winter_model_6, Test['Sales']) #54.24%

#winter model with additive seasonality = 5
winter_model_5 = ExponentialSmoothing(Train['Sales'], seasonal = 'add', trend = 'add', seasonal_periods = 5).fit()
pred_winter_model_5 = winter_model_5.predict(start = Test.index[0], end = Test.index[-1])
mape_winter_model_5 = MAPE(pred_winter_model_5, Test['Sales']) #60.52%

#winter model with multiplicative seasonality = 7
winter_model_7_mul = ExponentialSmoothing(Train['Sales'], seasonal = 'mul', trend = 'add', seasonal_periods = 7).fit()
pred_winter_model_7_mul = winter_model_7_mul.predict(start = Test.index[0], end = Test.index[-1])
mape_winter_model_7_mul = MAPE(pred_winter_model_7_mul, Test['Sales']) #24.28% (best)



import pylab
x = pd.date_range('2019-05-12', periods=30, freq='D')

pylab.plot(x, pred_winter_model_7_mul,':r', label = 'predicted')
pylab.plot(x, Test['Sales'],'-b', label = 'actual')
pylab.xlabel('Days')
pylab.ylabel('Sales per day')
pylab.title('Actual_Sales V/s Predicted_Sales')
pylab.legend(loc='upper left')
pylab.show()


winter_model_7_mul_resi = pd.DataFrame(winter_model_7_mul.resid)
tsa_plts.plot_acf(winter_model_7_mul_resi, lags=12) # 2 is showing significance

#Using ARIMA forecasting errors for Lag =2
winter_model_7_mul_resi = winter_model_7_mul_resi.reset_index(drop=True)
from statsmodels.tsa.arima_model import ARIMA
resi_winter_model_7_mul = ARIMA(winter_model_7_mul_resi[0], order=(2,0,0)).fit(transparams=True)
tsa_plts.plot_acf(resi_winter_model_7_mul.resid, lags=12) # No significant lags
winter_forecast_errors = resi_winter_model_7_mul.forecast(steps = 30)[0]

# Final improved predictions by adding Predicitons and errors
winters_predictions = pd.DataFrame(columns = ['forecast_sales','forecast_errors','improved'])
winters_predictions['forecast_sales'] = pd.Series(pred_winter_model_7_mul)
winters_predictions = winters_predictions.reset_index(drop=True)
winters_predictions['forecast_errors'] = pd.Series(winter_forecast_errors)
winters_predictions['improved'] = winters_predictions['forecast_sales']+winters_predictions['forecast_errors']
Test = Test.reset_index(drop=True)
mape_winter_model_7_mul_AR = MAPE(winters_predictions['improved'], Test['Sales']) #24.82%

###############################
######   ARIMA Model   ########
###############################

from statsmodels.tsa.arima_model import ARIMA

#ARIMA for Lag=1 and window_size = 2
arima_m2 = ARIMA(Train['Sales'], order=(1,1,2)).fit(transparams=True)

#ARIMA for Lag=1 and window_size = 5
arima_m5 = ARIMA(Train['Sales'], order=(1,1,5)).fit(transparams=True)

#residuals of residuals
arima_m2_resi = pd.DataFrame(arima_m2.resid)
arima_m5_resi = pd.DataFrame(arima_m5.resid)

# ACF Plot of Residuals of Residuals
tsa_plts.plot_acf(arima_m2_resi, lags=12)
tsa_plts.plot_acf(arima_m5_resi, lags=12)
#Showing Significance at Lag = 7 

arima_m2_resi_lag7 = ARIMA(arima_m2_resi[0], order=(7,0,0)).fit(transparams=True)
arima_m5_resi_lag7 = ARIMA(arima_m5_resi[0], order=(7,0,0)).fit(transparams=True)

#residuals of residuals of residuals
arima_m2_resi_lag7_resi = pd.DataFrame(arima_m2_resi_lag7.resid)
arima_m5_resi_lag7_resi = pd.DataFrame(arima_m5_resi_lag7.resid)

# ACF Plot of Residuals of Residuals of Residuals
tsa_plts.plot_acf(arima_m2_resi_lag7_resi, lags=12)
tsa_plts.plot_acf(arima_m5_resi_lag7_resi, lags=12)
# No significant lags #

# Predicting values using ARIMA model

# Moving Average = 2
arima_ma2_forecast_values =  arima_m2.forecast(steps = 30)[0]
arima_ma2_forecast_errors = arima_m2_resi_lag7.forecast(steps = 30)[0]

arima_predictions = pd.DataFrame(columns = ['forecast_sales','forecast_errors','improved'])
arima_predictions['forecast_sales'] = pd.Series(arima_ma2_forecast_values)
arima_predictions['forecast_errors'] = pd.Series(arima_ma2_forecast_errors)
arima_predictions['improved'] = arima_predictions['forecast_sales']+arima_predictions['forecast_errors']

Test = Test.reset_index(drop=True)
mape_arima_ma2 = MAPE(arima_predictions['improved'], Test['Sales']) #48.31


# Moving Average = 5
arima_ma5_forecast_values =  arima_m5.forecast(steps = 30)[0]
arima_ma5_forecast_errors = arima_m5_resi_lag7.forecast(steps = 30)[0]

arima_predictions_5 = pd.DataFrame(columns = ['forecast_sales','forecast_errors','improved'])
arima_predictions_5['forecast_sales'] = pd.Series(arima_ma5_forecast_values)
arima_predictions_5['forecast_errors'] = pd.Series(arima_ma5_forecast_errors)
arima_predictions_5['improved'] = arima_predictions_5['forecast_sales']+arima_predictions_5['forecast_errors']

mape_arima_ma5 = MAPE(arima_predictions_5['improved'], Test['Sales']) #65.81


##################################
######   Pre-Processing   ########
##################################

import numpy as np
sales_day_wise['t'] = np.arange(1,437)
sales_day_wise['t_sq'] = sales_day_wise['t'] * sales_day_wise['t']
sales_day_wise['log_sales'] = np.log(sales_day_wise['Sales'])

# rearranging column order
sales_day_wise = sales_day_wise[['t','t_sq','Sales','log_sales']]

#Time Plot
import matplotlib.pyplot as plt
plt.plot(sales_day_wise['t'],sales_day_wise['Sales'])
plt.title('Sales Days Wise')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.show()


#resetting index
sales_day_wise = sales_day_wise.reset_index()

# Splitting Train and Test Data
Train = sales_day_wise.head(406)
Test = sales_day_wise.tail(30)
Test = Test.set_index(np.arange(1,31))

##################################
######   Building Models   #######
##################################

import statsmodels.formula.api as smf

## Linear Model ##
linear = smf.ols('Sales~t', data = Train).fit()
pred_linear = pd.Series(linear.predict(Test['t']))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
mape_linear = MAPE(pred_linear, Test['Sales']) #45.10%


## Exponential Model ##
expo = smf.ols('log_sales~t', data = Train).fit()
pred_expo = pd.Series(expo.predict(Test['t']))
pred_expo = np.exp(pred_expo)
rmse_expo = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_expo))**2))
rmse_expo
mape_linear = MAPE(pred_expo, Test['Sales']) #45.17%

## Quadratic Model ##
quad = smf.ols('Sales~t+t_sq', data = Train).fit()
pred_quad = pd.Series(quad.predict(Test.iloc[:,1:3]))
rmse_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_quad))**2))
mape_quad = MAPE(pred_quad, Test['Sales']) #44.99%



#Building final model with whole Data set : winter model with multiplicative seasonality = 7
sales_day_wise = sales_dates.resample('D').sum()
sales_day_wise.loc[sales_day_wise['Sales']==0, 'Sales'] = 1
sales_day_wise.dtypes
from statsmodels.tsa.holtwinters import ExponentialSmoothing
final_model = ExponentialSmoothing(sales_day_wise['Sales'], seasonal = 'mul', trend = 'add', seasonal_periods = 7).fit()

# creating Dates for next 30 days to be forcasted
date_range = pd.date_range('2019-06-11', periods=30, freq='D')
next_30_days = pd.DataFrame(index=date_range)  

# Forecasting Sales of next 30 days using Final model
final_predictions = final_model.predict(start = next_30_days.index[0], end = next_30_days.index[-1])
final_predictions = final_predictions.astype(int)
final_predictions
