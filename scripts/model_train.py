import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import os
from datetime import timedelta,datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt



def  process_rhpi():

    #read in rhpi data

    rhpi = pd.read_csv(f'{script_dir}/../data/rhpi.csv',encoding="utf-16",sep='\t')

    #subset data for los angeles

    la_rhpi = rhpi[rhpi['Region Name'] == "Los Angeles, CA"][['Month, Year of Date','Redfin HPI MoM']]

    #convert types, rename, and sort index
    la_rhpi['Month, Year of Date'] = pd.to_datetime(la_rhpi['Month, Year of Date'])
    la_rhpi.rename(columns={'Month, Year of Date':'Date'},inplace=True)
    la_rhpi.sort_values('Date',inplace=True)
    la_rhpi.set_index('Date',inplace=True)

    #fix formatting
    la_rhpi['Redfin HPI MoM'] = la_rhpi['Redfin HPI MoM'].apply(lambda x:float(str(x).replace('%','')))

    #make time series just the RHPI month over month
    la_rhpi = la_rhpi['Redfin HPI MoM']

    la_rhpi.dropna(inplace=True)

    original_la_rhpi = la_rhpi.copy()
    #get last date in time series
    last_date = la_rhpi.index[-1].strftime('%Y-%m-%d')


    return la_rhpi,original_la_rhpi,last_date

def process_redfin_city_data(last_date):

    #import housing related information by city

    info = pd.read_csv(f'{script_dir}/../data/city_market_tracker.tsv000.gz',sep='\t',compression='gzip')

    #take subset for los angeles and take mean of variables across different periods and cities to get one mean per period for LA
    la_info = info[(info['parent_metro_region'] == 'Los Angeles, CA') & (info['property_type'] == 'Single Family Residential')]
    features = la_info.groupby(['period_begin']).mean()[['median_sale_price_mom','median_list_price_mom','median_ppsf_mom','median_list_ppsf_mom','homes_sold_mom','new_listings_mom','inventory_mom','sold_above_list_mom','price_drops_mom','median_dom_mom','months_of_supply_mom','avg_sale_to_list_mom']]

    #make index datetime type and set date to match rhpi
    features.index = pd.to_datetime(features.index)
    features = features[(features.index >= '2012-04-01') & (features.index <= last_date)]

    return features

def process_mortage_data(last_date):

    mortgage = pd.read_csv(f'{script_dir}/../data/MORTGAGE30US.csv')

    #set date in datetime format as index
    mortgage['DATE'] = pd.to_datetime(mortgage['DATE'])
    mortgage.set_index('DATE', inplace=True)

    #fill missing values that are '.' values with previous value
    mortgage = mortgage.replace('.',pd.NA).ffill()

    mortgage['MORTGAGE30US'] = mortgage['MORTGAGE30US'].apply(lambda x:float(x))

    #convert weekly data to monthly data and make every datetime the beginning of the month
    mortgage = mortgage.resample('M').mean()
    mortgage.index = mortgage.index.to_period('M').to_timestamp('D')

    #select dates to match rhpi data
    mortgage = mortgage[(mortgage.index >= '2012-04-01') & (mortgage.index <= last_date)]

    mortgage = mortgage['MORTGAGE30US']

    return mortgage

def process_unemployment(last_date):

    #read in unemployment data
    unemp = pd.read_csv(f'{script_dir}/../data/LOSA106UR.csv')
    unemp.rename(columns={'LOSA106UR':'unemployment rate'},inplace=True)

    #change date variable to datetime and make it the index
    unemp['DATE'] = pd.to_datetime(unemp['DATE'])
    unemp.set_index('DATE',inplace=True)

    #subset series to match la rhpi data
    unemp = unemp[(unemp.index >= '2012-04-01') & (unemp.index <= last_date)]
    unemp = unemp['unemployment rate']

    return unemp

def normalize_series(series):

    mean, std = series.mean(),series.std()
    series = (series - mean)/std

    return series


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.realpath(__file__))

    #read in and process data sources
    print('reading in data')
    la_rhpi,original_la_rhpi,last_date = process_rhpi()
    features = process_redfin_city_data(last_date)
    mortgage = process_mortage_data(last_date)
    unemp = process_unemployment(last_date)
    print('finished reading')

    important_features = ['median_list_price_mom','median_ppsf_mom','median_list_ppsf_mom','sold_above_list_mom','avg_sale_to_list_mom']

    #time series cleaning

    #normalize features 
    la_rhpi = normalize_series(la_rhpi)
    mortgage = normalize_series(mortgage)
    unemp = normalize_series(unemp)

    for feature in important_features:
        features[feature] = normalize_series(features[feature])

    # take first difference of series where it is necssary to remove trend
    mortgage = mortgage.diff().dropna()
    unemp = unemp.diff().dropna()

    print('finished cleaning')
    #modeling

    model_df = features[important_features].copy()
    model_df['mortgage'] = mortgage
    model_df['unemployemnt'] = unemp
    model_df['la_rhpi'] = la_rhpi

    #remove first column because of differencing
    model_df = model_df[model_df.index > '2012-04-01']

    model_df = model_df.asfreq('MS') 


    #data setup
    train_cutoff= round(len(model_df) * 0.80)

    train_data = model_df[:train_cutoff]
    test_data = model_df[train_cutoff:]


    #rolling forecast performance results
    rolling_predictions = {}
    for train_end in test_data.index:
        train_data = model_df[:train_end-timedelta(days=1)]
        model = VAR(train_data)
        model_fit = model.fit(maxlags=3)
        last_observations = train_data.values[-3:]
        pred = model_fit.forecast(last_observations, steps=1)[0][-1]
        rolling_predictions[train_end] = pred

    rolling_predictions = pd.Series(rolling_predictions).transpose()
    
    mean_absolute_error_val = sum(abs(test_data['la_rhpi'] - rolling_predictions))/len(test_data['la_rhpi'])

    #final model fit and forecasting

    model = VAR(model_df)
    model_fit = model.fit(maxlags=3)
    last_observations = train_data.values[-3:]
    pred = model_fit.forecast(last_observations, steps=3)
    forecast = pred[:,-1]

    forecast, lower_bounds, upper_bounds = model_fit.forecast_interval(last_observations, steps=3, alpha=0.05)

    forecast = forecast[:,-1]
    lower_bounds = lower_bounds[:,-1]
    upper_bounds = upper_bounds[:,-1]

    forecast = pd.Series(forecast, index=pd.date_range(start=model_df.index[-1] + relativedelta(months=1), periods=3, freq='M'))
    upper_bounds = pd.Series(upper_bounds, index=pd.date_range(start=model_df.index[-1] + relativedelta(months=1), periods=3, freq='M'))
    lower_bounds = pd.Series(lower_bounds, index=pd.date_range(start=model_df.index[-1] + relativedelta(months=1), periods=3, freq='M'))

    forecast.index = forecast.index.to_period('M').to_timestamp('D')

    #add last value of forecast to end of la_rhpi for clean forecast plot
    date_str = la_rhpi.index[-1:][0].strftime('%Y-%m-%d')
    forecast[date_str] = model_df.loc[date_str,'la_rhpi']
    forecast.sort_index(inplace=True)
    
    #denormalize forecast to plot on original scale

    mean, std = original_la_rhpi.mean(),original_la_rhpi.std()
    denormalized_forecast = (forecast * std) + mean


    #update metric spreadsheet

    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y%m%d')
    
    metric_df = pd.DataFrame({'Date':formatted_date,'Pred Date':denormalized_forecast.index[1],'Forecast Pred':denormalized_forecast[1],'MAE':mean_absolute_error_val},index=[0])
    metric_df.to_csv(f'{script_dir}/../metrics/model_metrics.csv',index=False,mode='a')


    #create forecast plot
    plt.style.use('Solarize_Light2')

    plt.figure(figsize=(20,8))
    plt.plot(original_la_rhpi)
    plt.plot(denormalized_forecast)
    plt.legend(('LA RHPI', 'Forecast'), fontsize=16)
    plt.title(f'Live Forecast of RHPI ({formatted_date})', fontsize=20)
    plt.ylabel('Percent MoM', fontsize=16)

    plt.savefig(f'{script_dir}/../forecast_plots/{formatted_date}_forecast.png', dpi=300, bbox_inches='tight')

    #create zoomed forcast plot

    zoomed_series = original_la_rhpi[-12:]

    plt.figure(figsize=(20,8))
    plt.plot(zoomed_series)
    plt.plot(denormalized_forecast)
    plt.legend(('LA RHPI', 'Forecast'), fontsize=16)
    plt.title(f'Live Forecast of RHPI ({formatted_date})', fontsize=20)
    plt.ylabel('Percent MoM', fontsize=16)

    plt.savefig(f'{script_dir}/../forecast_plots/{formatted_date}_zoomed_forecast.png', dpi=300, bbox_inches='tight')


    print('complete!')
    
