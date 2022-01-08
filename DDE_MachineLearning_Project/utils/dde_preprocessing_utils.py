import numpy as np
import pandas as pd
from datetime import date
from pandas.core.indexes.datetimes import date_range
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import timedelta
months_list = ['January','February','March','April','May','June','July','August','September','October','November','December']

def set_timestamp_index(df,time_col):
    """
    Creates a timestamp (datetime-object) column and sets it as an index 
    """

    datetime_series =  pd.to_datetime(df[time_col],utc=True)
    #datetime_series = datetime_series.tz_convert("Europe/Berlin")
    
    datetime_index = pd.DatetimeIndex(datetime_series.values,tz='UTC')
    df=(df.set_index(datetime_index)).drop([time_col],axis=1)
    df.index = df.index.tz_convert("Europe/Berlin")
    return df

def reorder_columns(df,order):
    """
    Reorders the columns by given order
    """

    new_order= order + [x for x in df.columns if x not in order]
    df = df.reindex(columns=new_order)#axis='columns',)
    return df

def remove_duplicates(df):
    """"
    Removes duplicate indecies from dataframe
    """

    df = df[~df.index.duplicated(keep='first')]
    return df

def process_weather_csv():
    """
    Loads weather_features.csv and preprocesses the data
    """

    df_weather = pd.read_csv('weather_features.csv')
    df_weather = set_timestamp_index(df_weather,'dt_iso')
    cities = df_weather.city_name.unique()
    df_cities = [df_weather.query(f"city_name == '{x}'") for x in cities]
    return df_cities

def process_energy_csv():
    """
    Loads energy_dataset.csv and preprocesses the data
    """

    df_energy = pd.read_csv('energy_dataset.csv')
    df_energy = set_timestamp_index(df_energy,'time') 
    return df_energy

def average_over_year(df):
    """
    Filters the dataframe and groups them by their month day and averages them. 
    Returns list with dataframes for each month
    """

    months_list = ['January','February','March','April','May','June','July','August','September','October','November','December']
    
    return df.groupby(df.index.month).mean()
    
def average_over_day(df):
    """
    Filters the dataframe and groups them by their specific data and averages them.
    Returns list with dataframe for each month and every year.
    """

    months_list = ['January','February','March','April','May','June','July','August','September','October','November','December']
    return [df.query(f"month == '{x}'").groupby('date').mean() for x in months_list]
    
def remove_columns(df,columns):
    """
    Removes columns from dataframe
    """
    for item in columns:
        df.drop(item,inplace=True,axis=1)
    return df

def extract_correlations(correlationMatrix,upper_treshhold,lower_treshhold):
    """
    Returns a list with tuples containing correlation-parameter and value
    """
    correlations = []
    for i,row in correlationMatrix.iterrows():
        
        for idx,value in row.iteritems():
            if value>=upper_treshhold or value <= lower_treshhold:
                correlations.append((i,idx, value))
                
    filtered_correlations = [x for x in correlations if x[0]!=x[1]]
    filtered_correlations = sorted(filtered_correlations, key= lambda tup: tup[2])
    filtered_correlations = filtered_correlations[::2]
    return filtered_correlations



def prepare_data_sarimax(data_y_train,data_y_test,exog_y_train,exog_y_test,horizon,training_window,feature,exog_feature):
    test_shifted = data_y_test.copy()
    #preparing the shifted test data:
    for t in range(1, horizon):
        test_shifted[feature+'+'+str(t)] = test_shifted[feature].shift(-t, freq='H')
    test_shifted = test_shifted.dropna()
    # Predictions on test data: -> df
    training_window = training_window
    train_ts = data_y_train[feature]
    test_ts = test_shifted

    #Creating the history -> df
    history = train_ts.copy()
    history = history[(-training_window):]

    #Exogenous variables: -> df
    history_temp = exog_y_train[exog_feature]
    history_temp = history_temp[(-training_window):]
    predictions_temp = exog_y_test[exog_feature]

    #Creating predictions to store the model outcomes: -> list
    predictions = list()
    
    return test_ts,history,predictions,history_temp,predictions_temp
    

def create_test_train_split(df_x=None,df_y=None,train_start=None,test_start=None,test_end=None,validation_start=None):
    # Train-Test Split:
    # We will split the data as training and test. No need for validation here. 
    df_x.index.freq = 'h'
    df_y.index.freq = 'h'
        
    data_x_test = df_x.copy()[(df_x.index >= test_start) & (df_x.index < test_end)]
    data_y_test = df_y.copy()[(df_y.index >= test_start) & (df_y.index < test_end)]

    if validation_start !=None:
        data_x_train = df_x.copy()[(df_x.index >= train_start)& (df_x.index < validation_start)]
        data_y_train = df_y.copy()[(df_y.index >= train_start)& (df_y.index < validation_start)]
            
        data_x_val =  df_x.copy()[(df_x.index >= validation_start)& (df_x.index < test_start)]
        data_y_val =  df_y.copy()[(df_y.index >= validation_start) & (df_y.index < test_start)]
        return data_x_train,data_x_val,data_x_test,data_y_train,data_y_val,data_y_test,
    else:
        data_x_train = df_x.copy()[(df_x.index >= train_start)& (df_x.index < test_start)]
        data_y_train = df_y.copy()[(df_y.index >= train_start)& (df_y.index < test_start)]
        return data_x_train,data_x_test,data_y_train,data_y_test


def create_sarimax_test_train_split(df,feature,exog_feature,train_start=None,test_start=None,test_end=None,):
    data_y_train = df.copy()[(df.index >= train_start)& (df.index < test_start)][[feature]]
    data_y_test = df.copy()[(df.index >= test_start) & (df.index < test_end)][[feature]]
    data_exog_train = df.copy()[(df.index >= train_start)& (df.index < test_start)][[exog_feature]]
    data_exog_test =df.copy()[(df.index >= test_start) & (df.index < test_end)][[exog_feature]]
    return data_y_train,data_y_test,data_exog_train,data_exog_test
    pass

def create_eval_df(predictions,data_y_test):
    eval_df = pd.DataFrame(predictions)
    eval_df.columns = ['Prediction']
    eval_df = (eval_df.set_index(data_y_test.index))
    eval_df['Actual'] = data_y_test
    return eval_df


def convert_to_sample_time_feature(data_x,data_y,n_inputs,n_outputs,batch_size):
    remainder = len(data_y)%batch_size
    first_date_index = data_y.index[0] 
    last_date_index = data_y.index[-1] - timedelta(hours = remainder+n_outputs)
    date_index_range = pd.date_range(start=first_date_index,end=last_date_index,freq='H')

    data_x = data_x[first_date_index:last_date_index].copy()
    data_y = data_y[first_date_index:last_date_index].copy()
    X,Y = list(), list()
   
    for idx in date_index_range[:-(n_inputs+n_outputs)]:
        start = idx 
        end_x = idx + timedelta(hours=n_inputs-1)
        start_y = idx + timedelta(hours=n_inputs)
        end_y = idx + timedelta(hours=n_inputs-1+n_outputs)
        X.append(data_x[start:end_x].to_numpy())
        Y.append(data_y[start_y:end_y].to_numpy().flatten())
    date_index_range =date_index_range[n_inputs:]
    date_index_range = date_index_range[:len(Y)]
    return np.array(X),np.array(Y), date_index_range 



# def convert_to_rnn_train_test_split(data_x_train,data_y_train,n_inputs,n_outputs,batch_size):
#     remainder = len(data_y_train)%batch_size
#     data_x_train = data_x_train[:-remainder]
#     data_y_train = data_y_train[:-remainder]
#     print(data_x_train)
#     pass