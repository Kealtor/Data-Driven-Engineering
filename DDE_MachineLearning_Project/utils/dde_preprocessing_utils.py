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
from scipy.signal import savgol_filter

from sklearn.metrics import r2_score
months_list = ['January','February','March','April','May','June','July','August','September','October','November','December']

def set_timestamp_index(df,time_col):
    """
    Creates a timestamp (datetime-object) column and sets it as an index 
    """

    datetime_series =  pd.to_datetime(df[time_col],utc=True)
    
    datetime_index = pd.DatetimeIndex(datetime_series.values,tz='UTC')
    df=(df.set_index(datetime_index)).drop([time_col],axis=1)
    df.index = df.index.tz_convert("Europe/Berlin")
    return df

def reorder_columns(df,order):
    """
    Reorders the columns by given order
    """

    new_order= order + [x for x in df.columns if x not in order]
    df = df.reindex(columns=new_order)
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



def prepare_data_sarimax(data_y_train,data_y_test,horizon,training_window,feature,exog_feature=None,exog_y_train=None,exog_y_test=None):
    """
    Preparing data for SARIMAX model
    """
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
    #Creating predictions to store the model outcomes: -> list
    predictions = list()
    
    if exog_feature!= None:
        #Exogenous variables: -> df
        history_temp = exog_y_train[exog_feature]
        history_temp = history_temp[(-training_window):]
        predictions_temp = exog_y_test[exog_feature]
        return test_ts,history,predictions,history_temp,predictions_temp
    else:
        return test_ts,history,predictions
            

def create_test_train_split(df_x=None,df_y=None,train_start=None,test_start=None,test_end=None,validation_start=None):
    """
    Creating a test-train-data-split
    """
    # df_x.index.freq = 'h'
    # df_y.index.freq = 'h'
        
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


def create_split(df_x=None,df_y=None,start=None,end=None):
    """
    Creating a test-train-data-split
    """
    # df_x.index.freq = 'h'
    # df_y.index.freq = 'h'
        
    data_x = df_x.copy()[(df_x.index >= start) & (df_x.index < end)]
    data_y = df_y.copy()[(df_y.index >= start) & (df_y.index < end)]
    
    return data_x,data_y

def create_sarimax_test_train_split(df,feature,exog_feature,train_start=None,test_start=None,test_end=None,):
    """
    Creating a test-train-data-split for SARIMAX model
    """
    data_y_train = df.copy()[(df.index >= train_start)& (df.index < test_start)][[feature]]
    data_y_test = df.copy()[(df.index >= test_start) & (df.index < test_end)][[feature]]
    data_exog_train = df.copy()[(df.index >= train_start)& (df.index < test_start)][[exog_feature]]
    data_exog_test =df.copy()[(df.index >= test_start) & (df.index < test_end)][[exog_feature]]
    return data_y_train,data_y_test,data_exog_train,data_exog_test
    pass

def create_eval_df(predictions,data_y_test):
    """
    Creating a evaluation dataframe  to compare predictions and actual data 
    """
    eval_df = pd.DataFrame(predictions)
    eval_df.columns = ['Prediction']
    eval_df = (eval_df.set_index(data_y_test.index))
    eval_df['Actual'] = data_y_test
    return eval_df


def convert_to_sample_time_feature(data_x,data_y,n_inputs,n_outputs,batch_size):
    """
    Converting test-train-data-split into (sample,time,feature)-format for Recurrent Neural Networks.
    """
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

def convert_from_differencing(Y_test_predictions,base_df,feature):
    idx_init = Y_test_predictions.index[0] - timedelta(hours=1)
    init_val = base_df.loc[idx_init,feature]
    Y_test_predictions.loc[idx_init] = init_val
    Y_test_predictions.sort_index(inplace=True)
    Y_test_predictions = Y_test_predictions.cumsum()[1:]
    return Y_test_predictions


def make_multiple_predictions(model,idx_test,data_x_test,base_df,feature,convert=False):
    idx_prediction = pd.date_range(start=idx_test[0],freq='h',periods=len(idx_test)+23)
    y_test_filtered = base_df.loc[idx_prediction][feature]
    df_pred =pd.DataFrame(index=y_test_filtered.index)
        
    for i in range(len(data_x_test)):
        case_test = data_x_test[i].reshape((1,data_x_test[0].shape[0], data_x_test[0].shape[1]))
        date_range = pd.date_range(start=idx_test[i],freq='h',periods=24)
        Y_test_predictions = pd.Series(model.predict(case_test).flatten(),index=date_range)
        if convert:
            Y_test_predictions = convert_from_differencing(Y_test_predictions,base_df,feature)
        df_pred[f'pred_{i}'] = Y_test_predictions
        
    df_pred['mean'] = df_pred.mean(axis=1)
    df_pred['Actual'] = y_test_filtered
    return df_pred

def calculate_r2_scores(df_pred):
    r2_list=[]
    for column in df_pred.columns:
        if column not in ['Actual','mean']:
            pred = df_pred[column].dropna()
            act = df_pred.loc[pred.index,'Actual']
            r2= r2_score(pred,act)
            r2_list.append(r2)
    return r2_list      

def apply_differencing(df,excluded):
    included = [x for x in df.columns if x not in excluded]
    included_df = df[included].diff().dropna()
    excluded_df = df[excluded][1:]
    merged_df = pd.concat([included_df,excluded_df],axis=1)
    return merged_df

def apply_savgol_filter(df,window,order,deriv=0,excluded=[]):
    included = [x for x in df.columns if x not in excluded]
    included_df = pd.DataFrame(savgol_filter(df[included],window,order,deriv,axis=0), index = df.index, columns=included)
    excluded_df = df[excluded]
    merged_df = pd.concat([included_df,excluded_df],axis=1)
    return merged_df
