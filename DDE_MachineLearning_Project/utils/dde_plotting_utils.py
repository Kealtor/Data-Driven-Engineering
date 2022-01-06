import numpy as np
import pandas as pd
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt
from .dde_preprocessing_utils import process_weather_csv
months_list = ['January','February','March','April','May','June','July','August','September','October','November','December']

def display_double_timestamps():
    """
    Returns double Timestamps from Weather Dataset
    """
    df_cities = process_weather_csv()
    df_city_dup=[]
    for i,item in enumerate(df_cities):
        df_city_dup.append(item[item.index.duplicated(keep=False)])
    return df_city_dup

def display_seasonal_decomposition(df,feature,window,mode):
    #taking load column:
    data = df[feature]#.sort_index()
    data.index = data.index.tz_convert('UTC')
    data.index.freq= 'h'
    window = data.rolling(window=window)
    new_temp_dataframe = pd.concat([window.min(),window.median(), window.max(), data], axis=1)
    new_temp_dataframe.columns = ['min', 'median', 'max', 'temp']
    new_temp_dataframe.fillna(0,inplace=True)
    decomposition = sm.tsa.seasonal_decompose(new_temp_dataframe[mode], model = 'additive')
   
    fig = decomposition.plot()
    return fig

def plot_predictions(eval_df,horizon,name,feature,path):
    fig = plt.figure();
    plt.plot(eval_df['Actual'], 'k.-');
    plt.plot(eval_df['Prediction'], 'x', alpha=0.70);
    plt.legend(['Actual',
                ('Predicted with ' + str (horizon) + ' hr horizon')])
    plt.ylabel(feature);
    plt.xlabel('Time Index');
    plt.title(name);
    fig.savefig(path)
    return fig

def plot_accuracy(eval_df,name,path):
    fig = plt.figure()
    plt.axes(aspect='equal')
    plt.scatter(eval_df['Actual'], eval_df['Prediction'],marker='*',alpha=0.80)
    plt.xlabel(f'True Values')
    plt.ylabel(f'Predicted Values')
    lims = [20000, 40000]
    plt.xlim(lims), plt.ylim(lims)
    plt.plot(lims, lims)
    plt.title(name);
    fig.savefig(path)
    return fig

def plot_error_variations(eval_df,name,path):
    # Calculating the error variations:
    error = (eval_df['Prediction'] - eval_df['Actual'])/eval_df['Actual']*100
    fig = plt.figure()
    plt.hist(error, bins=30,alpha=0.9)
    plt.xlabel('Predicted Relative % Error')
    plt.ylabel('Count')
    plt.title(name);
    fig.savefig(path)
    return fig 

# Defining our function to see the evolution of error:
def plot_learning_curves(history,name,path):
    #We will omit the first 10 points for a better visualization:
    fig = plt.figure()
    plt.plot(history['epoch'],history['loss'], "k--", linewidth=1.5, label="Training")
    plt.plot(history['epoch'],history['val_loss'], "b-.", linewidth=1.5, label="Validation")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epochs"),  plt.ylabel("Loss (MSE)")
    plt.title(name);
    fig.savefig(path)
    return fig