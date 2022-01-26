import numpy as np
import pandas as pd
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sympy import N
from .dde_preprocessing_utils import process_weather_csv
from sklearn.metrics import r2_score
import sklearn
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
    fig.figsize=(16,8)
    return fig

def plot_accuracy(eval_df,ax):
    ax.aspect='equal'
    ax.scatter(eval_df['Actual'], eval_df['Prediction'],marker='*',alpha=0.80)
    ax.set_xlabel(f'True Values')
    ax.set_ylabel(f'Predicted Values')
    minimum = min(eval_df['Actual'].min(),eval_df['Prediction'].min())
    maximum = max(eval_df['Actual'].max(),eval_df['Prediction'].max())
    lims = [minimum-abs(minimum)*0.1, maximum+abs(maximum)*0.1]
    ax.xlim = lims 
    ax.ylim = lims
    ax.plot(lims, lims)
    ax.set_title('Accuracy')
    return ax

def plot_error_variations(eval_df,ax):
    error = (eval_df['Prediction'] - eval_df['Actual']) #/eval_df['Actual']*100
    ax.hist(error, bins=30,alpha=0.9)
    ax.set_xlabel('Predicted Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Variations')
    
    return ax 

def plot_predictions(eval_df,ax):
    r2_test = r2_score(eval_df['Actual'][:-1], eval_df['Prediction'][:-1])
    print('R^2 score is  %3.2f' %r2_test)
    ax.plot(eval_df['Actual'], 'k.-')
    ax.plot(eval_df['Prediction'],'rx-', alpha=0.70)
    ax.legend(['Actual','Predicted'])
    ax.set_xlabel('Time Index')
    ax.set_title('Predictions with R^2 score of  %3.2f' %r2_test)
    return ax

def plot_evaluation(eval_df):
    fig= plt.figure(figsize = (16,16))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])

    ax1 = plot_accuracy(eval_df,ax1)
    ax2 = plot_error_variations(eval_df,ax2)
    ax3 = plot_predictions(eval_df,ax3)
    actual = 'Actual'
    if 'mean' in eval_df.columns:
        prediction ='mean'
    else:
        prediction = 'Prediction'
    mse = sklearn.metrics.mean_squared_error(eval_df[actual][:-1],eval_df[prediction][:-1])
    mape = sklearn.metrics.mean_absolute_percentage_error(eval_df[actual][:-1],eval_df[prediction][:-1])*100
    mae = sklearn.metrics.mean_absolute_error(eval_df[actual][:-1],eval_df[prediction][:-1])
    
    ax3.text(0.025,0.85,f'RMSE: {round(np.sqrt(mse),2)} \nMAE: {round(mae,2)} \nMAPE: {round(mape,2)}',bbox={
        'facecolor': 'grey', 'alpha': 0.5, 'pad': 10},transform=ax3.transAxes)
    # ax3.set_text()
    return fig


# Defining our function to see the evolution of error:
def plot_learning_curves(history,name,path):
    #We will omit the first 10 points for a better visualization:
    fig = plt.figure(figsize=(8.0,8.0))
    plt.plot(history['epoch'],history['loss'], "k--", linewidth=1.5, label="Training")
    plt.plot(history['epoch'],history['val_loss'], "b-.", linewidth=1.5, label="Validation")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epochs"),  plt.ylabel("Loss (MSE)")
    plt.title(name);
    fig.savefig(path)
    return fig

   
def plot_multiple_predictions(df_pred,path,add_traces = False):
    fig,ax = plt.subplots(1,figsize=(16,8))
    ax.plot(df_pred.index,df_pred['Actual'])
    ax.plot(df_pred.index,df_pred['mean'])
    r2_test = r2_score(df_pred['Actual'][:-1],df_pred['mean'][:-1])
    ax.set_title('Multiple averaged predictions with an R^2 score of:  %3.2f' %r2_test)
    if add_traces:
        for column in df_pred.columns:
            if column not in ['Actual','mean']:
                ax.plot(df_pred[column],'o')
    fig.savefig(path)
    return fig
    pass
