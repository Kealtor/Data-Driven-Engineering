from os import name
import keras_tuner as kt 
import keras
import tensorflow as tf
from tensorflow import keras
from keras import optimizers, models, layers, regularizers
import pandas as pd 
from sklearn import preprocessing as pp
from utils.dde_preprocessing_utils import (set_timestamp_index,
remove_duplicates,
remove_columns,
extract_correlations,
prepare_data_sarimax,
create_test_train_split,
create_sarimax_test_train_split,
create_eval_df,
convert_to_sample_time_feature)
from utils.dde_plotting_utils import(
display_double_timestamps,
display_seasonal_decomposition,
plot_predictions,
plot_accuracy,
plot_error_variations,
plot_learning_curves,
)
class HypermodelLSTM(kt.HyperModel):
    
    def __init__(self,
                 train_start= '2015-01-01',
                 validation_start='2017-01-01',
                 test_start='2017-03-01',
                 test_end='2017-03-14',
                 feature= 'total load actual',
                 modeltype= 'LSTM',
                 namespace= 'MS_LSTM_HPT'):
        
        self.train_start = train_start
        self.validation_start = validation_start
        self.test_start = test_start
        self.test_end = test_end
        self.feature = feature
        self.modeltype = modeltype
        self.namespace = namespace
        
        self.load_data()
        
        self.scale_data()
        
        self.feature_df = self.df[[self.feature]]
        self.data_x_train0,self.data_x_val0,self.data_x_test0,self.data_y_train0,self.data_y_val0,self.data_y_test0 = create_test_train_split(self.scaled_df,self.feature_df,self.train_start,self.test_start,self.test_end,validation_start=self.validation_start)
        
    def load_data(self):
        self.df = pd.read_csv('preprocessed_data.csv')
        self.df = set_timestamp_index(self.df,"Unnamed: 0")
        pass
    
    def scale_data(self):
        self.scaled_df = self.df.copy(deep=True)
        featuresToScale = self.scaled_df.columns
        sX = pp.StandardScaler(copy=True)
        self.scaled_df.loc[:,featuresToScale] = sX.fit_transform(self.scaled_df[featuresToScale])
    
    def build(self, hp):
        
        self.data_x_train,self.data_y_train,self.idx_train = convert_to_sample_time_feature(self.data_x_train0,self.data_y_train0,hp.Int('past_days',min_value=24,max_value=168,step=24),24,32)
        self.data_x_test,self.data_y_test,self.idx_test = convert_to_sample_time_feature(self.data_x_test0,self.data_y_test0,hp.Int('past_days',min_value=24,max_value=168,step=24),24,32)
        self.data_x_val,self.data_y_val,self.idx_val = convert_to_sample_time_feature(self.data_x_val0,self.data_y_val0,hp.Int('past_days',min_value=24,max_value=168,step=24),24,32)
        
        n_timesteps, n_features, n_outputs = self.data_x_train.shape[1], self.data_x_train.shape[2], self.data_y_train.shape[1]
        model = models.Sequential()
        
        model.add(layers.LSTM(hp.Int('Neurons',min_value=16,max_value=96,step=16), return_sequences=True,input_shape=(None, n_features), stateful=False)) 
        model.add(layers.BatchNormalization())

        for i in range(hp.Int('n_layers',1,4,1)):
            model.add(layers.LSTM(hp.Int('Neurons',min_value=16,max_value=96,step=16), return_sequences=True, stateful=False))
            model.add(layers.BatchNormalization())
            
        model.add(layers.LSTM(hp.Int('Neurons',min_value=16,max_value=96,step=16), stateful=False))
        
        model.add(layers.Dense(hp.Int('Neurons',min_value=16,max_value=96,step=16), kernel_regularizer=regularizers.l2(0.0001), kernel_initializer="he_normal"),)
        model.add(layers.LeakyReLU(alpha=0.1))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(n_outputs))                     
        
        model.compile(optimizer='Adam',loss='mse',metrics='mae') 
        return model