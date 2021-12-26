# usage in cmd line:
# 
# python RNA-Protein_predict_v4.3_performance_metric.py  [dataset number,range 0~19]
# 
# note: this script load the prediction result pickle file of a datasets, 
# then comput performance metric for each model, save a new pickle file.
# using 5-fold CV to benchmark on 20 datasets with multiple models, including 
# 6 classical machine learning models, 
# 3 Neural Network models, 
# 4 Ensemble methods: stacking/boosting/bagging, 
# and 3 existing methods: baseline Elastic Net, teamHYU, teamHL&YG

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from multiprocessing import Pool
from functools import partial
from copy import deepcopy
import time, sys

from sklearn import linear_model, metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import pickle
from scipy import stats

import mkl
mkl.set_num_threads(1)
nCPU= 10
N=5

model_dict = {
    "LR": linear_model.LinearRegression(), 
    "Lasso": linear_model.Lasso(alpha=0.02, max_iter=1e5), 
    "HR": linear_model.HuberRegressor(), #Linear regression model that is robust to outliers.
    "Ridge": linear_model.Ridge(), 
    "SVR": SVR( gamma='scale'),
    "RFR": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0), 
    # NN : construct later, because feature length have to be set 
    "NN1": None,
    "NN2": None,
    "NN3": None,
    ##### other ensemble models
    "Stacking": None,
    "Voting": None,
    "Boosting": None,
    "Bagging": None, 
    ##### reimplement published methods
    "baselineEN": linear_model.ElasticNet(l1_ratio = 0.5, random_state = 0, precompute=True), 
    "teamHYU": RandomForestRegressor(n_estimators=100, random_state=0), # Author not providing detail, use default
    "teamHL&YG": None, ## equivalent to RFR without feature selection
    }

list_model = list(model_dict.keys())
method_feature_select = ["cosine", "raw_cosine", "spearmanr", "random", "custom"]
list_metric =  ['r', 'rmse', 'mae']

list_topn = [10, 20, 50, 100, 200, 500, 1000, 5000, 'All']

file = '/media/eys/xwj/proteome/data/20211022_dict_matrix_20dataset.pkl'
with open(file, 'rb') as f:
    [ dict_dataset, df_summary]=pickle.load(f)

mykey = df_summary.index[np.int32(sys.argv[1])]

filelist = pd.read_csv("/media/eys/xwj/proteome/data/res_pkl20211220.list", sep="\t", low_memory=False, header=0, index_col=0)
df_summary.loc[:, "resultpkl"] = filelist.loc[ df_summary.index, 'resultpkl']
df_summary

print('df_time_list_topn & df_metric_list_topn: ')
print(mykey, end=":")

with open( '/media/eys/xwj/proteome/data/' + df_summary.loc[mykey, "resultpkl"], 'rb') as f:
    [ res ]=pickle.load(f)

X = dict_dataset[mykey]["RNA"].transform(lambda x: (x-x.mean())/x.std(), axis=1).round(3).transpose()
Y = dict_dataset[mykey]["protein"].transform(lambda x: (x-x.mean())/x.std(), axis=1).round(3).transpose()

df_template = pd.DataFrame(index= Y.columns, columns= range(N))
print(mykey, X.shape, Y.shape, df_template.shape)
##############
for mfs in method_feature_select[::-1]: 
    my_list_topn = list(res[mfs].keys())
    for topn in my_list_topn:
        if topn == df_summary.loc[mykey, 'totalRNA']:
            my_topn = 'All'
        else:
            my_topn = topn

        for m in list_model:
            if m in list_model[-3:]: ## if baselineEN, team1 and team2, skip feature selection
                if (mfs == method_feature_select[-1]) & (my_topn == 'All'):
                    pass
                else:
                    continue
            else: ## my models don't do 'custom'
                if (mfs == method_feature_select[-1]):
                    continue
                
                    
            print(time.ctime(), mfs.ljust(8,'-'), str(topn).ljust(8,'-'), m.ljust(10,'-'), end=":  ")
            print("training split ", end=" ")
            
            for metric in list_metric:
                res[mfs][topn][m]["df_CVtest_"+ metric] = deepcopy(df_template)
                res[mfs][topn][m]["df_CVtrain_"+ metric] = deepcopy(df_template)
                
            kf = KFold(n_splits=N, random_state=1, shuffle=True)
            i = 0  
            for train_index, test_index in kf.split(Y): # random trials to stablize errors
                X_train, X_val = X.iloc[train_index], X.iloc[test_index]
                Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]

                y_pred =  pd.DataFrame(data=np.array( res[mfs][topn][m]['y_pred_CVtrain'][i]).T, index=Y_train.index, columns =Y.columns)
                res[mfs][topn][m]["df_CVtrain_mae"].loc[:, i] = [ metrics.mean_absolute_error(Y_train[g], y_pred[g]) for g in Y.columns]
                res[mfs][topn][m]["df_CVtrain_rmse"].loc[:, i] = [ np.sqrt(metrics.mean_squared_error(Y_train[g], y_pred[g])) for g in Y.columns]
                res[mfs][topn][m]["df_CVtrain_r"].loc[:, i] = [ np.corrcoef(Y_train[g], y_pred[g])[1,0] for g in Y.columns]

                y_pred =  pd.DataFrame(data=np.array( res[mfs][topn][m]['y_pred_CVtest'][i]).T, index=Y_val.index, columns =Y.columns)
                res[mfs][topn][m]["df_CVtest_mae" ].loc[:, i] = [ metrics.mean_absolute_error(Y_val[g], y_pred[g]) for g in Y.columns]
                res[mfs][topn][m]["df_CVtest_rmse"].loc[:, i] = [ np.sqrt(metrics.mean_squared_error(Y_val[g], y_pred[g])) for g in Y.columns]
                res[mfs][topn][m]["df_CVtest_r"].loc[:, i]  = [ np.corrcoef(Y_val[g], y_pred[g])[1,0] for g in Y.columns]

                i = i + 1;  print(i, end=" ");  # end for split i

            print('done')

## save complete res
outfile = '/media/eys/xwj/proteome/data/res_v4.3_'+ mykey + '.pkl'
with open(outfile, 'wb') as f: 
    pickle.dump( [res],  f)