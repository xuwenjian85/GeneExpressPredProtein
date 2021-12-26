#!/usr/bin/env python
# coding: utf-8

# usage in cmd line:
# 
# python RNA-Protein_predict_v4.3.py  [dataset number,range 0~19]
# 
# note: this script do model fit, prediction, but not comput performance metric
# using 5-fold CV to benchmark on 20 datasets with multiple models, including 
# 6 classical machine learning models, 
# 3 Neural Network models, 
# 4 Ensemble methods: stacking/boosting/bagging, 
# and 3 existing methods: baseline Elastic Net, teamHYU, teamHL&YG

# Python [conda env:py3.7.3_skorch]
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from multiprocessing import Pool
from functools import partial
from copy import deepcopy
import time, os, sys

from sklearn import linear_model, metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import StackingRegressor, VotingRegressor, AdaBoostRegressor, BaggingRegressor

import pickle
import mkl
mkl.set_num_threads(1)
nCPU= 50
N=5
np.random.seed(2021)

import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

torch.manual_seed(0)
torch.cuda.manual_seed(0)


## load datasets
file = '/media/eys/xwj/proteome/data/20210926_dict_matrix_20dataset.pkl'
with open(file, 'rb') as f:
    [ dict_dataset, df_summary]=pickle.load(f)
datalist = list(dict_dataset.keys())

mykey = datalist[ np.int32(sys.argv[1]) ]

### team_HYU methods 
# load feature selection matrix, p1, p2. ~ 60% RNAs, 80% proteins in df_nb PPI neighbor table
file = '/media/eys/xwj/proteome/proteome_prediction_team_hyu/df_selected_part1-220211126-1529.txt'
df_nb = pd.read_csv( file, sep='\t', index_col=0 )
print(df_nb.shape)

# mykey = datalist[4]

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

basic_estimators = [
    ("LR",  model_dict['LR']), 
    ("Lasso", model_dict['Lasso']), 
    ("HR",  model_dict['HR']),
    ("Ridge", model_dict['Ridge']), 
    ("SVR", model_dict['SVR']),
    ("RFR", model_dict['RFR']), 
    ]
 
model_dict['Stacking'] = StackingRegressor(estimators=basic_estimators)
model_dict['Voting'] = VotingRegressor(basic_estimators)
model_dict['Boosting']  = AdaBoostRegressor(random_state=0)
model_dict['Bagging'] = BaggingRegressor(random_state=0)
    
list_model = list(model_dict.keys())
print( model_dict, list_model )
method_feature_select = ["cosine", "raw_cosine", "spearmanr", "random", "custom"]

# feature numbers to be prioritized in feature selection 
list_topn = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 200, 1000, 5000 ] # v1
list_topn = [5, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 5000 ] # v2
list_topn = [10, 20, 50, 100, 200, 500, 1000, 5000  ] # v3

#############
def create_res(list_topn):
    res = {}
    for mfs in method_feature_select:
        res[mfs] = {} 
        for topn in list_topn:
            # summary table
            res[mfs][topn] = {}
            res[mfs][topn]["time"] = pd.DataFrame(data = 0, dtype = np.int32, index = model_dict, columns= ["val","train","test"])
            # detail data for all models
            for m in model_dict:
                res[mfs][topn][m]={}
                res[mfs][topn][m]["y_pred_CVtrain"], res[mfs][topn][m]["y_pred_CVtest"] ={}, {}
    return res

def comp_y_pred(p): ### this function fit a classical machine learning model and pred y value
    # usage: comp_y_pred("ENSG00000000419") 
    select=X_train.columns[ df_topn[p] ]
    X_val_select, X_train_select, y_val, y_train = X_val[select], X_train[select], Y_val[p], Y_train[p]
    my_model=model_dict[m].fit(X_train_select, y_train) #'''fit model with selected features'''
    # training fit and test fit
    return my_model.predict(X_train_select), my_model.predict(X_val_select)
    
class RegressorModule3(nn.Module):## 10, 10, 10
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
            n_feature=10,
    ):
        super(RegressorModule3, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(n_feature,  num_units)
        self.dense1 = nn.Linear(num_units, num_units)
        self.dense2 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.nonlin(self.dense1(X))
        X = self.nonlin(self.dense2(X))
        X = self.output(X)
        return X
    
class RegressorModule2(nn.Module):## 50, 10
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
            n_feature=10,
    ):
        super(RegressorModule2, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(n_feature,  num_units*5)
        self.dense1 = nn.Linear(num_units*5, num_units)
        self.output = nn.Linear(num_units, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X

class RegressorModule1(nn.Module): ## 10, 10
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
            n_feature=10,
    ):
        super(RegressorModule1, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(n_feature,  num_units)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X

def comp_y_pred_nn3(p): ### this function fit a deep Neural network model and pred y value
    select=X_train.columns[ df_topn[p] ]
    X_val_select, X_train_select, y_val, y_train = X_val[select], X_train[select], Y_val[p], Y_train[p]
    #'''fit model with selected features'''
    my_model = NeuralNetRegressor( RegressorModule3(n_feature = topn ), max_epochs=10, lr=0.01, verbose=0, train_split=False)
    
    my_model.fit(X_train_select.values.astype(np.float32), y_train.values.astype(np.float32).reshape(-1,1))
    return np.nan_to_num( my_model.predict(X_train_select.values.astype(np.float32))),                 np.nan_to_num( my_model.predict(X_val_select.values.astype(np.float32)))

def comp_y_pred_nn2(p): ### this function fit a deep Neural network model and pred y value
    select=X_train.columns[ df_topn[p] ]
    X_val_select, X_train_select, y_val, y_train = X_val[select], X_train[select], Y_val[p], Y_train[p]
    #'''fit model with selected features'''
    my_model = NeuralNetRegressor( RegressorModule2(n_feature = topn ), max_epochs=10, lr=0.01, verbose=0, train_split=False)
    
    my_model.fit(X_train_select.values.astype(np.float32), y_train.values.astype(np.float32).reshape(-1,1))
    return np.nan_to_num( my_model.predict(X_train_select.values.astype(np.float32))),                 np.nan_to_num( my_model.predict(X_val_select.values.astype(np.float32)))

def comp_y_pred_nn1(p): ### this function fit a deep Neural network model and pred y value
    select=X_train.columns[ df_topn[p] ]
    X_val_select, X_train_select, y_val, y_train = X_val[select], X_train[select], Y_val[p], Y_train[p]
    #'''fit model with selected features'''
    my_model = NeuralNetRegressor( RegressorModule1(n_feature = topn ), max_epochs=10, lr=0.01, verbose=0, train_split=False)
    
    my_model.fit(X_train_select.values.astype(np.float32), y_train.values.astype(np.float32).reshape(-1,1))
    return np.nan_to_num( my_model.predict(X_train_select.values.astype(np.float32))),                 np.nan_to_num( my_model.predict(X_val_select.values.astype(np.float32)))

def comp_y_pred_baselineEN(p): ### this function fit a baseline Elastic Net model and pred y value
    ## best lambda parameter, Constant that multiplies the penalty term
    # baseline_EN, a ElasticNet model: The parameter l1_ratio corresponds to alpha in the glmnet R package,
    # while alpha corresponds to the lambda parameter in glmnet
    my_model= linear_model.ElasticNetCV(l1_ratio=0.5, cv = 5, fit_intercept=False, random_state=0, 
                                        alphas=np.arange(0.1, 0.9, 0.05), selection='random').fit(X_train, Y_train[p])# 
    return my_model.predict(X_train), my_model.predict(X_val)

def comp_y_pred_teamHYU(p): 
# compare to RNA proxy. Random Forest based models (RF) were compared with a baseline method using 5-fold cross validation, in which the RNA level is directly used as a proxy of corresponding protein level. For all proteins RF delivered the best performance, or which do not have cor-
# responding RNA data available, Random Forests were used. The baseline model was used for the remaining proteins
    mask = final_feature[p] 
    if not mask.any(): ## no feature RNA passed threshold pearson > 0.3
        if p in X_train.columns: # Special condition 1: mRNA proxy available, use it!
            return X_train[p].values, X_val[p].values
        else:  ## Special condition 2: no mRNA proxy available, return zeros
            return np.zeros_like(Y_train[p]), np.zeros_like(Y_val[p])
    
    X_val_select, X_train_select = X_val.loc[:, mask ], X_train.loc[:, mask ]

    if (p in both_RNA_protein):  ## protein and mRNA exists, RF vs Proxy, which better?
        N=5
        df_r_temp = pd.DataFrame(0, index= ['RF', 'Proxy'], columns = range(N))
        i = 0
        kf = KFold(n_splits=N, random_state=1, shuffle=True)
        for train_index, test_index in kf.split(Y_train): 
            X_train_inner, X_val_inner = X_train_select.iloc[train_index],  X_train_select.iloc[test_index]
            Y_train_inner, Y_val_inner = Y_train[p].iloc[train_index],  Y_train[p].iloc[test_index]
            my_model = model_dict[m]
            my_model.fit(X_train_inner, Y_train_inner)
            Y_val_inner_pred = my_model.predict(X_val_inner)
            # RF model performance
            df_r_temp.loc['RF', i] =  np.corrcoef(Y_val_inner,  Y_val_inner_pred)[1,0] 
            # proxy model performance
            df_r_temp.loc['Proxy', i] = np.corrcoef(Y_val_inner,  X_train[p][test_index] )[1,0] 
            i = i+1 
        
        if df_r_temp.mean(axis=1)['RF'] < df_r_temp.mean(axis=1)['Proxy']: # here, proxy wins
            return X_train[p].values, X_val[p].values
        
    ## 'other cases, RF wins'
    my_model = model_dict[m]
    my_model.fit(X_train_select, Y_train[p])
    return my_model.predict(X_train_select), my_model.predict(X_val_select)

def sp_parallel(X, Y, proteins): ## comput spearman r block by block, more efficiently than whole X and Y
    df_sp = pd.DataFrame(index= X.columns, columns=proteins)
    for rnas in np.array_split(X.columns, n_set_x):
        df_sp.loc[rnas, proteins] = pd.concat([ X.add_suffix('_x').loc[:,rnas +'_x'], Y.loc[:, proteins]], axis=1)        .corr(method='spearman').loc[rnas+'_x', proteins].values
    return df_sp

run_cross_validation = True
run_test = False
list_model_done = [ ]
print(time.ctime(), 'nCPU =', nCPU, 'run_cross_validation=', run_cross_validation, 'run_test=', run_test)
outfile = '/public/home/test1/mydata/proteome/data/res_v4.3_' +time.strftime("%Y%m%d-%H%M")+\
    '_temp_'+ mykey + '.pkl'
print(outfile)
########### 1. load and transform data 
X = dict_dataset[mykey]["RNA"].transform(lambda x: (x-x.mean())/x.std(), axis=1).round(3).transpose()
Y = dict_dataset[mykey]["protein"].transform(lambda x: (x-x.mean())/x.std(), axis=1).round(3).transpose()
# Y = Y.iloc[:, :30]

# X_use, X_test, Y_use, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123, shuffle=True)

my_list_topn = [X.shape[1]] + list_topn[::-1] 
print(time.ctime(), mykey, 'X',  X.shape, 'Y', Y.shape,  my_list_topn)

df_template = pd.DataFrame(index= Y.columns, columns= range(N))
res = create_res(my_list_topn)

# number of blocks for spearman r computation, 200x100 unit
n_set_x, n_set_y = np.floor_divide(X.shape[1], 200)+1, np.floor_divide(Y.shape[1], 100)+1

########### 2. Pre-comput feature ranking by feature selection methods 1~4
#rank RNAs by linear correlation, run once for the data set, use the matrix later, N-fold CV split i have its own feature ranking matrix
dict_feature_rank = dict.fromkeys(method_feature_select[:-1])
for mfs in dict_feature_rank:
    dict_feature_rank[mfs] = {'CV': dict.fromkeys(range(N)), 'test': dict.fromkeys(range(N))}

dict_peasonr_mat = dict.fromkeys(range(N))

df_template = pd.DataFrame(index= Y.columns, columns= range(N))
res = create_res(my_list_topn)

## first, new machine learning models by me
############## 1. rank all features by a method, run once for the data set, use the maxtrix later
# split i have its own feature rankings
if run_cross_validation == True:
    print(time.ctime(), "prepare training set.")
    # cross-validation on train set; 
    kf = KFold(n_splits=N, random_state=1, shuffle=True)
    i = 0
    for train_index, test_index in kf.split(Y): 
        X_train, X_val = X.iloc[train_index],  X.iloc[test_index]
        Y_train, Y_val = Y.iloc[train_index],  Y.iloc[test_index]
        
        cos = pd.DataFrame(data=metrics.pairwise.cosine_similarity(X=X_train.transpose(), Y=Y_train.transpose(), 
                                                                   dense_output=True), columns=Y.columns)
        
        with Pool(nCPU) as pool:
            spearman_corr = pd.concat( pool.map( partial(sp_parallel, X_train, Y_train), np.array_split(Y.columns, n_set_y) ), axis=1)
        
        np.random.seed(i)
        dict_feature_rank['random']['CV'][i] =             pd.DataFrame([ np.random.permutation(range(1,X.shape[1]+1)) for col in Y.columns], index = Y.columns).T
        dict_feature_rank['cosine']['CV'][i] = abs(cos).rank(axis=0, ascending=False, method='first').astype(np.int32)
        dict_feature_rank['raw_cosine']['CV'][i] = cos.rank(axis=0, ascending=False, method='first').astype(np.int32)
        dict_feature_rank['spearmanr']['CV'][i] = spearman_corr.rank(axis=0, ascending=False, method='first').astype(np.int32)

        dict_peasonr_mat[i] = pd.DataFrame( np.corrcoef(Y_train, X_train, rowvar=False)[Y.shape[1]:, :Y.shape[1]], 
                                       index=X.columns, columns=Y.columns) 
        i = i +1
        

##### 2. cross-validaton model fit and evaluate with test
########### 3. Classical machine learning models using prioritized features
for mfs in method_feature_select[ :-1]:
    for topn in my_list_topn:
        for m in list_model[ :-3]:
            if (topn == X.shape[1] ) & (mfs !='cosine'): ### avoid recomputing all features, 
                res[mfs][topn][m] = deepcopy( res['cosine'][topn][m])
                continue
            print(time.ctime(), mfs.ljust(8,'-'), str(topn).ljust(8,'-'), m.ljust(10,'-'), end=":  ")
            
            if run_cross_validation: # cross-validaton, model fit and evaluate with test
                start =time.time()
                print("training split ", end=" ")
                i = 0
                for train_index, test_index in kf.split(Y): # random trials to stablize errors
                    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
                    Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]
                    
                    df_topn = ( dict_feature_rank[mfs]['CV'][i] <= topn ) ## feature selection

                    if m.startswith('NN'): # faster to use forloop
                        if m == 'NN1':
                            comp_y_pred_nn = comp_y_pred_nn1
                        elif m == 'NN2':
                            comp_y_pred_nn = comp_y_pred_nn2
                        elif m == 'NN3':
                            comp_y_pred_nn = comp_y_pred_nn3
                        else:
                            print(m)
                            raise NameError 
                        y_pred_train_in_rows, y_pred_val_in_rows = list(),list()  
                        for g in Y.columns:
                            x1, x2 = comp_y_pred_nn(g)
                            y_pred_train_in_rows.extend( x1.reshape(1,-1))
                            y_pred_val_in_rows.extend( x2.reshape(1,-1))

                    else:
                        with Pool(nCPU) as pool:
                            y_pred_in_rows = []
                            temp = pool.map_async(comp_y_pred, Y_train.columns,  callback=y_pred_in_rows.extend)
                            temp.wait()
                            y_pred_train_in_rows, y_pred_val_in_rows = zip(*y_pred_in_rows)

                    ## store y_pred
                    res[mfs][topn][m]["y_pred_CVtest"][i] = y_pred_val_in_rows
                    res[mfs][topn][m]["y_pred_CVtrain"][i] = y_pred_train_in_rows
                    
                    i = i + 1;  print(i, end=" ");  # end for split i
                # end:  N fold
                res[mfs][topn]["time"].loc[m, "val"] =  int(time.time() - start)

            list_model_done.extend([ m ])
            with open(outfile, 'wb') as f: 
                pickle.dump( [res],  f)
            print("done.")

############## Next, implement published models
mfs = 'custom'
topn = X.shape[1] ## 'All' features
start = time.time()
print("training split ", end=" ")
i = 0
for train_index, test_index in kf.split(Y): # random trials to stablize errors
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]

    for m in [ 'baselineEN', 'teamHYU', 'teamHL&YG' ]:
        start = time.time()
        print(time.ctime(), mfs.ljust(8,'-'), str(topn).ljust(6,'-'), m.ljust(10,'-'), end="\n")

        if m == 'baselineEN':  ## existing model #1. baseline  ElasticNet regression
            with Pool(nCPU) as pool:      
                y_pred_train_in_rows, y_pred_val_in_rows = zip(*pool.map( comp_y_pred_baselineEN, Y.columns ))

        elif m == 'teamHYU': ### existing model #2. Random Forest regression + PPI selection features
            # now, precompute Pearson's correlation, merge with PPI feature matrix
            ### For teamHYU methods: load feature selection matrix, p1, p2. ~ 60% RNAs, 80% proteins in df_nb PPI neighbor table
            file = '/media/eys/xwj/proteome/proteome_prediction_team_hyu/df_selected_part1-220211126-1529.txt'
            df_nb = pd.read_csv( file, sep='\t', index_col=0 )
            peasonr_mat = dict_peasonr_mat[i].copy() 
            final_feature = (peasonr_mat > 0.3).astype(np.int32)
            common_proteins = df_nb.index & peasonr_mat.columns
            common_RNAs     = df_nb.index & peasonr_mat.index
            ## if PPI have True in (RNA, protein) add weight 1
            final_feature.loc[common_RNAs, common_proteins] += df_nb.loc[common_RNAs, common_proteins].values
            final_feature = final_feature.astype(np.bool)
            both_RNA_protein = X.columns & Y.columns
            print(m, len(both_RNA_protein), 'genes need to compare PPI-feature+RF and proxy') 
            with Pool(nCPU) as pool:
                y_pred_train_in_rows, y_pred_val_in_rows = zip(*pool.map( comp_y_pred_teamHYU, Y.columns ))

        elif m == 'teamHL&YG': ### existing model #3. Random Forest regression + All features
            res[mfs][topn][m] = deepcopy( res['cosine'][topn]['RFR'] )
            list_model_done.extend([ m ])
            continue # copy from RFR cosine, not need to recomput            
        else:
            raise NameError

        ## store y_pred
        res[mfs][topn][m]["y_pred_CVtest"][i] = y_pred_val_in_rows
        res[mfs][topn][m]["y_pred_CVtrain"][i] = y_pred_train_in_rows
        
        res[mfs][topn]["time"].loc[m, "val"] +=  int(time.time() - start) ## 
        with open(outfile, 'wb') as f: 
            pickle.dump( [res],  f)
        list_model_done.extend([ m ])
    i+=1;  print(i, end=" ");  # end for split i    

list_model_done = [m for m in list_model if m in list_model_done]
outfile = '/public/home/test1/mydata/proteome/data/res_v4.3_' +time.strftime("%Y%m%d-%H%M")+    '_'+'-'.join(list_model_done)+'_'+ mykey + '.pkl'
with open(outfile, 'wb') as f: 
    pickle.dump( [res],  f)
print(time.ctime(), "complete", outfile)
