#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[2]:

gene_tpm = pd.read_table("GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct",skiprows=2)
gene_tpm.count()[0:5]

# In[11]:

samples = pd.read_table("GTEx_v7_Annotations_SampleAttributesDS-blood_lung-ONLY.txt")
samples[1:5]

# In[16]:

samples[samples.SMTS == "Lung"].SAMPID.values[1:5]

# In[14]:

lung_gene_tpm = gene_tpm[["Name", "Description"] + list(samples[samples.SMTS == "Lung"].SAMPID.values)]
blood_gene_tpm = gene_tpm[["Name", "Description"] + list(samples[samples.SMTS == "Blood"].SAMPID.values)]
# 首先检查是否所有gene都在各个样本中有表达？剔除在血液样本（或者肺部样本)中不表达的基因, 剩余约10000多个gene有表达
'''
blood_gene_tpm.all(axis="columns").value_counts()
'''

lung_gene_tpm = lung_gene_tpm.loc[lung_gene_tpm.all(axis="columns")]
blood_gene_tpm = blood_gene_tpm.loc[blood_gene_tpm.all(axis="columns")]

# In[52]:

blood_gene_tpm.to_csv("blood_gene_tpm.txt",sep= '\t', index=False)
lung_gene_tpm.to_csv("lung_gene_tpm.txt",sep= '\t', index=False)

# In[2]:

blood_gene_tpm = pd.read_table("blood_gene_tpm.txt",index_col = 0)
lung_gene_tpm = pd.read_table("lung_gene_tpm.txt",index_col = 0)

# In[5]: 
# 有哪些基因有多个转录本？ 
lung_gene_tpm['Description'].value_counts().sort_values(ascending=False)[1:5]
blood_gene_tpm['Description'].value_counts().sort_values(ascending=False)[1:5]

# In[9]:

lung_gene_tpm.iloc[0:2,1:]

# In[6]:
lung_gene_tpm.shape, blood_gene_tpm.shape


# In[8]:
# 转置后：样本为行，基因为列 lung and blood transpose
# also equal to 
step = 1000
i,j = 0,0

lung_gene_tpm_T = pd.DataFrame(index=lung_gene_tpm.columns[1:]) # empty df
blood_gene_tpm_T = pd.DataFrame(index=blood_gene_tpm.columns[1:]) # empty df

gene_count = lung_gene_tpm.shape[0]
while i < gene_count:
    j = i + step
    if j > gene_count: j = gene_count
    batch = lung_gene_tpm.iloc[i:j,1:].transpose()
    lung_gene_tpm_T = pd.concat([lung_gene_tpm_T, batch], axis= 1)
    i=j

step = 1000
i,j = 0,0

gene_count = blood_gene_tpm.shape[0]
while i < gene_count:
    j = i + step
    if j > gene_count: j = gene_count    
    batch = blood_gene_tpm.iloc[i:j,1:].transpose()
    blood_gene_tpm_T = pd.concat([blood_gene_tpm_T, batch], axis= 1)
    i=j

lung_gene_tpm_T.shape, blood_gene_tpm_T.shape

# In[11]:
# blood_gene_tpm_T.describe() #don't run , will take 5 minutes 
'''
*_T重新规整化为DataFrame
'''

blood_gene_tpm_T = pd.DataFrame(blood_gene_tpm_T) 
lung_gene_tpm_T = pd.DataFrame(lung_gene_tpm_T)

# In[9]:
# 基因是在所有血液 或 lung样本中都有表达呢？ 答：Yes
# np.corrcoef(blood_gene_tpm_T[1],blood_gene_tpm_T[1])
blood_gene_tpm_T.all(axis="columns").value_counts()
lung_gene_tpm_T.all(axis="columns").value_counts()

# In[241]:
'''
比较血液中两个基因的表达趋势是否相关，以坐标或列名任选两列,简单测试
'''
blood_gene_tpm_T['ENSG00000227232.4']

# In[10]:
plt.scatter(blood_gene_tpm_T.iloc[:,0],blood_gene_tpm_T.iloc[:,2])

# In[282]:
np.corrcoef(blood_gene_tpm_T.iloc[:,1],lung_gene_tpm_T.iloc[:,1])[0,1]
# np.corrcoef(blood_gene_tpm.iloc[1,2:].astype(float),lung_gene_tpm.iloc[1,2:].astype(float)) 
# blood_gene_tpm_T[1].corr(blood_gene_tpm_T[1])

# In[11]:
plt.scatter(blood_gene_tpm_T.iloc[:,1], lung_gene_tpm_T.iloc[:,1])

# In[13]
''' 找出与lung gene最相关一组blood基因，最相关组大小为top_corr_gene_count'''
top_corr_gene_count = 10

''' 初始化'''
lung_blood_corr = pd.DataFrame(data=0,
                               index=range(0,top_corr_gene_count), 
                               columns=lung_gene_tpm_T.columns)
# np.zeros(shape=(gene_count,gene_count), dtype=np.int) # TOO large -> MemoryError
# lung_blood_corr = np.zeros(shape=(top_corr_gene_count,gene_count), dtype=np.str) 


# In[30]:
'''写个loop: coorelation coefficient between blood基因 and lung基因
（不需要一定是同一个基因）选择出top10基因
声明一个临时变量corr_values，用来保存一个lung gene与所有blood gene的相关性；
按相关性排序，保留top_corr_gene_count个blood gene的名称'''
for lung_gene in lung_gene_tpm_T.columns : 
    # make a temp vector
    # corr_values = np.zeros(shape=(gene_count), dtype=np.int)
    # corr_values = pd.Series(data=np.random.normal(size=gene_count), 
    #                         index= blood_gene_tpm_T.columns,dtype=np.float)
    corr_values = pd.Series(data=0, 
                            index= blood_gene_tpm_T.columns,
                            dtype=np.float)
    
    for blood_gene in blood_gene_tpm_T.columns:
        # lung_blood_corr['ENSG00000223972.4'] = corr_values.abs().sort_values().tail(10).index
        corr_values[blood_gene] = np.corrcoef(blood_gene_tpm_T[blood_gene],
                                              lung_gene_tpm_T[lung_gene])[0,1]
    lung_blood_corr[lung_gene] = corr_values.abs().sort_values().tail(top_corr_gene_count).index

#%%
lung_blood_corr.iloc[:,0:10]

# In[288]:
# python里的corr函数也能算corelation coefficent，不方便,作为备用
temp = lung_gene_tpm.iloc[1:20,2:].transpose().corr().abs()
np.fill_diagonal(temp.values, 0)
temp.stack().hist(bins=100)
temp.stack().describe(percentiles = [0.2,0.4,0.5,0.6,0.8,0.9])
# temp = np.corrcoef(blood_gene_tpm_T.iloc[:,1:5],rowvar=False)

# In[253]:


# In[254]:


# In[176]:




