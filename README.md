# GeneExpressPredProtein
We proposed a new method which use RNA expression profile to predict protein expression level more accurately than previous methods. We comprehensively evaluated the machine learning models on inferring protein expression levels using RNA expression profile using 20 proteogenomic datasets. 
The method and benchmarking results has been submitted for peer review in Nov 2021. 

## contact
If you believe you have found a issue, we would appreciate notification. If you have any questions or suggestions. Please send email to xuwenjian85@qq.com.

## Background & Inspiration
Since the proteomic analysis is more expensive and challenging than transcriptomic analysis, the question of how to use mRNA expression data to predict protein level is central important. Here, a total of 20 proteogenomic datasets from three mainstream proteomic platforms, which consist of > 2500 samples of 13 human tissues, were collected for model evaluation. Our results highlight that the appropriate feature selection methods combined with classical machine learning models could achieve good predictive performance. Specifically, the weighted mean models outperform other candidate models across datasets. Adding proxy model to interaction model further improves the prediction performance. The dataset and gene characteristics would affect the prediction performance. Finally, we applied the model to brain transcriptome of cerebral cortex regions to infer protein profile. This benchmarking work not only provides useful hints on the inherent correlation between transcriptome and proteome, but also shows the practical value of the transcriptome-based protein level predication.

## Installation
The codes are jupyter scripts. Just set up a jupyter notebook server and git clone a copy:
```
https://github.com/xuwenjian85/GeneExpressPredProtein.git
```
## Dependencies
for data processing and machine learning model: 

- Python v3.7
- scipy v1.3
- scikit-learn v0.21
- torch v1.10.0 
- skorch v0.11.0

for plotting
- python(Matplotlib v3.1)

## Quick start

We prepared mini demo dataset consists of only 100 proteins and 200 RNAs in liver  tissue. To train all models on this mini dataset, run 

```shell
RNA-protein_predict_demo.ipynb
```

The running time is about 15 mins. The pipeline out is proteins levels predicted.

## Datasets used in our benchmarking paper

All the omic data are 2D matrices, where columns are samples and rows are genes/proteins. The preprocessing steps are shared in '**RNA-Protein_data_preprocess_explained.ipynb**'

To run the following complete analysis, download the raw data( [upload_raw.tar.gz](https://www.ebi.ac.uk/biostudies/files/S-BSST733/Files/upload_raw.tar.gz) ) or the processed data( [20211022_dict_matrix_20dataset.pkl](https://www.ebi.ac.uk/biostudies/files/S-BSST733/Files/20211022_dict_matrix_20dataset.pkl) ) from here(https://www.ebi.ac.uk/biostudies/studies/S-BSST733)

**Name of datasets**

|    long  | short      |
| ---- | ---- |
|colon_86_2014_labelfree|CO_labelfree|
|brain_71_2017_labelfree|BN_labelfree|
|prostate_65_2019_labelfree |PR_labelfree|
|liver_62_2019_labelfree|LV_labelfree|
|lung_76_2020_labelfree|LU_labelfree|
|liver_318_2019_tmt|LV_tmt|
|colon_95_2019_tmt       |   CO_tmt|
|renal_185_2019_tmt        | RC_tmt|
|pedbrain_188_2020_tmt|PBN_tmt|
|breast_122_2020_tmt    |     BR_tmt|
|uterus_115_2020_tmt   |      EC_tmt|
|lung_211_2020_tmt       |  LU_1_tmt|
|lung_89_2020_tmt   |      LU_2_tmt|
|lung_202_2021_tmt      |   LU_3_tmt|
|headneck_151_2021_tmt   |     HN_tmt|
|pancrea_140_2021_tmt    |    PA_tmt|
|brain_108_2021_tmt        | BN_tmt|
|breast_77_2016_itraq     |  BR_itraq|
|ovary_119_2016_itraq      | OV_itraq|
|stomach_80_2019_itraq|GC_itraq|

## Complete guide to repeat our work

### Model fitting

```shell
python RNA-Protein_predict_v4.3_explained.py i
```

Model fitting on dataset *i (i=0~19)*. The step take several days and will output predicted protein values. Predicted result will be saved as a python dictionary named ‘res’. The output file size are up to 25GB. (**Execution time**: The hardware of our computation nodes is AMD EPYC 7742 64-Core CPU 2.25GHz, 1Tb RAM. The whole analysis on 20 benchmarking datasets take 186 days(d) if user run our codes in sequentially. For example, the largest dataset ‘LV_tmt’ (318 samples) take 19d by itself. We recommend user to run each dataset in parallel.)

After the output file is ready, editing a new configure file describing the path of ‘res’ files. Formatted like  res_pkl20211220.list.  

### Performance evaluation

```shell
python RNA-Protein_predict_v4.3_performance_metric_explained.py i
```

According to configure file from **model fitting**, compute performance metrics Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and Pearson’s correlation coefficient (PCC, r) on dataset *i (i=0~19)* . Result will be saved back to the python dictionary named ‘res’ to hard drive.

Editing a new configure file describing the path of updated ‘res’ files. Formatted like res_pkl_metric20211221.list

### Final Analysis

The analysis steps for **Figures and Tables** are shared in **RNA_result_analysis-v3.ipynb** and **RNA-protein_plot_v2-explained.ipynb**. The model named "Voting" is the best model proposed in our paper. Existing methods are "baselineEN", "teamHYU" and "teamHL&YG". 

### Case study on brain atlas transcriptome

Analysis of brain atlas data is done with **RNA-Protein_predict_v3-Brain.ipynb**.  Here we applied the model to the brain
transcriptome of cerebral cortex regions to infer the protein profile for better understanding the functional
characteristics of the brain regions. 

## Not satisfied? feel free to add new models 

The module are highly customizable. Just add a new key: value to develop new models!

```python
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
    "teamHL&YG": None, 
    }
```