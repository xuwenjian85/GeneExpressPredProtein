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

## Not satisfied? Add new models 

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

