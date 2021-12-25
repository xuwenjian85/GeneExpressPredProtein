# GeneExpressPredProtein
We proposed a new method which use RNA expression profile to predict protein expression level more accurately than previous methods. We comprehensively evaluated the machine learning models on inferring protein expression levels using RNA expression profile using 20 proteogenomic datasets. 
The method and benchmarking results has been submitted for peer review in Nov 2021. 

## contact
If you believe you have found a issue, we would appreciate notification. If you have any questions or suggestions. Please send email to xuwenjian85@qq.com.

## Background & Inspiration
Since the proteomic analysis is more expensive and challenging than transcriptomic analysis, the question of how to use mRNA expression data to predict protein level is central important. Here, a total of 20 proteogenomic datasets from three mainstream proteomic platforms, which consist of > 2500 samples of 13 human tissues, were collected for model evaluation. Our results highlight that the appropriate feature selection methods combined with classical machine learning models could achieve good predictive performance. Specifically, the weighted mean models outperform other candidate models across datasets. Adding proxy model to interaction model further improves the prediction performance. The dataset and gene characteristics would affect the prediction performance. Finally, we applied the model to brain transcriptome of cerebral cortex regions to infer protein profile. This benchmarking work not only provides useful hints on the inherent correlation between transcriptome and proteome, but also shows the practical value of the transcriptome-based protein level predication.

## Installation
The codes are scripts. Just Git clone a copy:
```
https://github.com/xuwenjian85/GeneExpressPredProtein.git
```
## Dependencies
for data processing and machine learning model: 

- Python v3.7
- scipy v1.3
- scikit-learn v0.21

for plotting

- Matplotlib v3.1
- R v4.0.3: 
- ggplot2 v3.3.3
- ggpubr v0.4.0

## Dataset

All the omic data are 2D matrices, where columns are samples and rows are genes/proteins. The preprocessing steps are shared in '/xxxxx'

To run the following code, download the raw data( [upload_raw.tar.gz](https://www.ebi.ac.uk/biostudies/files/S-BSST733/Files/upload_raw.tar.gz) ) and processed data( [20211022_dict_matrix_20dataset.pkl](https://www.ebi.ac.uk/biostudies/files/S-BSST733/Files/20211022_dict_matrix_20dataset.pkl) ) from here(https://www.ebi.ac.uk/biostudies/studies/S-BSST733)


## Quick start

We prepared mini demo dataset consists of only 10 proteins ini Brain tissue. To train all models on this mini dataset, run 

```shell
python demo.py
```

The running time is about xxx hours. The output contain 

## Learning

Use the code 

##  









