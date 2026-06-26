# Learning Agency Lab - Automated Essay Scoring 2.0
## EDA
Exploratory data analysis on essay length, word count, sentences, paragraphs, etc. 

## 1. Basic DeBERTa Model
### 1.1 Model
First, a basic DeBERTa model was used from checkpoint `microsoft/deberta-v3-small` with **Mean Pooling** and a **Linear Layer** in a PyTorch head to perform Multiclass Classification. A **Quadratic Weighted Kappa** score was calculated based on Kaggle competition requirements.  
The model can be found in `notebooks/base-deberta-models`. In the folder `classification`, scores were predicted with classification. In the folder `regression`, scores were predicted with regression and were rounded to the nearest score.
### 1.2 Validation
All models were validated using StratifiedKFold, with 5 folds.
### 1.3 Inference
A separate notebook was used to infer on the test set, to save resource time.