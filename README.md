# Learning Agency Lab - Automated Essay Scoring 2.0
## EDA
Exploratory data analysis on essay length, word count, sentences, paragraphs, etc. 

## 1. Model
### 1.1 Basic DeBERTa Model
First, a basic DeBERTa model was used from checkpoint `microsoft/deberta-v3-base` with **Mean Pooling** and a **Linear Layer** in a PyTorch head to perform Multiclass Classification. A **Quadratic Weighted Kappa** score was calculated based on Kaggle competition requirements. 
## 2. Validation
## 3. Inference