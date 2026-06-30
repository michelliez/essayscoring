# Learning Agency Lab - Automated Essay Scoring 2.0
## EDA
Exploratory data analysis on essay length, word count, sentences, paragraphs, etc. 

## 1. Basic DeBERTa Model
### 1.1 Model
First, a basic DeBERTa model was used from checkpoint `microsoft/deberta-v3-small` with **Mean Pooling** and a **Linear Layer** in a PyTorch head to perform Multiclass Classification. A **Quadratic Weighted Kappa** score was calculated based on Kaggle competition requirements.  
The model can be found in `notebooks/base-deberta-models`. In the folder `classification`, scores were predicted with classification. In the folder `regression`, scores were predicted with regression and were rounded to the nearest score.
### 1.2 Validation
All models were validated using StratifiedKFold, with 5 folds. The final best validation QWK score was **0.7922**. The best model was saved as a .pth.    
### 1.3 Inference
A separate notebook was used to infer on the test set, to save resource time. Model weights and configs were loaded from a JSON file and used for inference. This achieved a private score of **0.79530** and a public score of **0.78271**.  
This model was then optimized with thresholding and achieved a private score of **0.80092**.

## 2. DeBERTa Model + Feature Engineering
### 2.1 Model
The same DeBERTa model from checkpoint `microsoft/deberta-v3-small` was used with **Mean Pooling**. However, rather than just a linear layer, the **Torch head** included a linear layer, ReLU activation layer, dropout, and final linear layer. This was created as a regression and rounded to the nearest essay score. 

### 2.2 Validation
Models were validated using StratifiedKFold, with 5 folds. The final best QWK score was **0.7889**. The best model per fold was saved as a .pth.    

### 2.3 Inference
A seperate notebook was used to infer on the test set. Feature engineering was also performed on the test set, and model weights and configs were loaded from a JSON file. With thresholding, this achieved a score of **0.78484**.


## 3. DeBERTa Model Seed Ensemble + Thresholding
### 3.1 Model
The same DeBERTa model from checkpoint `microsoft/deberta-v3-small` from **Section 1.1** with just a linear head was trained twice with different random seeds: seed 42 and seed 23. Both trained models were saved. 

### 3.2 Validation
During validation, raw OOF predictions from the two models were averaged to obtain an ensemble prediction. These ensemble OOF predictions were then used to tune score thresholds for QWK.

### 3.3 Inference
A seperate inference notebook was used to generate predictions on the hidden test set. This achieved a private leaderboard score of **0.80169**.


## 4. DeBERTa + MLP Head + LightGBM Stacking
### 4.1 Feature Engineering
Extra features were created from **EDA** analysis. These features are:
- Word count
- Number of sentences
- Sentence length in words
- Number of paragraphs
- Paragraph length in words
- Whether the essay was over 2,500 words, because no essay over 2,000 words scored a 5 or above.
### 4.2 Model
The five fold-specific DeBERTa models with an **MLP** Torch prediction head from **Section 2** were used to generate OOF predictions for the training set. These OOF predictions were then added to the train_df DataFrame as an additional feature alongside the ones listed from **Section 4.1**. A **LightGBM Regressor** was then trained on these features for predictions.
### 4.3 Validation
The LightGBM model was trained using the same five-fold stratified CV splits through StratifiedKFold. For each fold, one fold was kept as the validation fold to prevent leakage. These OOF predictions were also kept for thresholding. 

### 4.4 Inference
During inference, predictions from the five DeBERTa models from **Section 2** were averaged to obtain the Transformer predictions for each essay. This was combined with the engineered features in test_df DataFrame. Five LightGBM models were trained on these features, one per fold. The predictions were averaged (ensmbled) to produce the final regression score, which was rounded based on optimized thresholds. Thresholding achieved a private leaderboard score of **0.825**.

## 5. DeBERTa + MLP Head + LightGBM & XGBoost Ensemble Stacking
### 5.1 Feature Engineering
Extra features were created from **EDA** analysis. These features are:
- Word count
- Number of sentences
- Sentence length in words
- Number of paragraphs
- Paragraph length in words
- Whether the essay was over 2,500 words, because no essay over 2,000 words scored a 5 or above.
- Unique word ratio
### 5.2 Model
The five fold-specific DeBERTa models with an **MLP** Torch prediction head from **Section 2** were used to generate OOF predictions for the training set. These OOF predictions were then added to the train_df DataFrame as an additional feature alongside the ones listed from **Section 4.1**.   
A **LightGBM Regressor** and **XGBoost Regressor** was then trained on these features for predictions. These two model's predictions were averaged together.
### 4.3 Validation
The LightGBM and XGBoost models were trained using the same five-fold stratified CV splits through StratifiedKFold. For each fold, one fold was kept as the validation fold to prevent leakage. These LightGBM and XGBoost OOF predictions were averaged and also kept for thresholding. 

### 4.4 Inference
During inference, predictions from the five DeBERTa models from **Section 2** were averaged to obtain the Transformer predictions for each essay. This was combined with the engineered features in test_df DataFrame. Five LightGBM models and five XGBoost models were trained on these features, one per fold. The predictions for each model were averaged (ensembled), and then these two averages were ensembled together to produce the final regression score, which was rounded based on optimized thresholds. Thresholding achieved a private leaderboard score of **0.82580**.


## 5. Challenges
Was unable to use DeBERTa v3 base or large on my system. Not enough GPU for the tensors to be 32 bit. Had to use DeBERTa small to prevent collapse.