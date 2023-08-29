# Quora Question Pairs

This repository contains code for the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) Kaggle competition. 

## Problem Statement

The goal of this competition is to identify which questions asked on Quora are duplicates of questions that have already been asked. This could be useful to instantly provide answers to questions that have already been answered.

We are tasked with predicting whether a pair of questions are duplicates or not.

## Data

The dataset consists of over 400,000 lines of training data with the following features:

- `qid1`, `qid2`: Unique IDs of each question
- `question1`, `question2`: The actual text of the questions 
- `is_duplicate`: The label to predict - 1 if duplicate question, 0 if not

## Approach

The following steps were taken to build a model for this problem:

1. Exploratory data analysis
   - Distribution of duplicate vs. non-duplicate pairs
   - Analysis of question length, word counts, etc
2. Data cleaning
   - Fill NA values
   - Remove HTML tags, punctuation, etc
   - Stemming and removing stopwords 
3. Feature engineering
   - Question length, word counts, common words, etc
   - NLP features like fuzz ratios, token sort ratios, etc
4. Visualization with t-SNE
5. Modeling
   - Train classification models like Logistic Regression, SVM, XGBoost, etc
   - Evaluate models with AUC ROC score
   - Hyperparameter tuning
   - Ensembles and blending

## Files

- `EDA.ipynb`: Notebook containing exploratory data analysis
- `data_preprocessing.py`: Functions for cleaning and preprocessing text data
- `feature_engineering.py`: Functions to extract features from text pairs 
- `modeling.py`: Model training, evaluation, and tuning code
- `model_blend.py`: Stacking and blending models for ensembling
- `README.md`: This file

The other files contain data, intermediate outputs, etc.

## Results

The best single model AUC ROC score obtained was 0.853 with XGBoost. 

Blending XGBoost, LightGBM, and Logistic Regression models using stratified k-fold splits gave an improved score of 0.862.

There is scope to further improve the score by tuning and experimenting with more advanced NLP techniques.
