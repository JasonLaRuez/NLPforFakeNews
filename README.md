# NLP for Fake News Detection.

We employ a Long Short-Term Memory (LSTM) network trained on a [dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data) containing both fake and true news articles to classify whether a given news item is genuine or fabricated.
 
## Problem Statement

This project aims to assess how well a deep learning approach based on Long Short-Term Memory (LSTM) networks performs relative to a conventional machine learning model, specifically logistic regression trained on TF-IDF features.

## Stakeholders

This project is highly relevant for companies that may face legal exposure for distributing or hosting false information. By integrating such a model into their content-moderation pipeline, organizations could proactively flag potentially misleading material, reducing the risk of legal action and protecting brand credibility.

## KPIs (Key Performance Indicators)
1. **Accuracy**: measures the overall proportion of correct predictions (both true positives and true negatives) among all samples.

2. **Recall**: quantifies how many of the actual positives were correctly identified. We want high recall to minimize the risk of missing real fake-news cases.

3. **Precision**: measures how many of the samples flagged as positive are truly positive. In some applications, we may tolerate lower precision if our priority is to catch as many fake-news cases as possible, even at the cost of some false alarms.

4 **AUC** (Area Under the ROC Curve): evaluates the model’s ability to distinguish between the two classes across all possible thresholds, with higher AUC indicating better overall separability between true and fake news.

# Project Setup

## DataBase

We use the Kaggle database from [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)

## Baseline model

The baseline model uses logistic regression trained on Term Frequency–Inverse Document Frequency (TF-IDF) representations of the news text, which quantify how important each word is within an article relative to the entire dataset.
This approach provides a classical linear model that captures correlations between word occurrence patterns and news authenticity.

## DeepLearning Model

The LSTM model is a deep learning architecture designed to capture long-range dependencies and contextual relationships within sequences of words.
It processes each article as an ordered sequence of tokens, allowing the model to learn patterns in language structure and meaning that simpler linear models cannot capture.

# Results 
We summarize our results in the following table:

| **Metric**       | **Logistic Regression (TF-IDF)** | **LSTM (Deep Learning)** |
|------------------:|:--------------------------------:|:------------------------:|
| **Accuracy**      |                                  |              0.99           |
| **Precision**     |                                  |                          |
| **Recall**        |                                  |                          |
| **ROC AUC**       |                                  |                          |
| **Running Time**  |                                  |                          |







