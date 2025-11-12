# NLP for Fake News Detection.

We employ a Long Short-Term Memory (LSTM) network trained on two datasets, the [Politifact](https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset/data) fact-check dataset and the [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data) fake news dataset. The Politifact dataset contains texts from news articles, social media, speeches, etc. which have been labelled as varying degrees of true or false: pants-fire, false, mostly-false, half-true, mostly-true and true. The WELFake dataset is a larger corpus of text containing both fake and true news articles to classify whether a given news item is genuine or fabricated.
 
## Problem Statement

This project aims to assess how well a deep learning approach based on Long Short-Term Memory (LSTM) networks performs relative to a conventional machine learning model, specifically logistic regression trained on TF-IDF features.

## Stakeholders

This project is highly relevant for companies that may face legal exposure for distributing or hosting false information. By integrating such a model into their content-moderation pipeline, organizations could proactively flag potentially misleading material, reducing the risk of legal action and protecting brand credibility.

## KPIs (Key Performance Indicators)
1. **Accuracy**: measures the overall proportion of correct predictions (both true positives and true negatives) among all samples.

2. **Recall**: quantifies how many of the actual positives were correctly identified. We want high recall to minimize the risk of missing real fake-news cases.

3. **Precision**: measures how many of the samples flagged as positive are truly positive. In some applications, we may tolerate lower precision if our priority is to catch as many fake-news cases as possible, even at the cost of some false alarms.

4. **AUC** (Area Under the ROC Curve): evaluates the model’s ability to distinguish between the two classes across all possible thresholds, with higher AUC indicating better overall separability between true and fake news.

# Project Setup

## DataBase

We use the Kaggle databases from [https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset/data]( https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset/data) and [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)

## Baseline model: Multi-class Logistic Regression

The baseline model uses logistic regression trained on Term Frequency–Inverse Document Frequency (TF-IDF) representations of the news text, which quantify how important each word is within an article relative to the entire dataset. We do not include sentiment-based features in this model, since ["A benchmark study of machine learning models for online fake news detection"](https://www.sciencedirect.com/science/article/pii/S266682702100013X) found that sentiment-based features are not useful in fake news detection. This approach provides a classical linear model that captures correlations between word occurrence patterns and news authenticity.

To compute TF-IDF, we first compute the term-frequency $tf(t,d)$, which is the number of times term $t$ occurs in document $d$. Next, we compute the inverse-document-frequency $$idf(t)=\log{\frac{1+n_d}{1+df(t)}}$$
where $n_d$ is the total number of documents in the corpus, and $df(t)$ is the document frequency of the term $t$ (the number of documents that contain the term $t$). If a word occurs in every document, then the $idf$ of that word is 0. Thus, words that occur in most documents (such as "the", "as", "it") are given very little weight. Finally, the term frequency-inverse document frequency (TF-IDF) is given by:

$$tfidf(t,d) = tf(t,d)\cdot (1+idf(t))$$

## DeepLearning Model: AWD-LSTM

The LSTM model is a deep learning architecture designed to capture long-range dependencies and contextual relationships within sequences of words.It processes each article as an ordered sequence of tokens, allowing the model to learn patterns in language structure and meaning that simpler linear models cannot capture. AWD-LSTM (ASGD Weight-Dropped LSTM) is a regularized variant of the Long Short-Term Memory network designed for efficient language modeling. It introduces weight-dropping (dropout on hidden-to-hidden weights), variational dropout, and NT-ASGD (averaged SGD) optimization to improve generalization. This architecture achieves strong performance on text tasks by combining stability, regularization, and efficient training dynamics

# Results 
We summarize our results in the following tables. These first set of tables correspond to the Politifact database.

6-class (Politifact)

| **Metric**        | **LR (TF-IDF)** | **LSTM-forward** | **LSTM-backward** | **LSTM-combined** |
|------------------:|:---------------:|:----------------:|:-----------------:|:-----------------:|
| **Accuracy**      | 0.302           | 0.292            | 0.292             | 0.300             |
| **Precision**     | 0.295           | 0.297            | 0.295             | 0.303             |
| **Recall**        | 0.289           | 0.302            | 0.292             | 0.305             |
| **ROC AUC**       | 0.667           | 0.682            | 0.676             | 0.692             |
| **Running Time**  | 110.65s         | 179.44s          | 244.17s           | 423.61s            |

3-class (Politifact)

| **Metric**        | **LR (TF-IDF)** | **LSTM-forward** | **LSTM-backward** | **LSTM-combined** |
|------------------:|:---------------:|:----------------:|:-----------------:|:-----------------:|
| **Accuracy**      | 0.555           | 0.518            | 0.524             | 0.548             |
| **Precision**     | 0.556           | 0.538            | 0.538             | 0.562             |
| **Recall**        | 0.558           | 0.522            | 0.535             | 0.555             |
| **ROC AUC**       | 0.726           | 0.713            | 0.714             | 0.724             |
| **Running Time**  | 23.05s          | 174.18s          | 251.51s           | 425.69s           |

2-class (Politifact)

| **Metric**        | **LR (TF-IDF)** | **LSTM-forward** | **LSTM-backward** | **LSTM-combined** |
|------------------:|:---------------:|:----------------:|:-----------------:|:-----------------:|
| **Accuracy**      | 0.705           | 0.702            | 0.688             | 0.706             |
| **Precision**     | 0.697           | 0.648            | 0.612             | 0.636             |
| **Recall**        | 0.753           | 0.778            | 0.779             | 0.793             |
| **ROC AUC**       | 0.769           | 0.774            | 0.765             | 0.778             |
| **Running Time**  | 11.76s          | 171.10s          | 250.27s           | 421.37s           |


The next table correspond tot he WELFake database which is just a 2-label classification problem.

| **Metric**       | **Logistic Regression (TF-IDF)** | **LSTM (Deep Learning)** |
|------------------:|:--------------------------------:|:------------------------:|
| **Accuracy**      |                 0.972            |           0.994          |
| **Precision**     |                 0.970            |           0.992          |
| **Recall**        |                 0.970            |           0.993          |
| **ROC AUC**       |                 0.996            |           0.998          |
| **Running Time**  |                 13.58 mins       |           78.77  mins   |


# Conclucions
For the [Politifact](https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset/data) dataset, we obtained comparable values of accuracy, recall, precision, and ROC–AUC for both models — logistic regression and LSTM.  

In contrast, for the [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data) dataset, the LSTM model achieved slightly higher accuracy, recall, precision, and ROC–AUC than the logistic regression model. Overall, the WELFake dataset yields better performance metrics, likely due to its larger number of samples and longer text content.  

We therefore expect that for even larger datasets, the LSTM model will continue to outperform logistic regression. This suggests that training LSTM-based models can provide a stronger defense against misinformation, making them particularly valuable for organizations focused on detecting fake news.

# Overview of Repository Layout

The Politifact.ipynb notebook contains the code used to analyze the Politifact dataset and to build both the TF IDF model and the AWD LSTM classifier. The WELFake.ipynb notebook performs a similar analysis for the WELFake dataset. The EDA.ipynb notebook provides exploratory data analysis for both datasets, including statistics such as the most common words per label, typical article lengths, frequent bigrams and trigrams, and for the Politifact dataset, additional features like the most common news sources and quoted speakers. The imgs folder stores the figures of the confusion matrices, and the processed_datasets folder contains the cleaned datasets produced by the PreProcessingDataset module, which is used to preprocess the WELFake data.




# Team Members
This project was developed for 2025 Fall Erdös Institute Deep Learning Boot Camp by:

-Victoria Knapp Perez -Jason LaRuez


