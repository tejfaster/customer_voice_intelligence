Customer Voice Intelligence Platform (CVIP)

An end-to-end Natural Language Processing (NLP) system designed to transform large-scale unstructured customer reviews into structured sentiment intelligence using classical machine learning techniques.

1. Project Overview

This project processes ~68,000 Amazon product reviews and builds a binary sentiment classification system (Positive vs Negative) using a fully structured ML pipeline.

The system includes:

Text preprocessing using spaCy

TF-IDF feature extraction

Class-weighted Logistic Regression

Stratified train/test splitting

Minority-class optimization through threshold tuning

Performance evaluation under severe class imbalance

2. Problem Statement

Customer reviews contain valuable signals regarding satisfaction and dissatisfaction.

However:

Reviews are unstructured text

Dataset is highly imbalanced (~96% Positive, ~4% Negative)

Accuracy alone is misleading

The challenge is to build a robust classifier capable of reliably detecting minority-class (Negative) reviews while handling imbalance properly.

3. Modeling Approach
Data Preprocessing

Lowercasing

URL removal

Stopword filtering

Lemmatization (spaCy)

Batch-optimized pipeline using nlp.pipe()

Feature Engineering

TF-IDF Vectorization

Unigrams and Bigrams

10,000 max features

Classification Model

Logistic Regression

class_weight="balanced"

Stratified 80/20 train-test split

4. Handling Class Imbalance

Given severe imbalance:

Used stratified sampling

Applied class-weighted loss function

Evaluated using Precision, Recall, and F1-score (not accuracy)

Tuned decision threshold (0.5 → 0.8)

Performance Improvement (Negative Class)
Threshold	Precision	Recall	F1-score
0.5	0.41	0.87	0.56
0.8	0.71	0.73	0.72

Threshold tuning significantly improved minority-class balance.

5. Evaluation Metrics

Primary focus was placed on minority-class performance:

Precision (Negative)

Recall (Negative)

F1-score (Negative)

Confusion Matrix

Macro-average F1

Accuracy was reported but not used for optimization due to class imbalance.

6. Project Structure
customer_voice_intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   └── cvi/
│       ├── preprocessing/
│       ├── models/
│       └── api/  (future expansion)
│
├── experiments/
│   └── notebooks/
│
├── models/
├── requirements.txt
└── README.md

7. Key Learnings

Impact of severe class imbalance in NLP

Importance of stratified splitting

Precision–Recall tradeoff analysis

Threshold calibration for business-driven optimization

Why accuracy is misleading in skewed datasets

Production-oriented modular project structuring

8. Future Improvements

Transformer-based model comparison (e.g., DistilBERT)

Cost-sensitive optimization

REST API deployment (FastAPI)

Real-time inference pipeline

Automated retraining pipeline

Author

Tej Faster
Data Analytics & Machine Learning Enthusiast
Focused on NLP systems and applied machine learning