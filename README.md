# Customer Voice Intelligence Platform (CVIP)

An AI-driven system that transforms large-scale unstructured customer
reviews into structured complaint intelligence, recurring issue
detection, and automated product risk monitoring signals.

------------------------------------------------------------------------

## 1. Project Overview

The Customer Voice Intelligence Platform (CVIP) is an end-to-end NLP
system designed to:

-   Detect dissatisfied customers
-   Discover recurring complaint themes
-   Quantify business impact of complaint categories
-   Monitor issue trends over time
-   Automatically flag potential product risk spikes

This project goes beyond traditional sentiment analysis and builds a
structured Product Quality Intelligence System.

------------------------------------------------------------------------

## 2. Business Problem

Organizations receive thousands of product reviews containing valuable
but unstructured feedback.

Challenges: - Reviews are unstructured text - Negative reviews are rare
(\~4%) but high-impact - Root causes are hidden inside noisy language -
Manual analysis does not scale - Issues are often detected too late

Goal: Convert raw customer voice into actionable product risk
intelligence.

------------------------------------------------------------------------

## 3. Dataset

-   \~65,000 total product reviews
-   \~2,510 negative reviews (≈4%)
-   Time span: 2009 -- 2018 (55 unique months)

------------------------------------------------------------------------

## 4. System Architecture

### Phase 1 --- Sentiment Detection

-   Binary classification (Positive vs Negative)
-   TF-IDF vectorization (10,000 features)
-   Logistic Regression with class imbalance handling
-   Threshold tuning for minority recall optimization

Purpose: Filter dissatisfied customers reliably.

------------------------------------------------------------------------

### Phase 2 --- Semantic Embedding Layer

-   SentenceTransformer (`all-MiniLM-L6-v2`)
-   384-dimensional dense semantic vectors
-   Cosine similarity reasoning
-   Vector-based semantic retrieval

Purpose: Move from word-frequency to meaning-based understanding.

------------------------------------------------------------------------

### Phase 3 --- Root Cause Discovery

-   Density-based clustering (HDBSCAN)
-   Clustering applied only to negative reviews
-   Automatic detection of recurring complaint themes

Result:

-   Major complaint cluster identified
-   672 reviews (26.77% of negative reviews)
-   Theme: Battery & Power Reliability Issues

Common issues discovered: - Rapid battery drain - Charging failures -
Premature device malfunction - High return tendency

This transforms:

Sentiment → Root Cause Intelligence

------------------------------------------------------------------------

### Phase 4 --- Temporal Trend Monitoring

-   Monthly aggregation (55 months)
-   Cluster share tracking over time
-   Denominator-aware validation (volume filtering)
-   Stable complaint band identified (20--35%)

Insight:

Battery-related complaints consistently represented approximately
20--35% of negative reviews during peak review years (2015--2017).

------------------------------------------------------------------------

### Phase 5 --- Automated Risk Alert Engine

Implemented:

-   Month-over-month complaint share delta
-   Statistical threshold detection (\>10% increase)
-   Minimum volume constraint (≥30 negative reviews per month)

Valid Risk Alert Months Identified:

-   2016-04
-   2016-10
-   2017-07

The system can automatically flag rising product quality risk signals.

------------------------------------------------------------------------

## 5. Key Results

-   26.77% of all negative reviews relate to battery reliability issues
-   Complaint share stable between 20--35% during high-volume years
-   Automated anomaly detection identifies meaningful spikes
-   Low-volume statistical noise successfully filtered

------------------------------------------------------------------------

## 6. Technologies Used

-   Python
-   Pandas
-   Scikit-learn
-   Sentence-Transformers
-   HDBSCAN
-   Matplotlib

------------------------------------------------------------------------

## 7. What Makes This Project Different

Most NLP projects: - Stop at sentiment classification.

This project: - Detects root causes. - Quantifies complaint
categories. - Adds time-series monitoring. - Implements automated risk
detection logic. - Integrates statistical safeguards.

It is a structured customer intelligence system.

------------------------------------------------------------------------

## 8. Business Impact

The system enables:

-   Early product defect detection
-   Complaint spike monitoring
-   Data-driven product improvement
-   Reduced investigation time
-   Proactive quality risk management

It converts:

Unstructured reviews → Structured product risk signals.

------------------------------------------------------------------------

## 9. Future Enhancements

-   Automated cluster labeling using LLM
-   Real-time ingestion pipeline
-   Dashboard integration (Streamlit / BI tool)
-   Cross-metric correlation (returns + sales + complaints)
-   Risk scoring model

------------------------------------------------------------------------

## Author

Tej Pratap\
AI & Data Intelligence Enthusiast\
Focused on building applied AI systems that bridge machine learning and
business decision-making.