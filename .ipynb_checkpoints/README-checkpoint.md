# Customer Segmentation Analysis using RFM Approach

## Overview
This project focuses on analyzing customer behavior and segmenting customers using the RFM (Recency, Frequency, Monetary) approach. By analyzing various customer attributes and transaction data, we aim to help businesses optimize their marketing strategies, improve customer service, and reduce customer churn.

## Problem Statement
Businesses need to understand their customers better to:
- Optimize marketing campaigns
- Improve customer retention
- Increase profitability
- Enhance customer service
- Prevent customer churn

## Dataset

The dataset contains more than 300k+ comprehensive customer information including:
- Customer demographics (ID, name, email, phone, address, age, gender, income)
- Transaction details (purchase dates, amounts, products)
- Product information (category, brand, type)
- Order information (shipping method, payment method, status)
- Customer feedback

## Project Structure
```
├── data/
│   └── customer_data.csv
├── notebooks/
│   └── customer_segmentation_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   └── rfm_analysis.py
├── README.md
└── requirements.txt
```

## Methodology

### 1. Data Exploration and Analysis
- Demographic analysis
- Transaction patterns
- Product preferences
- Payment and shipping analysis
- Temporal analysis
- Customer feedback analysis

### 2. Data Preprocessing
1. **Handling Missing Values**
   - Check for null values in each column
   - Apply appropriate imputation methods
   - Remove or fill missing values based on business context

2. **Feature Engineering**
   - Calculate RFM metrics
     - Recency: Days since last purchase
     - Frequency: Total number of purchases
     - Monetary: Total amount spent

3. **Data Transformation**
   - Label encoding for categorical variables
   - One-hot encoding for nominal variables
   - Feature scaling using StandardScaler
   - Normalization of RFM values to 0-1 range

4. **Feature Selection**
   - Remove redundant features
   - Select relevant features for clustering
   - Focus on RFM metrics for final segmentation

### 3. K-means Implementation

1. **Algorithm Setup**
   ```python
   from sklearn.cluster import KMeans
   from sklearn.preprocessing import StandardScaler
   ```

2. **Data Preparation**
   ```python
   # Scale the RFM features
   scaler = StandardScaler()
   rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
   ```

3. **Optimal Cluster Selection**
   - Elbow Method Implementation
   ```python
   inertias = []
   for k in range(1, 11):
       kmeans = KMeans(n_clusters=k, random_state=42)
       kmeans.fit(rfm_scaled)
       inertias.append(kmeans.inertia_)
   ```
   - Silhouette Score Analysis
   ```python
   from sklearn.metrics import silhouette_score
   silhouette_scores = []
   for k in range(2, 11):
       kmeans = KMeans(n_clusters=k, random_state=42)
       cluster_labels = kmeans.fit_predict(rfm_scaled)
       silhouette_avg = silhouette_score(rfm_scaled, cluster_labels)
       silhouette_scores.append(silhouette_avg)
   ```

4. **Final Model Training**
   ```python
   # Train K-means with optimal clusters
   final_kmeans = KMeans(n_clusters=3, random_state=42)
   clusters = final_kmeans.fit_predict(rfm_scaled)
   ```

5. **Customer Segmentation Results**
   - Cluster 1 (Gold): High frequency, high monetary value, low recency
   - Cluster 2 (Silver): Moderate values across all metrics
   - Cluster 0 (Bronze): Low frequency, low monetary value, high recency

## Key Findings

### Demographic Insights
- Majority of transactions from US
- Higher male customer representation
- Medium income customers form the largest segment
- Younger customer base (majority below 30)

### Transaction Patterns
- Electronics is the most popular category
- 14% negative feedback rate
- Credit/debit cards are preferred payment methods
- Equal distribution across shipping methods
- 16% orders in pending status

### Seasonal Trends
- Peak transactions in April and August
- Consistent weekly distribution with slight Thursday preference

### RFM Segmentation Results
- Clear distinction between customer segments based on purchasing behavior
- Identified high-value customers for targeted marketing
- Discovered potential growth segments

## Technical Implementation

### Requirements
```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
```

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run Jupyter notebook: `jupyter notebook`

## Future Work
- Implement predictive modeling for customer churn
- Develop real-time customer scoring system
- Integration with CRM systems
- Enhanced visualization dashboard
- Customer lifetime value prediction

