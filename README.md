# Customer Segmentation Using K-Means Clustering

## Overview
This project focuses on performing customer segmentation using K-Means clustering, with the aim to identify distinct customer groups based on recency, frequency, and monetary (RFM) metrics. The customer data is analyzed and clustered into three segments to help businesses personalize marketing strategies, improve customer service, and optimize resources.

## Project Structure
- **Data**: The project utilizes a customer dataset (`df2`), containing various attributes such as Customer ID, Transaction details, Age, Gender, Income, Customer Segment, and purchase behavior.
- **RFM Metrics**: The segmentation is based on three core metrics — Recency (time since last purchase), Frequency (number of transactions), and Monetary (total spent).
- **Clustering**: We use the K-Means clustering algorithm to segment the customers into three distinct groups.
- **Evaluation**: The clustering performance is evaluated using Davies-Bouldin and Calinski-Harabasz scores.
- **Visualization**: Various visualizations such as scatter plots, box plots, and PCA projections are used to understand the distribution of customers within the segments and compare the clusters with existing customer segments.

## Steps in the Project

### 1. **Data Preprocessing**
- **Loading Data**: The dataset is loaded and relevant columns are selected for analysis.
- **Feature Engineering**: 
  - Calculating RFM metrics from the dataset: `Recency`, `Frequency`, and `Monetary`.
  - Handling any outliers in the RFM metrics using the Interquartile Range (IQR) method.
  - Standardizing the RFM features using `StandardScaler` to ensure equal importance in the clustering model.

### 2. **K-Means Clustering**
- **Clustering Setup**: We performed K-Means clustering with 3 clusters using the `KMeans` algorithm from scikit-learn.
- **Fitting the Model**: The K-Means model was fitted on the scaled RFM features, and cluster labels were assigned to each customer.
- **Visualizing Clusters**: We visualized the clusters using a scatter plot (`Monetary` vs. `Frequency`) and PCA (Principal Component Analysis) to reduce dimensionality and create a 2D visualization of the clusters.

### 3. **Evaluation of Clustering**
- **Davies-Bouldin Score**: Measures the average similarity ratio of clusters. A lower value indicates better clustering.
- **Calinski-Harabasz Score**: Measures the ratio of the sum of between-cluster dispersion to within-cluster dispersion. A higher value indicates better-defined clusters.
- **Results**: These metrics were calculated to assess the quality of the clustering.

### 4. **Comparison with Existing Customer Segments**
- **Merging Clusters with Customer Segments**: The clustering results were compared with the predefined `Customer_Segment` to evaluate how well the K-Means clustering corresponds to existing segments.
- **Box Plots**: Box plots were used to compare `Monetary`, `Recency`, and `Frequency` distributions across clusters and original customer segments.

### 5. **Results & Visualizations**
- **Scatter Plot**: A scatter plot was created to visualize the clusters in the original feature space (`Monetary` vs. `Frequency`).
- **PCA Plot**: PCA was applied to reduce the data dimensions to two, allowing for a clear 2D visualization of the customer segments.
- **Box Plots**: Box plots were created for both the K-Means clusters and the original customer segments to visually compare the differences in the key metrics (Monetary, Recency, Frequency).

## Code Structure
1. **Data Preprocessing**: 
   - Loading and cleaning data.
   - Calculating the RFM metrics and handling outliers.
   - Standardizing features for clustering.

2. **Clustering & Visualization**:
   - Applying the K-Means clustering algorithm to the RFM data.
   - Visualizing the clusters using scatter plots and PCA.
   - Calculating clustering evaluation scores (Davies-Bouldin and Calinski-Harabasz).

3. **Evaluation & Comparison**:
   - Comparing the clusters with the predefined customer segments.
   - Visualizing the comparison using box plots.

## Future Improvements
1. **Cluster Tuning**: Experiment with different values for the number of clusters (`n_clusters`) and evaluate their impact on the clustering performance.
2. **Advanced Feature Engineering**: Explore additional features such as customer lifetime value, churn prediction, and other relevant metrics for better segmentation.
3. **Clustering Algorithms**: Test other clustering algorithms like DBSCAN or Agglomerative Clustering to compare performance.
4. **Deep Learning**: Use autoencoders or other deep learning techniques for clustering in case of more complex data.

## Dependencies
To run this project, you will need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Conclusion
This project demonstrates the application of K-Means clustering on customer data to identify meaningful customer segments. The segments can provide valuable insights for marketing, customer retention, and personalized experiences.
