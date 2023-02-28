# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_excel("online_retail.xlsx")

# Data Cleaning
# Remove missing values and canceled orders
df.dropna(inplace=True)

df.head()
df.info()
df.describe()

# Convert InvoiceDate to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract year and month from InvoiceDate
df['YearMonth'] = df['InvoiceDate'].apply(lambda x: x.strftime('%Y-%m'))

# Create a new column for total sales
df['TotalSales'] = df['Quantity'] * df['Price']

# Perform RFM analysis
# Recency
recency_df = df.groupby('Customer ID')['InvoiceDate'].max().reset_index()
recency_df['DaysSinceLastPurchase'] = (recency_df['InvoiceDate'].max() - recency_df['InvoiceDate']).dt.days
recency_df.drop('InvoiceDate', axis=1, inplace=True)

# Frequency
frequency_df = df.groupby('Customer ID')['Invoice'].nunique().reset_index()
frequency_df.columns = ['Customer ID', 'Frequency']

# Monetary
monetary_df = df.groupby('Customer ID')['TotalSales'].sum().reset_index()
monetary_df.columns = ['Customer ID', 'Monetary']

# Merge RFM dataframes
rfm_df = recency_df.merge(frequency_df, on='Customer ID').merge(monetary_df, on='Customer ID')
rfm_df.set_index('Customer ID', inplace=True)

# Compute total sales by year and month
sales_by_month = df.groupby('YearMonth')['TotalSales'].sum().reset_index()

# Visualize total sales by year and month
plt.figure(figsize=(12,6))
sns.lineplot(x='YearMonth', y='TotalSales', data=sales_by_month)
plt.xlabel('Year-Month')
plt.ylabel('Total Sales')
plt.title('Total Sales by Month and Year')
plt.show()

# Top selling products
top_products = df.groupby('Description')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False)
top_products.head()

# Top selling product categories
df['Category'] = df['Description'].apply(lambda x: x.split()[0])
top_categories = df.groupby('Category')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False)
top_categories.head()

# Histogram of total sales
plt.hist(df['TotalSales'])
plt.xlabel('TotalSales')
plt.ylabel('Frequency')
plt.show()

# Bar plot of product categories
plt.figure(figsize=(250,150))
sns.countplot(x='Category', data=df)
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Line plot of total sales by month
plt.figure(figsize=(10,5))
df.groupby(['YearMonth'])['TotalSales'].sum().plot()
plt.xlabel('YearMonth')
plt.ylabel('TotalSales')
plt.show()

# Feature scaling
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df)

# Determine optimal number of clusters
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# Choose 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(rfm_scaled)
cluster_labels = kmeans.labels_

# Add cluster labels to original dataframe
rfm_df['Cluster'] = cluster_labels

# Define targeted marketing strategies for each customer segment
def target_marketing(row):
    if row['Cluster'] == 0:
        return 'Low-hanging fruit: These customers are loyal and valuable. Target them with upsell and cross-sell opportunities.'
    elif row['Cluster'] == 1:
        return 'Growth potential: These customers are fairly loyal but spend relatively little. Target them with special offers and promotions.'
    elif row['Cluster'] == 2:
        return 'At-risk customers: These customers are not loyal and do not spend much. Target them with retention strategies, such as personalized offers or loyalty programs.'

# Apply targeted marketing strategies to each customer
rfm_df['Marketing Strategy'] = rfm_df.apply(target_marketing, axis=1)

# Visualize clusters and marketing strategies
sns.scatterplot(x='Frequency', y='Monetary', data=rfm_df, hue='Cluster', palette='deep')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Customer Segmentation')
plt.show()

print(rfm_df.head())