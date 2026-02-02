import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import re
from collections import defaultdict

# Define the datasets
datasets = {
    'legacy_customers': ['customer_id', 'first_name', 'last_name', 'email_address', 'phone_number', 
                         'street_address', 'city', 'state', 'zip_code', 'country', 
                         'registration_date', 'premium_member'],
    
    'legacy_products': ['product_sku', 'product_name', 'product_category', 'brand', 'manufacturer', 
                        'list_price', 'weight_lbs', 'dimensions', 'is_active', 'inventory_count'],
    
    'legacy_sales': ['order_id', 'customer_id', 'product_sku', 'quantity', 'unit_price', 
                     'total_amount', 'order_date', 'shipping_status', 'payment_method', 'shipping_address'],
    
    'modern_customers': ['user_id', 'full_name', 'email', 'phone', 'shipping_city', 
                         'shipping_state', 'shipping_zip', 'account_created', 'prime_member', 'loyalty_tier'],
    
    'modern_products': ['asin', 'title', 'category', 'subcategory', 'seller', 'brand', 
                        'current_price', 'weight_kg', 'specifications', 'stock_quantity', 'product_condition'],
    
    'modern_sales': ['transaction_id', 'user_id', 'product_asin', 'qty', 'price_per_item', 
                     'transaction_total', 'timestamp', 'fulfillment_channel', 'payment_type', 'promotion_applied'],
    
    'third_customers': ['account_id', 'customer_name', 'email_addr', 'contact_number', 'address_line', 
                        'city_name', 'state_province', 'postal_code', 'country_code', 'signup_date', 'membership_status'],
    
    'third_products': ['item_code', 'item_name', 'category_name', 'brand_name', 'supplier', 
                       'retail_price', 'item_weight', 'product_dimensions', 'availability_status', 'stock_level'],
    
    'third_sales': ['sale_id', 'account_id', 'item_code', 'order_quantity', 'item_price', 
                    'sale_total', 'sale_date', 'delivery_status', 'payment_option', 'customer_address']
}

# Preprocess column names for better clustering
def preprocess_column_name(col_name):
    """Normalize column names for clustering"""
    # Convert to lowercase
    col = col_name.lower()
    
    # Replace common separators with space
    col = re.sub(r'[_-]', ' ', col)
    
    # Remove common suffixes/prefixes
    words_to_remove = ['_', '-', 'field', 'column', 'data', 'info']
    for word in words_to_remove:
        col = col.replace(word, '')
    
    # Split into words and remove duplicates
    words = col.split()
    
    # Sort words to ensure consistent representation
    return ' '.join(sorted(set(words)))

# Create a DataFrame of all columns
all_columns = []
for dataset, columns in datasets.items():
    for col in columns:
        all_columns.append({
            'dataset': dataset,
            'original_column': col,
            'processed_column': preprocess_column_name(col),
            'column_type': dataset.split('_')[1]  # customers, products, or sales
        })

df_columns = pd.DataFrame(all_columns)

# Create feature vectors using TF-IDF
vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),  # Use single words and bi-grams
    min_df=1  # Include all terms
)

X = vectorizer.fit_transform(df_columns['processed_column'])
feature_names = vectorizer.get_feature_names_out()

print(f"Number of columns: {len(df_columns)}")
print(f"Vocabulary size: {len(feature_names)}")
print("\nTop terms in vocabulary:")
for term in feature_names[:20]:
    print(f"  - {term}")

## **2. K-Means Clustering**
print("\n" + "="*80)
print("K-MEANS CLUSTERING RESULTS")
print("="*80)

# Determine optimal number of clusters using elbow method
inertias = []
k_range = range(2, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()


# Calculate the differences in inertia
inertia_diffs = np.diff(inertias)
inertia_diff_ratios = inertia_diffs[1:] / inertia_diffs[:-1]

# Find where the reduction rate changes significantly
# Usually where the ratio increases (curve flattens)
elbow_point = np.argmax(inertia_diff_ratios) + 3  # +3 because we start at k=2
print(f"Suggested elbow point: k = {elbow_point}")



# Let's choose 8 clusters for initial analysis
optimal_k = elbow_point
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_columns['kmeans_cluster'] = kmeans.fit_predict(X)

# Display cluster assignments
print(f"\nK-Means Clusters (k={optimal_k}):")
print("-" * 80)

for cluster_id in range(optimal_k):
    cluster_columns = df_columns[df_columns['kmeans_cluster'] == cluster_id]
    
    print(f"\nCLUSTER {cluster_id} - {len(cluster_columns)} columns:")
    print(f"Dataset distribution: {cluster_columns['column_type'].value_counts().to_dict()}")
    
    # Group by dataset to show patterns
    grouped = cluster_columns.groupby('dataset')['original_column'].apply(list)
    
    print("\nSample columns from this cluster:")
    for dataset, cols in grouped.items():
        if len(cols) > 0:
            print(f"  {dataset}:")
            for col in cols[:3]:  # Show first 3 columns
                print(f"    - {col}")
            if len(cols) > 3:
                print(f"    ... and {len(cols) - 3} more")
    
    # Identify common themes
    cluster_terms = []
    cluster_indices = cluster_columns.index.tolist()
    if cluster_indices:
        cluster_vectors = X[cluster_indices]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        # Get top terms for this cluster
        top_indices = cluster_center.argsort()[-5:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        cluster_terms.append(top_terms)
    
    print(f"  Common terms: {top_terms}")

## **3. Hierarchical Clustering for Dendrogram**
print("\n" + "="*80)
print("HIERARCHICAL CLUSTERING DENDROGRAM")
print("="*80)

# Use cosine similarity for hierarchical clustering
cosine_sim = cosine_similarity(X)
linkage_matrix = linkage(cosine_sim, method='ward')

# Plot dendrogram
plt.figure(figsize=(15, 10))
dendrogram(
    linkage_matrix,
    labels=df_columns['original_column'].tolist(),
    leaf_rotation=90,
    leaf_font_size=8
)
plt.title('Column Similarity Dendrogram')
plt.xlabel('Columns')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

## **4. DBSCAN Clustering for Density-based grouping**
print("\n" + "="*80)
print("DBSCAN CLUSTERING (Density-based)")
print("="*80)

dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
df_columns['dbscan_cluster'] = dbscan.fit_predict(cosine_sim)

n_clusters_dbscan = len(set(df_columns['dbscan_cluster'])) - (1 if -1 in df_columns['dbscan_cluster'] else 0)
n_noise = list(df_columns['dbscan_cluster']).count(-1)

print(f"DBSCAN found {n_clusters_dbscan} clusters and {n_noise} noise points")

# Show meaningful clusters (excluding noise -1)
for cluster_id in sorted(set(df_columns['dbscan_cluster'])):
    if cluster_id == -1:
        continue
        
    cluster_data = df_columns[df_columns['dbscan_cluster'] == cluster_id]
    
    if len(cluster_data) >= 2:  # Only show clusters with at least 2 items
        print(f"\nDBSCAN Cluster {cluster_id} ({len(cluster_data)} columns):")
        
        # Get common words in this cluster
        all_words = ' '.join(cluster_data['processed_column']).split()
        from collections import Counter
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(5) if count > 1]
        
        print(f"  Common words: {common_words}")
        print("  Columns:")
        for _, row in cluster_data.iterrows():
            print(f"    [{row['dataset']}] {row['original_column']}")

## **5. Semantic Grouping Analysis**
print("\n" + "="*80)
print("SEMANTIC COLUMN GROUPING ANALYSIS")
print("="*80)

# Define semantic categories based on column names
semantic_categories = {
    'identifier': ['id', 'sku', 'asin', 'code', 'number'],
    'name': ['name', 'title'],
    'contact': ['email', 'phone', 'contact'],
    'address': ['address', 'city', 'state', 'zip', 'postal', 'country'],
    'date': ['date', 'time', 'created', 'signup', 'registration', 'timestamp'],
    'price': ['price', 'amount', 'cost', 'total', 'retail', 'current'],
    'quantity': ['quantity', 'qty', 'count', 'inventory', 'stock'],
    'status': ['status', 'active', 'premium', 'member', 'prime', 'loyalty', 'availability'],
    'measurement': ['weight', 'dimension', 'lbs', 'kg'],
    'category': ['category', 'type', 'subcategory'],
    'brand': ['brand', 'manufacturer', 'supplier', 'seller']
}

def categorize_column(col_name):
    """Categorize column based on semantic meaning"""
    col_lower = col_name.lower()
    categories = []
    
    for category, keywords in semantic_categories.items():
        for keyword in keywords:
            if keyword in col_lower:
                categories.append(category)
                break
    
    return ', '.join(categories) if categories else 'other'

# Apply categorization
df_columns['semantic_category'] = df_columns['original_column'].apply(categorize_column)

print("\nColumns grouped by semantic category:")
print("-" * 80)

for category in sorted(df_columns['semantic_category'].unique()):
    cat_cols = df_columns[df_columns['semantic_category'] == category]
    
    if len(cat_cols) > 1:  # Only show categories with multiple columns
        print(f"\n{category.upper()} ({len(cat_cols)} columns):")
        
        # Group by column type
        for col_type in sorted(cat_cols['column_type'].unique()):
            type_cols = cat_cols[cat_cols['column_type'] == col_type]
            print(f"  {col_type}:")
            for _, row in type_cols.iterrows():
                print(f"    - {row['dataset']}.{row['original_column']}")

## **6. Cross-Table Column Mapping**
print("\n" + "="*80)
print("CROSS-TABLE COLUMN MAPPING SUGGESTIONS")
print("="*80)

# Find potential mappings between customer tables
customer_tables = ['legacy_customers', 'modern_customers', 'third_customers']

print("\nCUSTOMER TABLE COLUMN MAPPINGS:")
for col_category in ['identifier', 'name', 'email', 'phone', 'date']:
    print(f"\n{col_category.upper()} columns:")
    
    category_cols = df_columns[
        (df_columns['column_type'] == 'customers') & 
        (df_columns['semantic_category'].str.contains(col_category))
    ]
    
    for _, row in category_cols.iterrows():
        print(f"  {row['dataset']}: {row['original_column']}")

# Find potential mappings between product tables
print("\n\nPRODUCT TABLE COLUMN MAPPINGS:")
product_tables = ['legacy_products', 'modern_products', 'third_products']

# Product identifier mapping
print("\nProduct Identifiers:")
for table in product_tables:
    cols = df_columns[
        (df_columns['dataset'] == table) & 
        (df_columns['semantic_category'].str.contains('identifier'))
    ]
    if not cols.empty:
        print(f"  {table}: {cols['original_column'].iloc[0]}")

## **7. Create Mapping Dictionary**
print("\n" + "="*80)
print("FINAL COLUMN MAPPING DICTIONARY")
print("="*80)

# Create a mapping dictionary for data integration
column_mapping = {
    'customer_identifier': {
        'legacy_customers': 'customer_id',
        'modern_customers': 'user_id',
        'third_customers': 'account_id',
        'legacy_sales': 'customer_id',
        'modern_sales': 'user_id',
        'third_sales': 'account_id'
    },
    'product_identifier': {
        'legacy_products': 'product_sku',
        'modern_products': 'asin',
        'third_products': 'item_code',
        'legacy_sales': 'product_sku',
        'modern_sales': 'product_asin',
        'third_sales': 'item_code'
    },
    'customer_name': {
        'legacy_customers': ['first_name', 'last_name'],
        'modern_customers': 'full_name',
        'third_customers': 'customer_name'
    },
    'email': {
        'legacy_customers': 'email_address',
        'modern_customers': 'email',
        'third_customers': 'email_addr'
    },
    'price': {
        'legacy_products': 'list_price',
        'modern_products': 'current_price',
        'third_products': 'retail_price',
        'legacy_sales': 'unit_price',
        'modern_sales': 'price_per_item',
        'third_sales': 'item_price'
    }
}

print("Recommended column mappings for data integration:")
for mapping_type, tables in column_mapping.items():
    print(f"\n{mapping_type.upper()}:")
    for table, column in tables.items():
        if isinstance(column, list):
            print(f"  {table}: {', '.join(column)}")
        else:
            print(f"  {table}: {column}")

# Save results to CSV
# df_columns.to_csv('column_clustering_analysis.csv', index=False)
print("\nAnalysis saved to 'column_clustering_analysis.csv'")