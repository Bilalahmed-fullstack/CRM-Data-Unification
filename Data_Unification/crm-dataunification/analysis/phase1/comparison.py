import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import itertools

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

# 1. Text preprocessing function (no hardcoded patterns)
def preprocess_text(text):
    """Generic text preprocessing without hardcoded patterns"""
    if not isinstance(text, str):
        text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Replace common separators with space (but keep the original structure too)
    text = re.sub(r'[_\-\s]+', ' ', text)
    
    # Remove punctuation (optional, keep numbers)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_tokens(text):
    """Extract meaningful tokens from column names"""
    text = preprocess_text(text)
    
    # Split into words
    words = text.split()
    
    # Keep tokens that might be meaningful
    # We'll keep all tokens initially and let TF-IDF handle importance
    return words

# 2. Create column feature vectors
all_columns = []
column_to_dataset = {}
for dataset, columns in datasets.items():
    for col in columns:
        col_id = f"{dataset}.{col}"
        all_columns.append(col_id)
        column_to_dataset[col_id] = dataset

# Vectorize column names using TF-IDF with n-grams
vectorizer = TfidfVectorizer(
    analyzer='word',
    tokenizer=extract_tokens,
    ngram_range=(1, 3),  # Capture single words, bi-grams, and tri-grams
    min_df=1,
    max_df=0.9,
    stop_words=None  # Let the algorithm learn
)

# Create feature matrix
X = vectorizer.fit_transform(all_columns)
feature_names = vectorizer.get_feature_names_out()

print(f"Total columns: {len(all_columns)}")
print(f"Vocabulary size: {len(feature_names)}")

# 3. Calculate similarity between columns
cosine_sim_matrix = cosine_similarity(X)

# Create similarity dataframe
similarity_df = pd.DataFrame(
    cosine_sim_matrix,
    index=all_columns,
    columns=all_columns
)

# 4. Function to compare first dataset with all others
def compare_first_dataset_with_others(reference_dataset='legacy_customers', top_n=5):
    """Compare reference dataset with all other datasets"""
    
    print(f"\n{'='*80}")
    print(f"COMPARING {reference_dataset.upper()} WITH ALL OTHER DATASETS")
    print(f"{'='*80}")
    
    # Get all columns from reference dataset
    ref_columns = [f"{reference_dataset}.{col}" for col in datasets[reference_dataset]]
    
    dataset_similarities = {}
    detailed_comparisons = {}
    
    # Compare with each other dataset
    for target_dataset in datasets.keys():
        if target_dataset == reference_dataset:
            continue
            
        target_columns = [f"{target_dataset}.{col}" for col in datasets[target_dataset]]
        
        # Calculate similarity scores between all pairs
        similarity_scores = []
        column_mappings = []
        
        for ref_col in ref_columns:
            best_match = None
            best_score = 0
            
            for target_col in target_columns:
                score = similarity_df.loc[ref_col, target_col]
                if score > best_score:
                    best_score = score
                    best_match = target_col
            
            if best_match and best_score > 0.1:  # Threshold for meaningful similarity
                similarity_scores.append(best_score)
                column_mappings.append({
                    'reference_column': ref_col,
                    'target_column': best_match,
                    'similarity_score': best_score
                })
        
        if similarity_scores:
            avg_similarity = np.mean(similarity_scores)
            max_similarity = np.max(similarity_scores)
            dataset_similarities[target_dataset] = {
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'num_matches': len(similarity_scores),
                'column_mappings': column_mappings
            }
            
            # Store detailed comparison
            detailed_comparisons[target_dataset] = column_mappings
    
    # Sort datasets by similarity
    sorted_datasets = sorted(
        dataset_similarities.items(),
        key=lambda x: x[1]['avg_similarity'],
        reverse=True
    )
    
    # Display results
    print(f"\nSimilarity Ranking (compared to {reference_dataset}):")
    print("-" * 60)
    
    for rank, (dataset, metrics) in enumerate(sorted_datasets[:top_n], 1):
        print(f"{rank}. {dataset}:")
        print(f"   Average Similarity: {metrics['avg_similarity']:.3f}")
        print(f"   Maximum Similarity: {metrics['max_similarity']:.3f}")
        print(f"   Matching Columns: {metrics['num_matches']}/{len(datasets[reference_dataset])}")
        
        # Show top 3 column mappings
        top_mappings = sorted(
            metrics['column_mappings'],
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:3]
        
        print("   Top column mappings:")
        for mapping in top_mappings:
            ref_col_short = mapping['reference_column'].split('.')[1]
            target_col_short = mapping['target_column'].split('.')[1]
            print(f"     {ref_col_short} ↔ {target_col_short} (score: {mapping['similarity_score']:.3f})")
        print()
    
    return sorted_datasets, detailed_comparisons

# 5. Perform comparison for the first dataset
reference_dataset = list(datasets.keys())[0]  # legacy_customers
sorted_datasets, detailed_comparisons = compare_first_dataset_with_others(reference_dataset)

# 6. Deep analysis of top 3 most similar datasets
print(f"\n{'='*80}")
print("DEEP ANALYSIS OF TOP 3 MOST SIMILAR DATASETS")
print(f"{'='*80}")

top_3_datasets = [item[0] for item in sorted_datasets[:3]]

for target_dataset in top_3_datasets:
    print(f"\n{'='*60}")
    print(f"DETAILED COMPARISON: {reference_dataset} ↔ {target_dataset}")
    print(f"{'='*60}")
    
    mappings = detailed_comparisons[target_dataset]
    
    # Group mappings by similarity score ranges
    high_similarity = [m for m in mappings if m['similarity_score'] > 0.7]
    medium_similarity = [m for m in mappings if 0.4 <= m['similarity_score'] <= 0.7]
    low_similarity = [m for m in mappings if m['similarity_score'] < 0.4]
    
    print(f"\nHigh Similarity (>0.7): {len(high_similarity)} columns")
    for mapping in sorted(high_similarity, key=lambda x: x['similarity_score'], reverse=True)[:5]:
        ref = mapping['reference_column'].split('.')[1]
        tgt = mapping['target_column'].split('.')[1]
        print(f"  ✓ {ref} ↔ {tgt} (score: {mapping['similarity_score']:.3f})")
    
    print(f"\nMedium Similarity (0.4-0.7): {len(medium_similarity)} columns")
    for mapping in sorted(medium_similarity, key=lambda x: x['similarity_score'], reverse=True)[:3]:
        ref = mapping['reference_column'].split('.')[1]
        tgt = mapping['target_column'].split('.')[1]
        print(f"  ~ {ref} ↔ {tgt} (score: {mapping['similarity_score']:.3f})")
    
    print(f"\nLow Similarity (<0.4): {len(low_similarity)} columns")
    for mapping in sorted(low_similarity, key=lambda x: x['similarity_score'], reverse=True)[:2]:
        ref = mapping['reference_column'].split('.')[1]
        tgt = mapping['target_column'].split('.')[1]
        print(f"  ? {ref} ↔ {tgt} (score: {mapping['similarity_score']:.3f})")
    
    # Calculate coverage
    coverage = len(mappings) / len(datasets[reference_dataset]) * 100
    print(f"\nCoverage: {coverage:.1f}% ({len(mappings)}/{len(datasets[reference_dataset])} columns)")
    
    # Identify unique columns in each dataset
    ref_cols_set = set([f"{reference_dataset}.{col}" for col in datasets[reference_dataset]])
    target_cols_set = set([f"{target_dataset}.{col}" for col in datasets[target_dataset]])
    
    matched_ref_cols = set([m['reference_column'] for m in mappings])
    matched_target_cols = set([m['target_column'] for m in mappings])
    
    unique_to_ref = ref_cols_set - matched_ref_cols
    unique_to_target = target_cols_set - matched_target_cols
    
    print(f"\nUnique to {reference_dataset}: {len(unique_to_ref)} columns")
    for col in sorted(list(unique_to_ref))[:3]:
        print(f"  - {col.split('.')[1]}")
    
    print(f"\nUnique to {target_dataset}: {len(unique_to_target)} columns")
    for col in sorted(list(unique_to_target))[:3]:
        print(f"  - {col.split('.')[1]}")

# 7. Calculate pairwise dataset similarity matrix
print(f"\n{'='*80}")
print("PAIRWISE DATASET SIMILARITY MATRIX")
print(f"{'='*80}")

dataset_names = list(datasets.keys())
similarity_matrix = np.zeros((len(dataset_names), len(dataset_names)))

for i, dataset1 in enumerate(dataset_names):
    for j, dataset2 in enumerate(dataset_names):
        if i == j:
            similarity_matrix[i, j] = 1.0
            continue
        
        cols1 = [f"{dataset1}.{col}" for col in datasets[dataset1]]
        cols2 = [f"{dataset2}.{col}" for col in datasets[dataset2]]
        
        # Calculate average similarity between all column pairs
        similarities = []
        for col1 in cols1:
            best_similarity = 0
            for col2 in cols2:
                sim = similarity_df.loc[col1, col2]
                if sim > best_similarity:
                    best_similarity = sim
            similarities.append(best_similarity)
        
        similarity_matrix[i, j] = np.mean(similarities) if similarities else 0

# Create similarity dataframe
similarity_df_datasets = pd.DataFrame(
    similarity_matrix,
    index=dataset_names,
    columns=dataset_names
)

print("\nDataset Similarity Matrix (averages):")
print("-" * 60)
print(similarity_df_datasets.round(3))

# 8. Find most similar dataset pairs overall
print(f"\n{'='*80}")
print("MOST SIMILAR DATASET PAIRS (Overall)")
print(f"{'='*80}")

similar_pairs = []
for i in range(len(dataset_names)):
    for j in range(i + 1, len(dataset_names)):
        dataset1 = dataset_names[i]
        dataset2 = dataset_names[j]
        similarity = similarity_matrix[i, j]
        
        similar_pairs.append({
            'dataset1': dataset1,
            'dataset2': dataset2,
            'similarity': similarity
        })

# Sort by similarity
similar_pairs_sorted = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)

print("\nTop 5 most similar dataset pairs:")
print("-" * 60)
for i, pair in enumerate(similar_pairs_sorted[:5], 1):
    print(f"{i}. {pair['dataset1']} ↔ {pair['dataset2']}: {pair['similarity']:.3f}")
    
    # Show why they're similar
    cols1 = datasets[pair['dataset1']]
    cols2 = datasets[pair['dataset2']]
    
    # Find best matching columns
    top_matches = []
    for col1 in cols1:
        col1_full = f"{pair['dataset1']}.{col1}"
        best_match = None
        best_score = 0
        
        for col2 in cols2:
            col2_full = f"{pair['dataset2']}.{col2}"
            score = similarity_df.loc[col1_full, col2_full]
            if score > best_score:
                best_score = score
                best_match = col2
        
        if best_match and best_score > 0.5:
            top_matches.append((col1, best_match, best_score))
    
    # Show top 3 matches
    top_matches_sorted = sorted(top_matches, key=lambda x: x[2], reverse=True)[:3]
    for col1, col2, score in top_matches_sorted:
        print(f"   {col1} ↔ {col2} (score: {score:.3f})")
    print()

# 9. Cluster datasets based on similarity
print(f"\n{'='*80}")
print("DATASET CLUSTERING BASED ON SIMILARITY")
print(f"{'='*80}")

from sklearn.cluster import AgglomerativeClustering

# Convert similarity to distance
distance_matrix = 1 - similarity_matrix

# Perform hierarchical clustering
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.8,  # Adjust this threshold
    metric='precomputed',
    linkage='average'
)

cluster_labels = clustering.fit_predict(distance_matrix)

# Group datasets by cluster
cluster_groups = defaultdict(list)
for dataset, cluster_id in zip(dataset_names, cluster_labels):
    cluster_groups[cluster_id].append(dataset)

print("\nDataset Clusters:")
print("-" * 60)
for cluster_id, datasets_in_cluster in cluster_groups.items():
    print(f"\nCluster {cluster_id}: {len(datasets_in_cluster)} datasets")
    for dataset in datasets_in_cluster:
        print(f"  - {dataset}")
    
    # Analyze what's common in this cluster
    if len(datasets_in_cluster) > 1:
        all_cols = []
        for dataset in datasets_in_cluster:
            all_cols.extend(datasets[dataset])
        
        # Find common terms in column names
        from collections import Counter
        all_terms = []
        for col in all_cols:
            all_terms.extend(extract_tokens(col))
        
        common_terms = Counter(all_terms).most_common(5)
        print(f"  Common terms: {[term for term, count in common_terms]}")

# 10. Export results for further analysis
print(f"\n{'='*80}")
print("EXPORTING RESULTS")
print(f"{'='*80}")

# Create comprehensive results dataframe
results = []
for col1 in all_columns:
    for col2 in all_columns:
        if col1 < col2:  # Avoid duplicates and self-comparisons
            dataset1 = col1.split('.')[0]
            dataset2 = col2.split('.')[0]
            if dataset1 == dataset2:
                continue
            column1 = col1.split('.')[1]
            column2 = col2.split('.')[1]
            
            similarity = similarity_df.loc[col1, col2]
            
            results.append({
                'dataset1': dataset1,
                'dataset2': dataset2,
                'column1': column1,
                'column2': column2,
                'similarity': similarity
            })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('similarity', ascending=False)

# Save to CSV
results_df.to_csv('dataset_column_similarity_analysis.csv', index=False)

# Save dataset similarity matrix
similarity_df_datasets.to_csv('dataset_similarity_matrix.csv')

print("\nResults exported:")
print("- dataset_column_similarity_analysis.csv: Detailed column-level similarities")
print("- dataset_similarity_matrix.csv: Dataset-level similarity matrix")
print("\nAnalysis complete!")