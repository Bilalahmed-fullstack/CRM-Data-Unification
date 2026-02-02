# import pandas as pd
# import numpy as np
# from collections import defaultdict, Counter
# import re
# import hashlib
# from datasketch import MinHash, MinHashLSH
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# import warnings
# warnings.filterwarnings('ignore')

# class OptimizedDeduplicator:
#     def __init__(self, n_workers=4):
#         """
#         Optimized deduplication pipeline with improved performance.
#         """
#         self.df = None
#         self.column_types = {}
#         self.text_columns = []
#         self.numeric_columns = []
#         self.categorical_columns = []
#         self.id_columns = []
        
#     def load_csv(self, csv_path):
#         """Fast CSV loading with chunking for large files"""
#         print(f"📂 Loading {csv_path}...")
        
#         # Estimate file size
#         import os
#         file_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
        
#         # Use chunking for large files
#         if file_size > 50 * 1024 * 1024:  # > 50MB
#             print(f"⚠️ Large file detected ({file_size/1024/1024:.1f} MB), using chunking...")
#             chunks = []
#             for chunk in pd.read_csv(csv_path, chunksize=10000, dtype=str, low_memory=False):
#                 chunks.append(chunk)
#             self.df = pd.concat(chunks, ignore_index=True)
#         else:
#             self.df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        
#         # Fast column cleaning
#         self.df.columns = [re.sub(r'[^\w]', '_', str(col).strip().lower()) for col in self.df.columns]
        
#         print(f"📊 Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
#         return self.df
    
#     def fast_column_analysis(self):
#         """Fast column type analysis using sampling"""
#         print("\n🔍 Fast column analysis...")
        
#         # Sample data for analysis (for large datasets)
#         sample_size = min(10000, len(self.df))
#         sample = self.df.sample(n=sample_size, random_state=42) if len(self.df) > 10000 else self.df
        
#         for column in self.df.columns:
#             col_data = sample[column]
            
#             # Check if column is mostly numeric
#             numeric_pct = self._fast_numeric_check(col_data)
            
#             if numeric_pct > 0.8:
#                 self.column_types[column] = 'numeric'
#                 self.numeric_columns.append(column)
                
#             elif col_data.nunique() / len(col_data.dropna()) < 0.3 and len(col_data.dropna()) > 10:
#                 self.column_types[column] = 'categorical'
#                 self.categorical_columns.append(column)
                
#             else:
#                 self.column_types[column] = 'text'
#                 self.text_columns.append(column)
                
#                 # Fast ID detection
#                 if self._fast_id_detection(col_data):
#                     self.id_columns.append(column)
        
#         print(f"📋 Detected: {len(self.text_columns)} text, {len(self.numeric_columns)} numeric, "
#               f"{len(self.categorical_columns)} categorical columns")
        
#         if self.id_columns:
#             print(f"🔑 ID columns: {self.id_columns[:3]}")  # Show only first 3
    
#     def _fast_numeric_check(self, series):
#         """Fast numeric detection using regex"""
#         if len(series.dropna()) == 0:
#             return 0
        
#         # Sample for faster checking
#         sample = series.dropna().head(1000)
        
#         # Simple regex check for numbers
#         numeric_pattern = re.compile(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$')
#         numeric_count = sum(1 for val in sample if numeric_pattern.match(str(val).replace(',', '').replace('$', '').strip()))
        
#         return numeric_count / len(sample)
    
#     def _fast_id_detection(self, series):
#         """Fast ID column detection"""
#         sample = series.dropna().head(100)
#         if len(sample) == 0:
#             return False
        
#         # Check uniqueness ratio quickly
#         unique_ratio = series.nunique() / max(1, len(series.dropna()))
        
#         # Common ID patterns
#         id_patterns = [
#             r'^\d{5,}',  # Long numbers
#             r'^[A-Z]{2,}-\d{3,}',  # AB-123 format
#             r'^[A-Z0-9]{8,}',  # Long alphanumeric
#         ]
        
#         pattern_matches = 0
#         for val in sample:
#             val_str = str(val)
#             for pattern in id_patterns:
#                 if re.match(pattern, val_str, re.IGNORECASE):
#                     pattern_matches += 1
#                     break
        
#         return unique_ratio > 0.95 or (pattern_matches / len(sample) > 0.5)
    
#     def fast_preprocessing(self):
#         """Vectorized preprocessing for speed"""
#         print("\n⚡ Fast preprocessing...")
        
#         df = self.df.copy()
        
#         # Vectorized text cleaning
#         for col in self.text_columns:
#             df[f"{col}_clean"] = df[col].fillna('').astype(str).str.lower()
#             # Remove extra whitespace
#             df[f"{col}_clean"] = df[f"{col}_clean"].str.replace(r'\s+', ' ', regex=True)
        
#         # Vectorized numeric extraction
#         for col in self.numeric_columns:
#             # Extract numbers using vectorized operations
#             df[f"{col}_clean"] = pd.to_numeric(
#                 df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
#                 errors='coerce'
#             )
        
#         # Vectorized categorical cleaning
#         for col in self.categorical_columns:
#             df[f"{col}_clean"] = df[col].fillna('unknown').astype(str).str.lower().str.strip()
        
#         return df
    
#     def create_fingerprint_blocks(self, df):
#         """Create blocking using MinHash/LSH for near-duplicate detection"""
#         print("\n🔑 Creating fingerprint blocks...")
        
#         # Strategy 1: Use ID columns for exact matching
#         if self.id_columns:
#             id_col = self.id_columns[0]
#             df['exact_block'] = df[f"{id_col}_clean"].fillna('')
#             print(f"✅ Created exact blocks using '{id_col}'")
        
#         # Strategy 2: Create text fingerprints for fuzzy matching
#         if self.text_columns:
#             # Choose the most descriptive text column
#             best_text_col = self._select_best_text_column(df)
            
#             if best_text_col:
#                 print(f"📝 Creating MinHash fingerprints for '{best_text_col}'...")
                
#                 # Create MinHash signatures
#                 df['minhash'] = df[f"{best_text_col}_clean"].apply(self._create_minhash)
                
#                 # Use LSH for efficient near-duplicate detection
#                 lsh = MinHashLSH(threshold=0.5, num_perm=128)
                
#                 # Add hashes to LSH index
#                 for idx, mh in enumerate(df['minhash']):
#                     lsh.insert(idx, mh)
                
#                 # Find candidates efficiently
#                 candidate_pairs = set()
#                 for idx in range(len(df)):
#                     candidates = lsh.query(df['minhash'].iloc[idx])
#                     for cand_idx in candidates:
#                         if cand_idx > idx:  # Avoid duplicates
#                             candidate_pairs.add((idx, cand_idx))
                
#                 print(f"✅ Found {len(candidate_pairs)} candidate pairs via LSH")
#                 return list(candidate_pairs)
        
#         return []
    
#     def _select_best_text_column(self, df):
#         """Select best text column for fingerprinting"""
#         if not self.text_columns:
#             return None
        
#         # Score columns by: completeness * average length * uniqueness
#         scores = {}
#         for col in self.text_columns[:5]:  # Limit to first 5
#             clean_col = f"{col}_clean"
#             if clean_col in df.columns:
#                 # Completeness
#                 completeness = df[clean_col].notna().mean()
                
#                 # Average length (non-empty)
#                 avg_len = df[clean_col].str.len().mean()
                
#                 # Uniqueness (higher is better for discrimination)
#                 uniqueness = df[clean_col].nunique() / len(df)
                
#                 scores[col] = completeness * avg_len * uniqueness
        
#         return max(scores.items(), key=lambda x: x[1])[0] if scores else None
    
#     def _create_minhash(self, text, num_perm=128):
#         """Create MinHash for text"""
#         m = MinHash(num_perm=num_perm)
        
#         if not text or not isinstance(text, str):
#             return m
        
#         # Use shingles (character n-grams)
#         shingle_size = 3
#         for i in range(len(text) - shingle_size + 1):
#             shingle = text[i:i+shingle_size]
#             m.update(shingle.encode('utf-8'))
        
#         return m
    
#     def fast_similarity_calculation(self, df, candidate_pairs):
#         """Fast similarity calculation using vectorized operations"""
#         print(f"\n⚡ Calculating similarities for {len(candidate_pairs)} pairs...")
        
#         similarities = []
        
#         # Precompute numeric arrays for faster access
#         numeric_arrays = {}
#         for col in self.numeric_columns:
#             clean_col = f"{col}_clean"
#             if clean_col in df.columns:
#                 numeric_arrays[col] = df[clean_col].fillna(0).values
        
#         # Process candidate pairs in batches
#         batch_size = 10000
#         for i in range(0, len(candidate_pairs), batch_size):
#             batch = candidate_pairs[i:i+batch_size]
            
#             for idx1, idx2 in batch:
#                 try:
#                     sim = self._fast_pair_similarity(df, idx1, idx2, numeric_arrays)
#                     if sim > 0.7:  # Threshold for potential duplicates
#                         similarities.append((idx1, idx2, sim))
#                 except:
#                     continue
        
#         print(f"✅ Found {len(similarities)} high-similarity pairs")
#         return similarities
    
#     def _fast_pair_similarity(self, df, idx1, idx2, numeric_arrays):
#         """Fast similarity calculation for one pair"""
#         row1 = df.iloc[idx1]
#         row2 = df.iloc[idx2]
        
#         total_sim = 0
#         total_weight = 0
        
#         # Text columns similarity (simplified)
#         for col in self.text_columns[:3]:  # Limit to 3 most important text columns
#             clean_col = f"{col}_clean"
#             if clean_col in row1 and clean_col in row2:
#                 val1 = str(row1[clean_col]) if pd.notna(row1[clean_col]) else ""
#                 val2 = str(row2[clean_col]) if pd.notna(row2[clean_col]) else ""
                
#                 if val1 and val2:
#                     # Fast text similarity (token overlap)
#                     tokens1 = set(val1.split())
#                     tokens2 = set(val2.split())
                    
#                     if tokens1 and tokens2:
#                         intersection = len(tokens1.intersection(tokens2))
#                         union = len(tokens1.union(tokens2))
#                         sim = intersection / union if union > 0 else 0
                        
#                         # Boost for exact match
#                         if val1 == val2:
#                             sim = 1.0
                        
#                         total_sim += sim * 0.4
#                         total_weight += 0.4
        
#         # Numeric columns similarity
#         for col in self.numeric_columns[:2]:  # Limit to 2 most important numeric columns
#             if col in numeric_arrays:
#                 val1 = numeric_arrays[col][idx1]
#                 val2 = numeric_arrays[col][idx2]
                
#                 if val1 != 0 or val2 != 0:
#                     # Fast numeric similarity
#                     max_val = max(abs(val1), abs(val2))
#                     if max_val > 0:
#                         sim = 1 - min(abs(val1 - val2) / max_val, 1)
#                         total_sim += sim * 0.3
#                         total_weight += 0.3
        
#         # Categorical columns similarity
#         for col in self.categorical_columns[:2]:  # Limit to 2
#             clean_col = f"{col}_clean"
#             if clean_col in row1 and clean_col in row2:
#                 val1 = str(row1[clean_col]) if pd.notna(row1[clean_col]) else ""
#                 val2 = str(row2[clean_col]) if pd.notna(row2[clean_col]) else ""
                
#                 if val1 and val2:
#                     sim = 1.0 if val1 == val2 else 0.0
#                     total_sim += sim * 0.3
#                     total_weight += 0.3
        
#         return total_sim / total_weight if total_weight > 0 else 0
    
#     def fast_clustering(self, df, similar_pairs, threshold=0.8):
#         """Fast clustering using Union-Find algorithm"""
#         print(f"\n🎯 Clustering {len(similar_pairs)} high-similarity pairs...")
        
#         # Initialize Union-Find structure
#         parent = {i: i for i in range(len(df))}
        
#         def find(x):
#             while parent[x] != x:
#                 parent[x] = parent[parent[x]]  # Path compression
#                 x = parent[x]
#             return x
        
#         def union(x, y):
#             root_x, root_y = find(x), find(y)
#             if root_x != root_y:
#                 parent[root_y] = root_x
        
#         # Union similar pairs
#         for idx1, idx2, sim in similar_pairs:
#             if sim >= threshold:
#                 union(idx1, idx2)
        
#         # Collect clusters
#         clusters = defaultdict(list)
#         for idx in range(len(df)):
#             clusters[find(idx)].append(idx)
        
#         # Filter out singletons
#         duplicate_clusters = [indices for indices in clusters.values() if len(indices) > 1]
        
#         print(f"✅ Found {len(duplicate_clusters)} duplicate clusters")
        
#         # Assign cluster IDs
#         df = df.copy()
#         df['cluster_id'] = -1
        
#         for cluster_id, indices in enumerate(duplicate_clusters):
#             for idx in indices:
#                 df.at[idx, 'cluster_id'] = cluster_id
        
#         # Assign unique IDs to singletons
#         max_cluster_id = len(duplicate_clusters)
#         singleton_mask = df['cluster_id'] == -1
#         df.loc[singleton_mask, 'cluster_id'] = range(max_cluster_id, 
#                                                     max_cluster_id + singleton_mask.sum())
        
#         self.clusters = duplicate_clusters
#         return df
    
#     def create_golden_records_fast(self, df):
#         """Fast golden record creation using vectorized operations"""
#         print("\n🏆 Creating golden records...")
        
#         cluster_ids = df['cluster_id'].unique()
#         golden_records = []
        
#         for cluster_id in cluster_ids:
#             cluster_df = df[df['cluster_id'] == cluster_id]
            
#             if len(cluster_df) == 0:
#                 continue
            
#             golden = {'cluster_id': cluster_id, 'source_records': len(cluster_df)}
            
#             # Process each column type efficiently
#             for col in self.df.columns:
#                 clean_col = f"{col}_clean"
                
#                 if clean_col not in cluster_df.columns:
#                     continue
                
#                 col_type = self.column_types.get(col, 'text')
#                 col_data = cluster_df[clean_col].dropna()
                
#                 if len(col_data) == 0:
#                     continue
                
#                 if col_type == 'numeric':
#                     # Use median for numeric
#                     try:
#                         golden[col] = float(col_data.median())
#                     except:
#                         golden[col] = col_data.iloc[0]
                
#                 elif col_type == 'categorical':
#                     # Use mode for categorical
#                     golden[col] = col_data.mode().iloc[0] if not col_data.mode().empty else col_data.iloc[0]
                
#                 else:  # text
#                     # Use longest non-empty value
#                     text_values = col_data.astype(str)
#                     non_empty = text_values[text_values.str.strip() != ""]
#                     if len(non_empty) > 0:
#                         golden[col] = non_empty.iloc[non_empty.str.len().argmax()]
#                     else:
#                         golden[col] = col_data.iloc[0]
            
#             golden_records.append(golden)
        
#         golden_df = pd.DataFrame(golden_records)
        
#         # Generate IDs if needed
#         if not any(col in golden_df.columns for col in self.id_columns):
#             golden_df['record_id'] = [f'REC-{i:06d}' for i in range(len(golden_df))]
        
#         print(f"✅ Created {len(golden_df)} golden records")
#         print(f"📊 Reduction: {len(self.df)} → {len(golden_df)} records "
#               f"({(len(self.df)-len(golden_df))/len(self.df)*100:.1f}% reduction)")
        
#         return golden_df
    
#     def save_optimized(self, golden_df, output_path='deduplicated_fast.csv'):
#         """Optimized saving"""
#         print(f"\n💾 Saving to {output_path}...")
        
#         # Remove helper columns
#         keep_cols = [col for col in golden_df.columns 
#                     if not col.endswith('_clean') and col != 'minhash']
        
#         golden_df[keep_cols].to_csv(output_path, index=False)
#         print(f"✅ Saved {len(golden_df)} records")
        
#         return output_path
    
#     def run_fast_pipeline(self, csv_path):
#         """Optimized pipeline for speed"""
#         print("=" * 80)
#         print("⚡ OPTIMIZED DEDUPLICATION PIPELINE")
#         print("=" * 80)
        
#         import time
#         start_time = time.time()
        
#         # 1. Load data
#         self.load_csv(csv_path)
        
#         # 2. Fast analysis
#         self.fast_column_analysis()
        
#         # 3. Fast preprocessing
#         processed_df = self.fast_preprocessing()
        
#         # 4. Create fingerprint blocks and find candidates
#         candidate_pairs = self.create_fingerprint_blocks(processed_df)
        
#         if not candidate_pairs:
#             print("⚠️ No duplicates found. Returning original data.")
#             return self.df
        
#         # 5. Fast similarity calculation
#         similar_pairs = self.fast_similarity_calculation(processed_df, candidate_pairs)
        
#         if not similar_pairs:
#             print("⚠️ No similar pairs found. Returning original data.")
#             return self.df
        
#         # 6. Fast clustering
#         clustered_df = self.fast_clustering(processed_df, similar_pairs)
        
#         # 7. Create golden records
#         golden_df = self.create_golden_records_fast(clustered_df)
        
#         # 8. Save results
#         output_path = self.save_optimized(golden_df)
        
#         end_time = time.time()
        
#         print("\n" + "=" * 80)
#         print("📊 PERFORMANCE SUMMARY")
#         print("=" * 80)
#         print(f"   Time taken: {end_time - start_time:.2f} seconds")
#         print(f"   Original records: {len(self.df)}")
#         print(f"   Golden records: {len(golden_df)}")
#         print(f"   Duplicate clusters: {len(self.clusters) if self.clusters else 0}")
#         print(f"   Output file: {output_path}")
#         print("=" * 80)
        
#         return golden_df


# # Ultra-fast version for very large datasets
# class UltraFastDeduplicator:
#     def __init__(self):
#         """Ultra-fast deduplicator using approximation algorithms"""
#         pass
    
#     def run_ultrafast(self, csv_path, sample_size=10000):
#         """
#         Ultra-fast deduplication using sampling and approximation.
#         Best for exploratory analysis on large datasets.
#         """
#         print("🚀 ULTRA-FAST DEDUPLICATION (Using Sampling)")
        
#         # Load sample
#         total_rows = sum(1 for _ in open(csv_path, 'r', encoding='utf-8', errors='ignore')) - 1
#         print(f"📊 Total rows in file: {total_rows:,}")
        
#         if total_rows > 100000:
#             print(f"⚠️ Large dataset, using {sample_size:,} sample rows")
#             df = pd.read_csv(csv_path, nrows=sample_size)
#         else:
#             df = pd.read_csv(csv_path)
        
#         # Simple deduplication based on exact matches
#         print("\n🔍 Finding exact duplicates...")
        
#         # Find columns that might be unique identifiers
#         potential_id_cols = []
#         for col in df.columns:
#             unique_ratio = df[col].nunique() / len(df)
#             if unique_ratio > 0.9:  # High uniqueness
#                 potential_id_cols.append(col)
        
#         print(f"📋 Potential ID columns: {potential_id_cols[:5]}")
        
#         if potential_id_cols:
#             # Use first potential ID column
#             id_col = potential_id_cols[0]
#             print(f"🔑 Using '{id_col}' for deduplication")
            
#             # Remove exact duplicates
#             initial_count = len(df)
#             df_deduped = df.drop_duplicates(subset=[id_col], keep='first')
#             final_count = len(df_deduped)
            
#             print(f"✅ Removed {initial_count - final_count} exact duplicates")
#             print(f"📊 Before: {initial_count:,} → After: {final_count:,} records")
            
#             # Save results
#             output_path = 'deduplicated_ultrafast.csv'
#             df_deduped.to_csv(output_path, index=False)
#             print(f"💾 Saved to {output_path}")
            
#             return df_deduped
        
#         print("⚠️ No good ID column found. Performing fuzzy deduplication on text columns...")
        
#         # Find text columns
#         text_cols = [col for col in df.columns if df[col].dtype == 'object']
#         if not text_cols:
#             print("❌ No text columns found for fuzzy deduplication")
#             return df
        
#         # Use first text column for fuzzy matching
#         text_col = text_cols[0]
#         print(f"🔍 Using '{text_col}' for fuzzy deduplication")
        
#         # Simple fuzzy deduplication: group by first 20 characters
#         df['text_prefix'] = df[text_col].astype(str).str[:20]
        
#         # Keep first of each group
#         initial_count = len(df)
#         df_deduped = df.drop_duplicates(subset=['text_prefix'], keep='first')
#         final_count = len(df_deduped)
        
#         print(f"✅ Removed {initial_count - final_count} fuzzy duplicates")
#         print(f"📊 Before: {initial_count:,} → After: {final_count:,} records")
        
#         # Remove helper column and save
#         df_deduped = df_deduped.drop(columns=['text_prefix'])
#         output_path = 'deduplicated_ultrafast.csv'
#         df_deduped.to_csv(output_path, index=False)
#         print(f"💾 Saved to {output_path}")
        
#         return df_deduped


# # Choose the right deduplicator based on your needs
# def deduplicate_file(csv_path, mode='balanced'):
#     """
#     Deduplicate any CSV file with optimal performance.
    
#     Parameters:
#     -----------
#     csv_path : str
#         Path to CSV file
#     mode : str
#         'fast' : For large datasets (>100K rows)
#         'balanced' : For medium datasets (10K-100K rows)
#         'accurate' : For small datasets (<10K rows) needing high accuracy
#         'ultrafast' : For very large datasets, using sampling
#     """
    
#     # Estimate dataset size
#     try:
#         import os
#         file_size = os.path.getsize(csv_path)
        
#         # Default to balanced if no mode specified
#         if mode == 'auto':
#             if file_size > 100 * 1024 * 1024:  # > 100MB
#                 mode = 'ultrafast'
#             elif file_size > 10 * 1024 * 1024:  # > 10MB
#                 mode = 'fast'
#             else:
#                 mode = 'balanced'
#     except:
#         mode = 'balanced'  # Fallback
    
#     print(f"🎯 Mode selected: {mode}")
    
#     if mode == 'ultrafast':
#         deduplicator = UltraFastDeduplicator()
#         return deduplicator.run_ultrafast(csv_path)
    
#     elif mode == 'fast':
#         deduplicator = OptimizedDeduplicator()
#         return deduplicator.run_fast_pipeline(csv_path)
    
#     # else:  # balanced or accurate
#     #     # Use the original UniversalDeduplicator for accuracy
#     #     deduplicator = UniversalDeduplicator()
#     #     return deduplicator.run_pipeline(csv_path)


# # Example usage
# if __name__ == "__main__":
#     import sys
    
#     csv_file = "../unified_data/cluster_3_unified.csv"
    
    
#     # Choose mode based on expected size
#     # mode = input("Choose mode [fast/balanced/accurate/ultrafast]: ").strip().lower()
#     # if mode not in ['fast', 'balanced', 'accurate', 'ultrafast', 'auto']:
#     mode = 'fast'
    
#     try:
#         result = deduplicate_file(csv_file, mode)
#         print(f"\n✅ Deduplication completed successfully!")
#     except Exception as e:
#         print(f"\n❌ Error: {e}")



















































# import pandas as pd
# import splink.comparison_library as cl
# from splink import DuckDBAPI, Linker, SettingsCreator

# # -------------------------
# # 1. Read CSV
# # -------------------------
# csv_path = "../unified_data/cluster_3_unified.csv"
# df = pd.read_csv(csv_path)
# df = df.astype(str)
# df["_row_id"] = range(len(df))

# # -------------------------
# # 2. Generic comparisons
# # -------------------------
# comparisons = [
#     cl.JaroWinklerAtThresholds(col, [0.9, 0.7])
#     for col in df.columns
#     if col != "_row_id"
# ]

# # -------------------------
# # 3. Settings (NO training)
# # -------------------------
# settings = SettingsCreator(
#     link_type="dedupe_only",
#     unique_id_column_name="_row_id",
#     comparisons=comparisons
# )

# # -------------------------
# # 4. Linker
# # -------------------------
# linker = Linker(df, settings, DuckDBAPI())

# # -------------------------
# # 5. Predict
# # -------------------------
# pairwise_predictions = linker.inference.predict(
#     threshold_match_weight=-5
# )

# # -------------------------
# # 6. Cluster
# # -------------------------
# clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
#     pairwise_predictions
# )

# # -------------------------
# # 7. Save to CSV
# # -------------------------
# df_clusters = clusters.as_pandas_dataframe()

# output_path = "../unified_data/cluster_3_deduplicated_clusters.csv"
# df_clusters.to_csv(output_path, index=False)

# print(f"Saved clusters to: {output_path}")
# print(df_clusters.head())



























import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict, Counter
import warnings
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')

class UniversalDeduplicator:
    def __init__(self):
        """
        Universal deduplication pipeline that makes no assumptions about data.
        Automatically analyzes any CSV and performs entity resolution.
        """
        self.df = None
        self.column_types = {}
        self.text_columns = []
        self.numeric_columns = []
        self.id_columns = []
        self.clusters = None
        self.golden_records = None
        
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load any CSV with automatic encoding detection"""
        print(f"📂 Loading {csv_path}...")
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(csv_path, encoding=encoding, dtype=str, low_memory=False)
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"⚠️ Error with {encoding}: {e}")
                continue
        
        if self.df is None:
            # Last resort: read with errors='replace'
            self.df = pd.read_csv(csv_path, encoding='utf-8', dtype=str, errors='replace', low_memory=False)
            print("⚠️ Loaded with errors replaced")
        
        # Clean column names
        self.df.columns = [self._clean_column_name(col) for col in self.df.columns]
        
        print(f"📊 Shape: {self.df.shape}")
        print(f"📋 Columns: {list(self.df.columns)}")
        
        return self.df
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name"""
        if not isinstance(name, str):
            name = str(name)
        name = name.strip().lower()
        name = re.sub(r'[^\w\s]', '_', name)  # Replace special chars with underscore
        name = re.sub(r'\s+', '_', name)  # Replace spaces with underscore
        name = re.sub(r'_+', '_', name)  # Remove multiple underscores
        name = name.strip('_')
        return name if name else 'unnamed_column'
    
    def analyze_data_structure(self) -> Dict[str, Any]:
        """Automatically analyze data structure without assumptions"""
        print("\n🔍 Analyzing data structure...")
        
        analysis = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': {},
            'column_types': {},
            'suggested_id_columns': [],
            'suggested_text_columns': [],
            'suggested_numeric_columns': [],
            'duplicate_rows': self.df.duplicated().sum(),
            'issues': []
        }
        
        for column in self.df.columns:
            col_data = self.df[column]
            
            # Calculate missing values
            missing = col_data.isna().sum()
            missing_pct = missing / len(self.df) * 100
            analysis['missing_values'][column] = f"{missing_pct:.1f}%"
            
            # Determine column type
            col_type = self._infer_column_type(col_data)
            analysis['column_types'][column] = col_type
            
            # Store column type for later use
            self.column_types[column] = col_type
            
            if col_type == 'text':
                self.text_columns.append(column)
                analysis['suggested_text_columns'].append(column)
                
                # Check if column looks like an ID
                if self._looks_like_id(col_data):
                    self.id_columns.append(column)
                    analysis['suggested_id_columns'].append(column)
                    
            elif col_type == 'numeric':
                self.numeric_columns.append(column)
                analysis['suggested_numeric_columns'].append(column)
                
            
            # Detect potential issues
            if missing_pct > 50:
                analysis['issues'].append(f"⚠️ Column '{column}' has {missing_pct:.1f}% missing values")
            
            if col_type == 'text' and self._has_json_strings(col_data):
                analysis['issues'].append(f"⚠️ Column '{column}' contains JSON strings")
        
        # Display analysis
        print(f"📈 Data Analysis Summary:")
        print(f"   • Total records: {analysis['total_rows']}")
        print(f"   • Total columns: {analysis['total_columns']}")
        print(f"   • Duplicate rows: {analysis['duplicate_rows']}")
        print(f"\n   Column Types:")
        for col, col_type in analysis['column_types'].items():
            print(f"     • {col}: {col_type} ({analysis['missing_values'][col]} missing)")
        
        if analysis['suggested_id_columns']:
            print(f"\n   Suggested ID columns: {analysis['suggested_id_columns']}")
        
        if analysis['issues']:
            print(f"\n   ⚠️ Detected Issues:")
            for issue in analysis['issues']:
                print(f"     {issue}")
        
        return analysis
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer column type automatically"""
        # Take a sample for analysis
        sample_size = min(100, len(series))
        if sample_size == 0:
            return 'unknown'
        
        sample = series.dropna().head(sample_size)
        
        if len(sample) == 0:
            return 'unknown'
        
        # Check if it's numeric
        numeric_count = 0
        for val in sample:
            try:
                if pd.notna(val):
                    # Try to convert to number
                    float(str(val).replace(',', '').replace('$', '').strip())
                    numeric_count += 1
            except:
                pass
        
        numeric_ratio = numeric_count / len(sample)
        
        if numeric_ratio > 0.8:
            return 'numeric'
        
        
        # Default to text
        return 'text'
    
    def _looks_like_id(self, series: pd.Series) -> bool:
        """Check if column looks like an ID column"""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Check patterns
        id_patterns = [
            r'^[A-Z]{2,}-\d{3,}',  # AB-123
            r'^\d{5,}',  # 12345
            r'^[A-Z]{2,}\d{3,}',  # AB123
            r'^ITEM-', r'^PROD-', r'^ID-',  # Common prefixes
            r'^[A-Z0-9]{8,}',  # Long alphanumeric
        ]
        
        for pattern in id_patterns:
            matches = sum(1 for val in sample if re.match(pattern, str(val), re.IGNORECASE))
            if matches / len(sample) > 0.5:  # 50% match rate
                return True
        
        # Check uniqueness
        unique_ratio = series.nunique() / len(series.dropna())
        return unique_ratio > 0.95  # Very high uniqueness
    
    def _has_json_strings(self, series: pd.Series) -> bool:
        """Check if column contains JSON strings"""
        sample = series.dropna().head(50)
        json_count = sum(1 for val in sample if isinstance(val, str) and '{' in val and '}' in val)
        return json_count > len(sample) * 0.3
    
    def preprocess_columns(self) -> pd.DataFrame:
        """Preprocess all columns based on their detected type"""
        print("\n🧹 Preprocessing columns...")
        
        df = self.df.copy()
        
        for column in df.columns:
            col_type = self.column_types.get(column, 'text')
            
            if col_type == 'text':
                df[f"{column}_clean"] = df[column].apply(self._clean_text)
                
            elif col_type == 'numeric':
                df[f"{column}_clean"] = df[column].apply(self._extract_number)
                
            
            # Also keep original for reference
            df[f"{column}_original"] = df[column]
        
        print(f"✅ Preprocessed {len(df.columns)} columns")
        return df
    
    def _clean_text(self, value: Any) -> str:
        """Clean text values"""
        if pd.isna(value):
            return ""
        
        value = str(value).strip()
        
        # Remove extra whitespace
        value = re.sub(r'\s+', ' ', value)
        
        # Try to parse JSON if it looks like JSON
        if value.startswith('{') and value.endswith('}'):
            try:
                parsed = json.loads(value.replace("'", '"'))
                # Extract key information from JSON
                if isinstance(parsed, dict):
                    # Join important values
                    important_keys = ['name', 'title', 'value', 'description']
                    extracted = [str(parsed.get(k, '')) for k in important_keys if k in parsed]
                    return ' '.join(extracted).strip()
            except:
                pass
        
        return value.lower()
    
    def _extract_number(self, value: Any) -> float:
        """Extract numeric value from various formats"""
        if pd.isna(value):
            return np.nan
        
        value = str(value).strip()
        
        # Remove common non-numeric characters
        value = re.sub(r'[$,%€£¥]', '', value)
        
        # Handle ranges (e.g., "10-20" -> take average)
        if '-' in value and not value.startswith('-'):
            parts = value.split('-')
            try:
                nums = [float(p.strip()) for p in parts if p.strip()]
                if nums:
                    return sum(nums) / len(nums)
            except:
                pass
        
        # Extract numbers (including decimals and scientific notation)
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
        
        if numbers:
            try:
                return float(numbers[0])
            except:
                return np.nan
        
        return np.nan
    
    def create_blocking_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dynamically create blocking strategies based on data"""
        print("\n🔑 Creating blocking strategies...")
        
        blocking_df = df.copy()
        
        # Strategy 1: Use ID columns if available
        if self.id_columns:
            id_col = self.id_columns[0]  # Use first ID column
            blocking_df['block_id'] = df[f"{id_col}_clean"].apply(
                lambda x: str(x)[:10] if pd.notna(x) else "no_id"
            )
            print(f"   • Created ID-based blocks using '{id_col}'")
        
        # Strategy 2: Use text columns with most information
        if self.text_columns:
            # Find text column with most unique values (most discriminating)
            text_col = max(self.text_columns, 
                          key=lambda col: df[f"{col}_clean"].nunique() if col in df.columns else 0)
            
            blocking_df['block_text'] = df[f"{text_col}_clean"].apply(
                lambda x: '_'.join(str(x).split()[:3])[:30] if pd.notna(x) and str(x) else "no_text"
            )
            print(f"   • Created text-based blocks using '{text_col}'")
        
        
        # Strategy 4: Use numeric ranges if we have numeric columns
        if self.numeric_columns:
            num_col = self.numeric_columns[0]
            if num_col in df.columns and df[f"{num_col}_clean"].notna().any():
                # Create numeric ranges
                series = pd.to_numeric(df[f"{num_col}_clean"], errors='coerce')
                if series.notna().any():
                    q25, q75 = series.quantile([0.25, 0.75])
                    blocking_df['block_num'] = pd.cut(
                        series,
                        bins=[-np.inf, q25, q75, np.inf],
                        labels=['low', 'medium', 'high']
                    ).astype(str)
                    print(f"   • Created numeric blocks using '{num_col}'")
        
        print(f"✅ Created blocking strategies")
        return blocking_df
    
    def find_duplicate_candidates(self, df: pd.DataFrame) -> List[Dict]:
        """Find potential duplicate pairs using blocking"""
        print("\n🔍 Finding duplicate candidates...")
        
        candidate_pairs = []
        processed_pairs = set()
        
        # Try each blocking strategy
        block_columns = [col for col in df.columns if col.startswith('block_')]
        
        if not block_columns:
            print("⚠️ No blocking strategies created, comparing all pairs (slow!)")
            # Fallback: compare all pairs (only for small datasets)
            if len(df) > 1000:
                print("❌ Dataset too large for all-pairs comparison")
                return []
            
            indices = df.index.tolist()
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    candidate_pairs.append({
                        'record1': indices[i],
                        'record2': indices[j],
                        'block_key': 'all_pairs'
                    })
            
            return candidate_pairs
        
        # Compare within blocks
        for block_col in block_columns:
            for block_value, block_df in df.groupby(block_col):
                if len(block_df) > 1:
                    indices = block_df.index.tolist()
                    
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            idx1, idx2 = indices[i], indices[j]
                            pair_key = tuple(sorted([idx1, idx2]))
                            
                            if pair_key not in processed_pairs:
                                candidate_pairs.append({
                                    'record1': idx1,
                                    'record2': idx2,
                                    'block_key': block_col
                                })
                                processed_pairs.add(pair_key)
        
        print(f"✅ Found {len(candidate_pairs)} candidate pairs for comparison")
        return candidate_pairs
    def calculate_record_similarity(self, df: pd.DataFrame, idx1: int, idx2: int) -> float:
        """Calculate similarity between two records"""
        row1 = df.loc[idx1]
        row2 = df.loc[idx2]
        
        similarities = []
        weights = []
        
        # Compare each column type appropriately
        for column in self.df.columns:
            clean_col = f"{column}_clean"
            
            if clean_col not in row1 or clean_col not in row2:
                continue
            
            val1 = row1[clean_col]
            val2 = row2[clean_col]
            col_type = self.column_types.get(column, 'text')
            
            if pd.isna(val1) or pd.isna(val2):
                # Missing values - lower weight
                continue
            
            if col_type == 'text':
                sim = self._text_similarity(str(val1), str(val2))
                weights.append(0.5 if column in self.text_columns else 0.25)  # Increased
                
            elif col_type == 'numeric':
                # Normalize numeric values for comparison
                try:
                    num1 = float(val1)
                    num2 = float(val2)
                    
                    if num1 == 0 and num2 == 0:
                        sim = 1.0
                    elif max(abs(num1), abs(num2)) > 0:
                        diff = abs(num1 - num2) / max(abs(num1), abs(num2))
                        sim = max(0, 1 - diff)
                    else:
                        sim = 0.0
                except:
                    sim = 0.0
                weights.append(0.35)
        
                
            else:
                sim = 0.0
                weights.append(0.1)
            
            similarities.append(sim)
        
        # Weighted average
        if weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight
                return weighted_sim
        
        return 0.0
    import re

    def clean_string_for_tokens(self,text):
        """
        Clean a string for tokenization, keeping only alphanumeric characters.
        
        Args:
            text (str): Input string to clean
            
        Returns:
            str: Cleaned string with only a-z, 0-9, and spaces
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Step 1: Handle common patterns that should be preserved as separate tokens
        # Replace common separators with spaces before removing special chars
        text = re.sub(r'[-_./]', ' ', text)  # Replace separators with spaces
        
        # Step 2: Remove all characters that are NOT a-z, 0-9, or whitespace
        # This removes punctuation, brackets, quotes, symbols, etc.
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Step 3: Normalize whitespace - multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Step 4: Remove leading/trailing spaces
        text = text.strip()
        
        return text

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity with multiple methods"""
        if text1 == text2:
            return 1.0
        
        if not text1 or not text2:
            return 0.0
        
        # Method 1: Jaro-Winkler (good for names, addresses)
        jw_sim = self._jaro_winkler_similarity(text1, text2)
        text1 = self.clean_string_for_tokens(text1)
        text2 = self.clean_string_for_tokens(text2)
        # Method 2: Token set similarity (good for descriptions)
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if tokens1 and tokens2:
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            token_sim = intersection / union if union > 0 else 0.0
        else:
            token_sim = 0.0

        # Use the maximum of both methods
        return max(jw_sim, token_sim)
    
    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Jaro-Winkler similarity implementation"""
        s1, s2 = s1.lower(), s2.lower()
        
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1
        
        matches1 = [False] * len1
        matches2 = [False] * len2
        matches = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if not matches2[j] and s1[i] == s2[j]:
                    matches1[i] = True
                    matches2[j] = True
                    matches += 1
                    break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = transpositions = 0
        for i in range(len1):
            if matches1[i]:
                while not matches2[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
        
        transpositions //= 2
        
        # Calculate Jaro similarity
        jaro = ((matches / len1) + (matches / len2) + ((matches - transpositions) / matches)) / 3
        
        # Winkler prefix bonus
        prefix = 0
        for i in range(min(4, len1, len2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        winkler = jaro + (prefix * 0.1 * (1 - jaro))
        return min(winkler, 1.0)
    
    def cluster_duplicates(self, df: pd.DataFrame, candidate_pairs: List[Dict]) -> pd.DataFrame:
        """Cluster duplicate records"""
        print("\n🎯 Clustering duplicates...")
        
        # Filter pairs by similarity threshold
        high_similarity_pairs = []
        
        for pair in candidate_pairs:
            similarity = self.calculate_record_similarity(df, pair['record1'], pair['record2'])
            if similarity > 0.8:  # High threshold for clustering
                high_similarity_pairs.append((pair['record1'], pair['record2'], similarity))
        
        # Build adjacency graph
        from collections import defaultdict
        adj = defaultdict(set)
        
        for idx1, idx2, sim in high_similarity_pairs:
            adj[idx1].add(idx2)
            adj[idx2].add(idx1)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for node in adj.keys():
            if node not in visited:
                # BFS to find connected component
                cluster = set()
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        stack.extend(adj[current] - visited)
                
                if len(cluster) > 1:  # Only keep clusters with duplicates
                    clusters.append(list(cluster))
        
        print(f"✅ Found {len(clusters)} duplicate clusters")
        
        # Assign cluster IDs
        clustered_df = df.copy()
        clustered_df['cluster_id'] = -1
        
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                clustered_df.at[idx, 'cluster_id'] = cluster_id
        
        # Assign IDs to singletons (non-duplicates)
        max_cluster_id = len(clusters)
        for idx in clustered_df.index:
            if clustered_df.at[idx, 'cluster_id'] == -1:
                clustered_df.at[idx, 'cluster_id'] = max_cluster_id
                max_cluster_id += 1
        
        self.clusters = clusters
        return clustered_df
    
    def create_golden_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create golden records from clusters"""
        print("\n🏆 Creating golden records...")
        
        golden_records = []
        
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            if len(cluster_df) == 0:
                continue
            
            golden_record = {'cluster_id': cluster_id, 'source_records': len(cluster_df)}
            
            # For each original column, create the best value
            for column in self.df.columns:
                clean_col = f"{column}_clean"
                original_col = f"{column}_original"
                
                if clean_col in cluster_df.columns:
                    values = cluster_df[clean_col].dropna().tolist()
                    
                    if values:
                        col_type = self.column_types.get(column, 'text')
                        
                        if col_type == 'numeric':
                            # For numeric: use median
                            try:
                                numeric_vals = [float(v) for v in values if self._is_convertible_to_float(v)]
                                if numeric_vals:
                                    golden_record[column] = np.median(numeric_vals)
                            except:
                                # Use most common if conversion fails
                                golden_record[column] = Counter(values).most_common(1)[0][0]
                        
                        
                        else:  # text
                            # For text: use most complete (longest) non-empty value
                            non_empty = [str(v).strip() for v in values if str(v).strip()]
                            if non_empty:
                                # Prefer values that look more complete
                                complete_vals = [v for v in non_empty if len(v) > 3]
                                if complete_vals:
                                    golden_record[column] = max(complete_vals, key=len)
                                else:
                                    golden_record[column] = non_empty[0]
            
            golden_records.append(golden_record)
        
        # Create DataFrame
        golden_df = pd.DataFrame(golden_records)
        
        # Generate unique IDs if no ID column exists
        if not any(col in golden_df.columns for col in self.id_columns):
            golden_df['generated_id'] = [f'REC-{i:04d}' for i in range(len(golden_df))]
        
        print(f"✅ Created {len(golden_df)} golden records")
        print(f"📊 Reduction: {len(self.df)} → {len(golden_df)} records")
        
        self.golden_records = golden_df
        return golden_df
    
    def _is_convertible_to_float(self, value: Any) -> bool:
        """Check if value can be converted to float"""
        try:
            float(value)
            return True
        except:
            return False
    
    def save_results(self, golden_df: pd.DataFrame, output_path: str = 'unified_data/deduplicated_output.csv'):
        """Save results to CSV"""
        print(f"\n💾 Saving results to {output_path}...")
        
        # Format numeric columns
        for column in golden_df.columns:
            if column in self.numeric_columns:
                # Format to 2 decimal places
                golden_df[column] = golden_df[column].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) and self._is_convertible_to_float(x) else x
                )
        
        # Remove helper columns
        columns_to_keep = [col for col in golden_df.columns 
                          if not col.endswith('_clean') and not col.endswith('_original')]
        
        golden_df = golden_df[columns_to_keep]
        
        # Save
        golden_df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(golden_df)} records")
        
        # Also save detailed version
        detailed_path = output_path.replace('.csv', '_detailed.csv')
        golden_df.to_csv(detailed_path, index=False)
        print(f"✅ Detailed version saved to {detailed_path}")
        
        return output_path
    
    def run_pipeline(self, csv_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Run the complete universal deduplication pipeline"""
        print("=" * 80)
        print("🔄 UNIVERSAL DEDUPLICATION PIPELINE")
        print("=" * 80)
        
        # Step 1: Load any CSV
        self.load_csv(csv_path)
        
        # Step 2: Analyze structure
        analysis = self.analyze_data_structure()
        
        # Step 3: Preprocess
        processed_df = self.preprocess_columns()
        
        # Step 4: Create blocking strategies
        blocked_df = self.create_blocking_strategies(processed_df)
        
        # Step 5: Find candidate pairs
        candidate_pairs = self.find_duplicate_candidates(blocked_df)
        
        if not candidate_pairs:
            print("⚠️ No duplicate candidates found. Returning original data.")
            return self.df, analysis
        
        # Step 6: Cluster duplicates
        clustered_df = self.cluster_duplicates(blocked_df, candidate_pairs)
        
        # Step 7: Create golden records
        golden_df = self.create_golden_records(clustered_df)
        
        # Step 8: Save results
        output_path = self.save_results(golden_df)
        
        # Step 9: Create summary
        summary = {
            'original_records': len(self.df),
            'golden_records': len(golden_df),
            'reduction_percentage': f"{(len(self.df)-len(golden_df))/len(self.df)*100:.1f}%",
            'duplicate_clusters': len(self.clusters) if self.clusters else 0,
            'output_file': output_path
        }
        
        print("\n" + "=" * 80)
        print("📊 PIPELINE SUMMARY")
        print("=" * 80)
        for key, value in summary.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📋 First 5 golden records:")
        print(golden_df.head().to_string())
        
        return golden_df, summary

def deduplicate():
        deduplicator = UniversalDeduplicator()
    
        # Run on any CSV file
        csv_file = './unified_data/cluster_2_unified.csv'  #input("Enter CSV file path: ").strip() or
        
        try:
            golden_df, summary = deduplicator.run_pipeline(csv_file)
            
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📁 Output saved to: {summary['output_file']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
# Usage example
if __name__ == "__main__":
    # Initialize the universal deduplicator
    deduplicator = UniversalDeduplicator()
    
    # Run on any CSV file
    csv_file = '../unified_data/cluster_2_unified.csv'  #input("Enter CSV file path: ").strip() or
    
    try:
        golden_df, summary = deduplicator.run_pipeline(csv_file)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Output saved to: {summary['output_file']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()