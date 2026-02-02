import numpy as np
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sentence_transformers import SentenceTransformer
class SchemaSimilarityCalculator:
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize schema similarity calculator.
        
        Args:
            weights: Optional weights for each criterion. If None, uses defaults.
        """
        # Default weights for each criterion
        self.default_weights = {
            'file_name': 0.15,      # Criterion 1
            'schema_length': 0.15,   # Criterion 2  
            'column_names': 0.30,    # Criterion 3
            'data_types': 0.20,      # Criterion 4
            'value_patterns': 0.20   # Criterion 5
        }
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.weights = weights if weights else self.default_weights
        self.file_names = []
        self.similarity_matrices = {}
        self.final_similarity_matrix = None
        self.column_mappings = {}
    
    def load_features(self, features_file): #: str = 'all_features.json'
        """
        Load features from JSON file.
        """
        # with open(features_file, 'r') as f:
        #     self.features = json.load(f)
        self.features = features_file
        
        # Get list of files
        self.file_names = list(self.features['file_names'].keys())
        print(f"✅ Loaded features for {len(self.file_names)} files")
    
    def calculate_all_pairwise_similarities(self):
        """
        Calculate pairwise similarities for all file pairs across all criteria.
        """
        print("\n" + "=" * 80)
        print("CALCULATING PAIRWISE SIMILARITIES")
        print("=" * 80)
        
        n = len(self.file_names)
        
        # Initialize similarity matrices for each criterion
        self.similarity_matrices = {
            'file_name': np.zeros((n, n)),
            'schema_length': np.zeros((n, n)),
            'column_names': np.zeros((n, n)),
            'data_types': np.zeros((n, n)),
            'value_patterns': np.zeros((n, n)),
            'total': np.zeros((n, n))
        }
        
        # Store detailed results
        self.detailed_results = {}
        
        # Calculate similarities for all pairs
        for i in range(n):
            file1 = self.file_names[i]
            
            for j in range(i, n):
                file2 = self.file_names[j]
                
                if i == j:
                    # Same file - perfect similarity
                    for criterion in self.similarity_matrices.keys():
                        self.similarity_matrices[criterion][i, j] = 1.0
                    continue
                
                # Calculate similarity for each criterion
                results = self._calculate_pair_similarity(file1, file2)
                
                # Store in matrices
                for criterion, score in results['scores'].items():
                    self.similarity_matrices[criterion][i, j] = score
                    self.similarity_matrices[criterion][j, i] = score  # Symmetric
                
                # Store detailed results
                self.detailed_results[(file1, file2)] = results
                self.detailed_results[(file2, file1)] = results  # Both directions
        
        print("✅ Pairwise similarity calculation complete!")
    
    def _calculate_pair_similarity(self, file1: str, file2: str) -> Dict[str, Any]:
        """
        Calculate similarity between two files for all criteria.
        """
        results = {
            'scores': {},
            'details': {},
            'column_mapping': []
        }
        
        # 1. File Name Similarity
        file_name_score = self._calculate_file_name_similarity(file1, file2)
        results['scores']['file_name'] = file_name_score
        results['details']['file_name'] = {
            'score': file_name_score
        }
        
        # 2. Schema Length Similarity
        schema_length_score = self._calculate_schema_length_similarity(file1, file2)
        results['scores']['schema_length'] = schema_length_score
        results['details']['schema_length'] = {
            'score': schema_length_score
        }
        
        # 3. Column Name Similarity
        column_name_score, column_mapping = self._calculate_column_name_similarity(file1, file2)
        results['scores']['column_names'] = column_name_score
        results['details']['column_names'] = {
            'score': column_name_score,
            'mapping_count': len(column_mapping)
        }
        
        # Store column mapping for use in other criteria
        results['column_mapping'] = column_mapping
        
        # 4. Data Type Similarity (uses column mapping)
        data_type_score = self._calculate_data_type_similarity(file1, file2, column_mapping)
        results['scores']['data_types'] = data_type_score
        results['details']['data_types'] = {
            'score': data_type_score,
            'uses_mapping': True
        }
        
        # 5. Value Pattern Similarity (uses column mapping)
        pattern_score = self._calculate_value_pattern_similarity(file1, file2, column_mapping)
        results['scores']['value_patterns'] = pattern_score
        results['details']['value_patterns'] = {
            'score': pattern_score,
            'uses_mapping': True
        }
        
        # Calculate weighted total score
        total_score = sum(
            self.weights[criterion] * results['scores'][criterion]
            for criterion in self.weights.keys()
        )
        results['scores']['total'] = total_score
        
        return results
    
    def _calculate_file_name_similarity(self, file1: str, file2: str) -> float:
        """Calculate file name similarity."""
        if file1 not in self.features['file_names'] or file2 not in self.features['file_names']:
            return 0.0
        
        # Get normalized names (without extensions)
        name1 = file1.rsplit('.', 1)[0].lower()
        name2 = file2.rsplit('.', 1)[0].lower()
        
        # Simple token-based similarity
        tokens1 = set(re.split(r'[_\-. ]+', name1))
        tokens2 = set(re.split(r'[_\-. ]+', name2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_schema_length_similarity(self, file1: str, file2: str) -> float:
        """Calculate schema length similarity."""
        if (file1 not in self.features['schema_lengths'] or 
            file2 not in self.features['schema_lengths']):
            return 0.0
        
        features1 = self.features['schema_lengths'][file1]
        features2 = self.features['schema_lengths'][file2]
        
        count1 = features1.get('column_count', 0)
        count2 = features2.get('column_count', 0)
        
        if count1 == 0 and count2 == 0:
            return 1.0
        if count1 == 0 or count2 == 0:
            return 0.0
        
        # Normalized difference
        similarity = 1 - abs(count1 - count2) / max(count1, count2)
        
        # Also consider type ratios
        type_features = ['string_ratio', 'numeric_ratio', 'datetime_ratio']
        type_similarities = []
        
        for feat in type_features:
            val1 = features1.get(feat, 0)
            val2 = features2.get(feat, 0)
            type_sim = 1 - abs(val1 - val2)
            type_similarities.append(type_sim)
        
        avg_type_similarity = np.mean(type_similarities) if type_similarities else 0.5
        
        # Combined score
        return 0.7 * similarity + 0.3 * avg_type_similarity
    
    def _calculate_column_name_similarity(self, file1: str, file2: str) -> Tuple[float, List]:
        """Calculate column name similarity and return mapping."""
        if (file1 not in self.features['column_names'] or 
            file2 not in self.features['column_names']):
            return 0.0, []
        
        features1 = self.features['column_names'][file1]
        features2 = self.features['column_names'][file2]
        
        cols1 = features1.get('column_names', [])
        cols2 = features2.get('column_names', [])
        
        if not cols1 or not cols2:
            return 0.0, []
        
        # Convert to lowercase for comparison
        cols1_lower = [c.lower() for c in cols1]
        cols2_lower = [c.lower() for c in cols2]
        
        # Find best column mapping using greedy algorithm
        column_mapping = []
        matched_cols2 = set()
        
        for i, col1 in enumerate(cols1_lower):
            best_match_idx = -1
            best_score = 0
            
            for j, col2 in enumerate(cols2_lower):
                if j in matched_cols2:
                    continue
                
                # Calculate similarity
                score = self._calculate_column_similarity(col1, col2, file1, file2)
                
                if score > best_score and score > 0.3:  # Threshold
                    best_score = score
                    best_match_idx = j
            
            if best_match_idx != -1:
                column_mapping.append((cols1[i], cols2[best_match_idx], best_score))
                matched_cols2.add(best_match_idx)
        
        # Calculate overall similarity
        if column_mapping:
            avg_score = sum(score for _, _, score in column_mapping) / len(column_mapping)
            coverage = len(column_mapping) / max(len(cols1), len(cols2))
            similarity = avg_score * coverage
        else:
            similarity = 0.0
        
        return similarity, column_mapping
    
    def _calculate_column_similarity(self, col1: str, col2: str , file1: str = None, file2: str = None) -> float:
            name_embeddings = self.semantic_model.encode([col1, col2])
            name_similarity = cosine_similarity([name_embeddings[0]], [name_embeddings[1]])[0][0]
            # Ensure between 0 and 1
            name_similarity = max(0.0, min(1.0, name_similarity))


            file1_descriptions = self.features['description'].get(file1, {})
            file2_descriptions = self.features['description'].get(file2, {})
            
            # Get descriptions for these specific columns
            description1 = file1_descriptions.get(col1, '')
            description2 = file2_descriptions.get(col2, '')

            description_embeddings = self.semantic_model.encode([description1, description2])
            description_similarity = cosine_similarity([description_embeddings[0]], [description_embeddings[1]])[0][0]
            # Ensure between 0 and 1
            description_similarity = max(0.0, min(1.0, description_similarity))


            w_name = 0.4
            w_description = 0.6

            final_similarity = w_name * name_similarity + w_description * description_similarity
            
            return final_similarity
        # """Calculate similarity between two column names."""
        # if col1 == col2:
        #     return 1.0
        
        # # Edit distance (normalized)
        # from difflib import SequenceMatcher
        # edit_similarity = SequenceMatcher(None, col1, col2).ratio()
        
        # # Token overlap
        # tokens1 = set(re.split(r'[_\-. ]+', col1))
        # tokens2 = set(re.split(r'[_\-. ]+', col2))
        
        # if tokens1 and tokens2:
        #     token_overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        # else:
        #     token_overlap = 0.0
        
        # # Combine scores
        # return 0.7 * edit_similarity + 0.3 * token_overlap

    
    def _calculate_data_type_similarity(self, file1: str, file2: str, 
                                       column_mapping: List) -> float:
        """Calculate data type similarity using column mapping."""
        if not column_mapping:
            return 0.5  # Neutral score if no mapping
        
        if (file1 not in self.features['data_types'] or 
            file2 not in self.features['data_types']):
            return 0.5
        
        features1 = self.features['data_types'][file1]
        features2 = self.features['data_types'][file2]
        
        type_compatibilities = []
        
        for col1, col2, _ in column_mapping:
            type1 = features1['column_types'].get(col1, 'unknown')
            type2 = features2['column_types'].get(col2, 'unknown')
            
            if type1 == type2:
                compat = 1.0
            elif (type1 in ['integer', 'float'] and type2 in ['integer', 'float']):
                compat = 0.8
            elif (type1 == 'string' and type2 != 'string') or (type2 == 'string' and type1 != 'string'):
                compat = 0.3
            else:
                compat = 0.5
            
            type_compatibilities.append(compat)
        
        return np.mean(type_compatibilities) if type_compatibilities else 0.5
    
    def _calculate_value_pattern_similarity(self, file1: str, file2: str,
                                           column_mapping: List) -> float:
        """Calculate value pattern similarity using column mapping."""
        if not column_mapping:
            # Compare overall pattern summaries
            return self._calculate_overall_pattern_similarity(file1, file2)
        
        if (file1 not in self.features['value_patterns'] or 
            file2 not in self.features['value_patterns']):
            return 0.5
        
        features1 = self.features['value_patterns'][file1]
        features2 = self.features['value_patterns'][file2]
        
        pattern_compatibilities = []
        
        for col1, col2, _ in column_mapping:
            patterns1 = features1['column_patterns'].get(col1, {})
            patterns2 = features2['column_patterns'].get(col2, {})
            
            if not patterns1 or not patterns2:
                pattern_compatibilities.append(0.5)
                continue
            
            # Compare multiple pattern features
            feature_similarities = []
            
            # Character distribution
            char_dist1 = patterns1.get('char_distribution', {})
            char_dist2 = patterns2.get('char_distribution', {})
            
            if char_dist1 and char_dist2:
                char_sims = []
                for char_type in set(char_dist1.keys()).union(set(char_dist2.keys())):
                    val1 = char_dist1.get(char_type, 0)
                    val2 = char_dist2.get(char_type, 0)
                    char_sims.append(1 - abs(val1 - val2))
                
                if char_sims:
                    feature_similarities.append(np.mean(char_sims))
            
            # Digit ratio
            digit1 = patterns1.get('digit_ratio', 0.5)
            digit2 = patterns2.get('digit_ratio', 0.5)
            feature_similarities.append(1 - abs(digit1 - digit2))
            
            # Unique ratio
            unique1 = patterns1.get('unique_ratio', 0.5)
            unique2 = patterns2.get('unique_ratio', 0.5)
            feature_similarities.append(1 - abs(unique1 - unique2))
            
            if feature_similarities:
                pattern_compatibilities.append(np.mean(feature_similarities))
        
        return np.mean(pattern_compatibilities) if pattern_compatibilities else 0.5
    
    def _calculate_overall_pattern_similarity(self, file1: str, file2: str) -> float:
        """Calculate overall pattern similarity when no column mapping exists."""
        if (file1 not in self.features['value_patterns'] or 
            file2 not in self.features['value_patterns']):
            return 0.5
        
        features1 = self.features['value_patterns'][file1]
        features2 = self.features['value_patterns'][file2]
        
        summary1 = features1.get('pattern_summary', {})
        summary2 = features2.get('pattern_summary', {})
        
        similarities = []
        
        # Compare average digit ratio
        digit1 = summary1.get('avg_digit_ratio', 0.5)
        digit2 = summary2.get('avg_digit_ratio', 0.5)
        similarities.append(1 - abs(digit1 - digit2))
        
        # Compare average unique ratio
        unique1 = summary1.get('avg_unique_ratio', 0.5)
        unique2 = summary2.get('avg_unique_ratio', 0.5)
        similarities.append(1 - abs(unique1 - unique2))
        
        return np.mean(similarities) if similarities else 0.5
    
    def calculate_final_similarity_matrix(self) -> np.ndarray:
        """
        Calculate final weighted similarity matrix.
        """
        n = len(self.file_names)
        self.final_similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    self.final_similarity_matrix[i, j] = 1.0
                    continue
                
                total_score = 0
                for criterion, weight in self.weights.items():
                    total_score += weight * self.similarity_matrices[criterion][i, j]
                
                self.final_similarity_matrix[i, j] = total_score
                self.final_similarity_matrix[j, i] = total_score
        
        print("✅ Final similarity matrix calculated!")
        return self.final_similarity_matrix
    
    def perform_hierarchical_clustering(self, threshold: float = 0.5) -> List[List[str]]:
        """
        Perform hierarchical clustering on the similarity matrix.
        
        Args:
            threshold: Cutoff threshold for forming clusters
            
        Returns:
            List of clusters, each cluster is a list of filenames
        """
        if self.final_similarity_matrix is None:
            self.calculate_final_similarity_matrix()
        
        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - self.final_similarity_matrix
        
        # Perform hierarchical clustering
        linkage_matrix = sch.linkage(distance_matrix, method='average')
        
        # Form clusters at threshold
        clusters = sch.fcluster(linkage_matrix, t=1-threshold, criterion='distance')
        
        # Group files by cluster
        cluster_dict = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_dict[cluster_id].append(self.file_names[idx])
        
        # Convert to list and sort by cluster size
        result_clusters = list(cluster_dict.values())
        result_clusters.sort(key=len, reverse=True)
        
        print(f"✅ Hierarchical clustering complete!")
        print(f"📊 Found {len(result_clusters)} clusters with threshold {threshold}")
        
        # Store for later use
        self.clusters = result_clusters
        self.linkage_matrix = linkage_matrix
        
        return result_clusters
    
    def analyze_clusters(self, clusters: List[List[str]]) -> Dict[str, Any]:
        """
        Analyze the quality and characteristics of clusters.
        
        Returns:
            Dictionary with cluster analysis results
        """
        analysis = {
            'cluster_stats': [],
            'overall_quality': 0.0,
            'intra_cluster_similarities': [],
            'inter_cluster_similarities': []
        }
        
        # Calculate intra-cluster similarities
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue
            
            # Get indices of files in this cluster
            indices = [self.file_names.index(f) for f in cluster]
            
            # Calculate average similarity within cluster
            intra_similarities = []
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    sim = self.final_similarity_matrix[indices[i], indices[j]]
                    intra_similarities.append(sim)
            
            avg_intra_sim = np.mean(intra_similarities) if intra_similarities else 1.0
            
            cluster_info = {
                'cluster_id': cluster_idx + 1,
                'size': len(cluster),
                'files': cluster,
                'avg_intra_similarity': avg_intra_sim,
                'cohesion': 'High' if avg_intra_sim > 0.7 else 
                           'Medium' if avg_intra_sim > 0.5 else 'Low'
            }
            
            analysis['cluster_stats'].append(cluster_info)
            analysis['intra_cluster_similarities'].extend(intra_similarities)
        
        # Calculate inter-cluster similarities (for clusters with >1 file)
        multi_file_clusters = [c for c in clusters if len(c) > 1]
        
        if len(multi_file_clusters) > 1:
            inter_similarities = []
            for i in range(len(multi_file_clusters)):
                for j in range(i+1, len(multi_file_clusters)):
                    # Calculate average similarity between clusters
                    cluster_i_files = multi_file_clusters[i]
                    cluster_j_files = multi_file_clusters[j]
                    
                    similarities = []
                    for file_i in cluster_i_files:
                        idx_i = self.file_names.index(file_i)
                        for file_j in cluster_j_files:
                            idx_j = self.file_names.index(file_j)
                            similarities.append(self.final_similarity_matrix[idx_i, idx_j])
                    
                    if similarities:
                        inter_similarities.extend(similarities)
            
            if inter_similarities:
                analysis['avg_inter_similarity'] = np.mean(inter_similarities)
                analysis['separation'] = 'Good' if analysis['avg_inter_similarity'] < 0.3 else 'Poor'
            else:
                analysis['avg_inter_similarity'] = 0.0
                analysis['separation'] = 'N/A'
        
        # Calculate overall quality (higher intra, lower inter = better)
        if analysis['intra_cluster_similarities']:
            avg_intra = np.mean(analysis['intra_cluster_similarities'])
            avg_inter = analysis.get('avg_inter_similarity', 0.0)
            
            # Quality metric: intra similarity - inter similarity
            analysis['overall_quality'] = max(0, avg_intra - avg_inter)
            analysis['quality_rating'] = 'Good' if analysis['overall_quality'] > 0.4 else \
                                        'Fair' if analysis['overall_quality'] > 0.2 else 'Poor'
        
        return analysis
    
    def visualize_results(self, output_dir: str = 'results'):
        """
        Create visualizations of the clustering results.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Heatmap of similarity matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.final_similarity_matrix, 
                   xticklabels=self.file_names,
                   yticklabels=self.file_names,
                   cmap='RdYlGn',
                   vmin=0, vmax=1,
                   square=True)
        plt.title('Schema Similarity Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Dendrogram
        if hasattr(self, 'linkage_matrix'):
            plt.figure(figsize=(15, 8))
            sch.dendrogram(self.linkage_matrix,
                          labels=self.file_names,
                          orientation='top',
                          leaf_rotation=45,
                          leaf_font_size=10)
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Files')
            plt.ylabel('Distance (1 - Similarity)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/dendrogram.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Cluster visualization
        if hasattr(self, 'clusters'):
            plt.figure(figsize=(12, 6))
            
            # Create color map for clusters
            colors = plt.cm.tab20(np.linspace(0, 1, len(self.clusters)))
            
            for cluster_idx, cluster in enumerate(self.clusters):
                # Calculate average similarity for cluster
                indices = [self.file_names.index(f) for f in cluster]
                if len(indices) > 1:
                    intra_sims = []
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            intra_sims.append(self.final_similarity_matrix[indices[i], indices[j]])
                    avg_sim = np.mean(intra_sims) if intra_sims else 1.0
                else:
                    avg_sim = 1.0
                
                plt.barh(cluster_idx, len(cluster), 
                        color=colors[cluster_idx], 
                        alpha=0.7, 
                        label=f'Cluster {cluster_idx+1}')
                plt.text(len(cluster) + 0.1, cluster_idx, 
                        f'({len(cluster)} files, sim={avg_sim:.2f})',
                        va='center')
            
            plt.yticks(range(len(self.clusters)), [f'Cluster {i+1}' for i in range(len(self.clusters))])
            plt.xlabel('Number of Files')
            plt.title('Cluster Sizes and Average Similarities')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cluster_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizations saved to {output_dir}/")
    
    def save_results(self, output_file: str = 'clustering_results.json'):
        """
        Save all results to JSON file.
        """
        results = {
            'file_names': self.file_names,
            'weights': self.weights,
            'similarity_matrices': {
                criterion: matrix.tolist() 
                for criterion, matrix in self.similarity_matrices.items()
            },
            'final_similarity_matrix': self.final_similarity_matrix.tolist() if self.final_similarity_matrix is not None else None,
            'clusters': self.clusters if hasattr(self, 'clusters') else [],
            'detailed_results': self._serialize_detailed_results()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {output_file}")
    
    def _serialize_detailed_results(self) -> Dict:
        """Convert detailed results to serializable format."""
        serialized = {}
        for (file1, file2), results in self.detailed_results.items():
            key = f"{file1}||{file2}"
            serialized[key] = results
        return serialized
























    def find_optimal_threshold(self, min_threshold: float = 0.1, max_threshold: float = 0.9, 
                            step: float = 0.05) -> Dict[str, Any]:
        """
        Find optimal clustering threshold using multiple metrics.
        
        Args:
            min_threshold: Minimum threshold to test
            max_threshold: Maximum threshold to test  
            step: Step size between thresholds
            expected_clusters: Expected number of clusters (if known)
            
        Returns:
            Dictionary with optimal threshold and analysis
        """
        if self.final_similarity_matrix is None:
            self.calculate_final_similarity_matrix()
        
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        results = []
        
        print("\n" + "=" * 80)
        print("FINDING OPTIMAL THRESHOLD")
        print("=" * 80)
        
        for threshold in thresholds:
            # Perform clustering at this threshold
            clusters = self.perform_hierarchical_clustering(threshold=threshold)
            
            # Skip if all files are separate clusters
            if len(clusters) == len(self.file_names):
                continue
                
            # Analyze clusters
            analysis = self.analyze_clusters(clusters)
            
            # Calculate silhouette-like score
            silhouette_score = self._calculate_silhouette_score(clusters)
            
            # Calculate quality score
            avg_intra = np.mean(analysis['intra_cluster_similarities']) if analysis['intra_cluster_similarities'] else 0
            avg_inter = analysis.get('avg_inter_similarity', 0)
            quality_score = avg_intra - avg_inter
            
            # Calculate cluster size balance
            cluster_sizes = [len(c) for c in clusters]
            size_balance = 1 - np.std(cluster_sizes) / np.mean(cluster_sizes) if len(cluster_sizes) > 1 else 0
            
            # Combine scores
            combined_score = (
                0.4 * quality_score + 
                0.3 * silhouette_score + 
                0.2 * size_balance +
                0.1 * (1 / len(clusters))  # Prefer fewer clusters
            )
            
            # # Penalize if expected_clusters is specified and doesn't match
            # if expected_clusters is not None:
            #     if len(clusters) != expected_clusters:
            #         combined_score *= 0.7  # Penalize mismatched cluster count
            
            result = {
                'threshold': threshold,
                'num_clusters': len(clusters),
                'avg_intra_similarity': avg_intra,
                'avg_inter_similarity': avg_inter,
                'quality_score': quality_score,
                'silhouette_score': silhouette_score,
                'size_balance': size_balance,
                'combined_score': combined_score,
                'clusters': clusters
            }
            results.append(result)
            
            print(f"  Threshold {threshold:.2f}: {len(clusters)} clusters | "
                f"Quality: {quality_score:.3f} | Silhouette: {silhouette_score:.3f}")
        
        # if not results:
        #     return {'optimal_threshold': 0.5, 'error': 'No valid thresholds found'}
        if not results:
            # Return a valid structure with default threshold
            print("⚠️  No valid thresholds found, using default threshold 0.5")
            clusters = self.perform_hierarchical_clustering(threshold=0.5)
            analysis = self.analyze_clusters(clusters)
            
            return {
                'optimal_threshold': 0.5,
                'optimal_clusters': clusters,
                'selection_method': 'default',
                'all_results': [{
                    'threshold': 0.5,
                    'num_clusters': len(clusters),
                    'avg_intra_similarity': np.mean(analysis['intra_cluster_similarities']) if analysis['intra_cluster_similarities'] else 0,
                    'avg_inter_similarity': analysis.get('avg_inter_similarity', 0),
                    'quality_score': 0,
                    'silhouette_score': 0,
                    'size_balance': 1.0,
                    'combined_score': 0,
                    'clusters': clusters
                }],
                'best_result': {
                    'threshold': 0.5,
                    'num_clusters': len(clusters),
                    'clusters': clusters
                },
                'elbow_threshold': 0.5
            }
        
        # Find best result by combined score
        best_result = max(results, key=lambda x: x['combined_score'])
        
        # Also find elbow point in cluster count vs threshold
        elbow_threshold = self._find_elbow_point(results)
        
        # Final decision: prefer elbow point if quality is good, otherwise use combined score
        elbow_result = next((r for r in results if abs(r['threshold'] - elbow_threshold) < step/2), None)
        
        if elbow_result and elbow_result['quality_score'] > 0.2:
            final_result = elbow_result
            selection_method = 'elbow_method'
        else:
            final_result = best_result
            selection_method = 'combined_score'
        
        print(f"\n✅ Optimal threshold: {final_result['threshold']:.2f} ({selection_method})")
        print(f"   Clusters: {final_result['num_clusters']}")
        print(f"   Quality score: {final_result['quality_score']:.3f}")
        print(f"   Silhouette score: {final_result['silhouette_score']:.3f}")
        
        return {
            'optimal_threshold': final_result['threshold'],
            'optimal_clusters': final_result['clusters'],
            'selection_method': selection_method,
            'all_results': results,
            'best_result': final_result,
            'elbow_threshold': elbow_threshold
        }












    # def find_optimal_threshold(self, min_threshold: float = 0.1, max_threshold: float = 0.9, 
    #                        step: float = 0.05) -> Dict[str, Any]:
    #     """
    #     Find optimal clustering threshold using multiple metrics.
    #     """
    #     if self.final_similarity_matrix is None:
    #         self.calculate_final_similarity_matrix()
        
    #     thresholds = np.arange(min_threshold, max_threshold + step, step)
    #     results = []
        
    #     print("\n" + "=" * 80)
    #     print("FINDING OPTIMAL THRESHOLD")
    #     print("=" * 80)
        
    #     for threshold in thresholds:
    #         try:
    #             # Perform clustering at this threshold
    #             clusters = self.perform_hierarchical_clustering(threshold=threshold)
                
    #             # Skip if all files are separate clusters
    #             if len(clusters) == len(self.file_names):
    #                 continue
                    
    #             # Analyze clusters
    #             analysis = self.analyze_clusters(clusters)
                
    #             # Calculate scores
    #             silhouette_score = self._calculate_silhouette_score(clusters)
                
    #             avg_intra = np.mean(analysis['intra_cluster_similarities']) if analysis['intra_cluster_similarities'] else 0
    #             avg_inter = analysis.get('avg_inter_similarity', 0)
    #             quality_score = avg_intra - avg_inter
                
    #             # Cluster size balance
    #             cluster_sizes = [len(c) for c in clusters]
    #             if len(cluster_sizes) > 1 and np.mean(cluster_sizes) > 0:
    #                 size_balance = 1 - np.std(cluster_sizes) / np.mean(cluster_sizes)
    #             else:
    #                 size_balance = 1.0
                
    #             # Combine scores
    #             combined_score = (
    #                 0.4 * quality_score + 
    #                 0.3 * silhouette_score + 
    #                 0.2 * size_balance +
    #                 0.1 * (1 / len(clusters)) if len(clusters) > 0 else 0
    #             )
                
    #             # # Penalize if expected_clusters is specified and doesn't match
    #             # if expected_clusters is not None:
    #             #     if len(clusters) != expected_clusters:
    #             #         combined_score *= 0.7
                
    #             result = {
    #                 'threshold': threshold,
    #                 'num_clusters': len(clusters),
    #                 'avg_intra_similarity': avg_intra,
    #                 'avg_inter_similarity': avg_inter,
    #                 'quality_score': quality_score,
    #                 'silhouette_score': silhouette_score,
    #                 'size_balance': size_balance,
    #                 'combined_score': combined_score,
    #                 'clusters': clusters
    #             }
    #             results.append(result)
                
    #             print(f"  Threshold {threshold:.2f}: {len(clusters)} clusters | "
    #                 f"Quality: {quality_score:.3f} | Silhouette: {silhouette_score:.3f}")
    #         except Exception as e:
    #             # Skip thresholds that cause errors
    #             continue
        
    #     if not results:
    #         # Return a valid structure with default threshold
    #         print("⚠️  No valid thresholds found, using default threshold 0.5")
    #         clusters = self.perform_hierarchical_clustering(threshold=0.5)
    #         analysis = self.analyze_clusters(clusters)
            
    #         return {
    #             'optimal_threshold': 0.5,
    #             'optimal_clusters': clusters,
    #             'selection_method': 'default',
    #             'all_results': [{
    #                 'threshold': 0.5,
    #                 'num_clusters': len(clusters),
    #                 'avg_intra_similarity': np.mean(analysis['intra_cluster_similarities']) if analysis['intra_cluster_similarities'] else 0,
    #                 'avg_inter_similarity': analysis.get('avg_inter_similarity', 0),
    #                 'quality_score': 0,
    #                 'silhouette_score': 0,
    #                 'size_balance': 1.0,
    #                 'combined_score': 0,
    #                 'clusters': clusters
    #             }],
    #             'best_result': {
    #                 'threshold': 0.5,
    #                 'num_clusters': len(clusters),
    #                 'clusters': clusters
    #             },
    #             'elbow_threshold': 0.5
    #         }
        
    #     # Find best result by combined score
    #     best_result = max(results, key=lambda x: x['combined_score'])
        
    #     # Find elbow point
    #     elbow_threshold = self._find_elbow_point(results)
        
    #     # Choose final threshold
    #     elbow_result = next((r for r in results if abs(r['threshold'] - elbow_threshold) < step/2), None)
        
    #     if elbow_result and elbow_result['quality_score'] > 0.2:
    #         final_result = elbow_result
    #         selection_method = 'elbow_method'
    #     else:
    #         final_result = best_result
    #         selection_method = 'combined_score'
        
    #     print(f"\n✅ Optimal threshold: {final_result['threshold']:.2f} ({selection_method})")
    #     print(f"   Clusters: {final_result['num_clusters']}")
    #     print(f"   Quality score: {final_result['quality_score']:.3f}")
    #     print(f"   Silhouette score: {final_result['silhouette_score']:.3f}")
        
    #     return {
    #         'optimal_threshold': final_result['threshold'],
    #         'optimal_clusters': final_result['clusters'],
    #         'selection_method': selection_method,
    #         'all_results': results,
    #         'best_result': final_result,
    #         'elbow_threshold': elbow_threshold
    #     }
    def _calculate_silhouette_score(self, clusters: List[List[str]]) -> float:
        """
        Calculate silhouette-like score for clustering.
        Simplified version that doesn't require full distance matrix.
        """
        if len(clusters) <= 1:
            return 0.0
        
        silhouette_scores = []
        
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue
                
            # Get indices for this cluster
            indices = [self.file_names.index(f) for f in cluster]
            
            for i, idx_i in enumerate(indices):
                # Calculate a(i): average distance to other points in same cluster
                intra_dists = []
                for j, idx_j in enumerate(indices):
                    if i != j:
                        dist = 1 - self.final_similarity_matrix[idx_i, idx_j]
                        intra_dists.append(dist)
                
                a_i = np.mean(intra_dists) if intra_dists else 0
                
                # Calculate b(i): smallest average distance to points in other clusters
                b_i_values = []
                for other_cluster_idx, other_cluster in enumerate(clusters):
                    if other_cluster_idx == cluster_idx:
                        continue
                    
                    other_indices = [self.file_names.index(f) for f in other_cluster]
                    inter_dists = [1 - self.final_similarity_matrix[idx_i, idx_j] 
                                for idx_j in other_indices]
                    
                    if inter_dists:
                        b_i_values.append(np.mean(inter_dists))
                
                b_i = min(b_i_values) if b_i_values else 0
                
                # Calculate silhouette for this point
                if max(a_i, b_i) > 0:
                    s_i = (b_i - a_i) / max(a_i, b_i)
                    silhouette_scores.append(s_i)
        
        return np.mean(silhouette_scores) if silhouette_scores else 0.0

    def _find_elbow_point(self, results: List[Dict]) -> float:
        """
        Find elbow point in cluster count vs threshold curve.
        """
        if len(results) < 3:
            return results[0]['threshold'] if results else 0.5
        
        # Sort by threshold
        results_sorted = sorted(results, key=lambda x: x['threshold'])
        
        # Get cluster counts
        thresholds = [r['threshold'] for r in results_sorted]
        cluster_counts = [r['num_clusters'] for r in results_sorted]
        
        # Normalize
        t_norm = (np.array(thresholds) - min(thresholds)) / (max(thresholds) - min(thresholds))
        c_norm = (np.array(cluster_counts) - min(cluster_counts)) / (max(cluster_counts) - min(cluster_counts))
        
        # Calculate distance from each point to line from first to last point
        # Line equation: y = mx + b
        x1, y1 = t_norm[0], c_norm[0]
        x2, y2 = t_norm[-1], c_norm[-1]
        
        if x2 == x1:
            return thresholds[len(thresholds) // 2]
        
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        # Distance from point (x, y) to line y = mx + b
        distances = []
        for i in range(len(t_norm)):
            x, y = t_norm[i], c_norm[i]
            dist = abs(m * x - y + b) / np.sqrt(m**2 + 1)
            distances.append(dist)
        
        # Find point with maximum distance (elbow)
        elbow_idx = np.argmax(distances)
        
        return thresholds[elbow_idx]

    def visualize_threshold_analysis(self, results: List[Dict], output_file: str = 'threshold_analysis.png'):
        """
        Visualize threshold analysis results.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        thresholds = [r['threshold'] for r in results]
        num_clusters = [r['num_clusters'] for r in results]
        quality_scores = [r['quality_score'] for r in results]
        silhouette_scores = [r['silhouette_score'] for r in results]
        combined_scores = [r['combined_score'] for r in results]
        
        # Plot 1: Cluster count vs threshold
        ax = axes[0, 0]
        ax.plot(thresholds, num_clusters, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Cluster Count vs Threshold')
        ax.grid(True, alpha=0.3)
        
        # Highlight elbow point
        elbow_threshold = self._find_elbow_point(results)
        elbow_idx = thresholds.index(min(thresholds, key=lambda x: abs(x - elbow_threshold)))
        ax.plot(thresholds[elbow_idx], num_clusters[elbow_idx], 'ro', markersize=10, label='Elbow Point')
        ax.legend()
        
        # Plot 2: Quality metrics vs threshold
        ax = axes[0, 1]
        ax.plot(thresholds, quality_scores, 'go-', linewidth=2, markersize=6, label='Quality Score')
        ax.plot(thresholds, silhouette_scores, 'mo-', linewidth=2, markersize=6, label='Silhouette Score')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Quality Metrics vs Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Combined score vs threshold
        ax = axes[1, 0]
        ax.plot(thresholds, combined_scores, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Combined Score')
        ax.set_title('Combined Score vs Threshold')
        ax.grid(True, alpha=0.3)
        
        # Highlight best threshold
        best_idx = np.argmax(combined_scores)
        ax.plot(thresholds[best_idx], combined_scores[best_idx], 'k*', markersize=15, label='Best Threshold')
        ax.legend()
        
        # Plot 4: Intra vs Inter similarity
        ax = axes[1, 1]
        intra_sims = [r['avg_intra_similarity'] for r in results]
        inter_sims = [r['avg_inter_similarity'] for r in results]
        
        ax.plot(thresholds, intra_sims, 'b^-', linewidth=2, markersize=6, label='Intra-cluster Similarity')
        ax.plot(thresholds, inter_sims, 'rv-', linewidth=2, markersize=6, label='Inter-cluster Similarity')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Similarity')
        ax.set_title('Intra vs Inter Cluster Similarity')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.suptitle('Threshold Analysis for Schema Clustering', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Threshold analysis visualization saved to {output_file}")
def clustering(all_features):

    # 1. Load features
    calculator = SchemaSimilarityCalculator()
    calculator.load_features(all_features)

    # 2. Calculate all pairwise similarities
    calculator.calculate_all_pairwise_similarities()
    
    # 3. Calculate final similarity matrix
    final_matrix = calculator.calculate_final_similarity_matrix()



    # Find optimal threshold (expecting 3 clusters)
    threshold_analysis = calculator.find_optimal_threshold(
        min_threshold=0.1,
        max_threshold=0.9,
        step=0.05
        # expected_clusters=3  # You know there should be 3 entity types
    )
    import os
    output_dir = 'clustering_results'
    os.makedirs(output_dir, exist_ok=True)
    # Visualize threshold analysis
    calculator.visualize_threshold_analysis(
        threshold_analysis['all_results'],
        'clustering_results/threshold_analysis.png'
    )
    
    # Use optimal threshold for final clustering
    optimal_threshold = threshold_analysis['optimal_threshold']
    optimal_clusters = threshold_analysis['optimal_clusters']
    
    # Update calculator with optimal clusters
    clusters = optimal_clusters




    # 4. Perform clustering with different thresholds
    print("\n🔍 Testing different clustering thresholds:")
    for threshold in [optimal_threshold]:#[0.3, 0.4, 0.5, 0.6, 0.7]:
        clusters = calculator.perform_hierarchical_clustering(threshold=threshold)
        print(f"  Threshold {threshold}: {len(clusters)} clusters")
        
        # Analyze clusters for threshold 0.5 (default)
        if threshold == 0.3:
            analysis = calculator.analyze_clusters(clusters)
            print(f"\n📊 Cluster Analysis (threshold={threshold}):")
            print(f"  Overall quality: {analysis['overall_quality']:.3f} ({analysis.get('quality_rating', 'N/A')})")
            print(f"  Number of clusters: {len(clusters)}")
            
            # Show cluster details
            for cluster_info in analysis['cluster_stats']:
                if cluster_info['size'] > 1:
                    print(f"  Cluster {cluster_info['cluster_id']}: {cluster_info['size']} files, "
                          f"avg similarity: {cluster_info['avg_intra_similarity']:.3f}")
    
    # 5. Visualize results
    calculator.visualize_results('clustering_results')
    
    # 6. Save all results
    calculator.save_results('final_clustering_results.json')
    
    # 7. Generate final report
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    # Get final clusters with threshold 0.5
    final_clusters = calculator.clusters
    print(f"\n📁 Total files analyzed: {len(calculator.file_names)}")
    print(f"📊 Number of clusters: {len(final_clusters)}")
    
    print("\n📋 Detected Clusters:")
    for i, cluster in enumerate(final_clusters, 1):
        print(f"\nCluster {i} ({len(cluster)} files):")
        for file in sorted(cluster):
            print(f"  - {file}")
    
    print("\n✅ Phase 3 complete! Results saved to:")
    print("   - clustering_results/ (visualizations)")
    print("   - final_clustering_results.json (detailed results)")
    print("=" * 80)
    

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 3: PAIRWISE SIMILARITY CALCULATION & CLUSTERING")
    print("=" * 80)
    
    # 1. Load features
    calculator = SchemaSimilarityCalculator()
    calculator.load_features('all_features.json')
    
    # 2. Calculate all pairwise similarities
    calculator.calculate_all_pairwise_similarities()
    
    # 3. Calculate final similarity matrix
    final_matrix = calculator.calculate_final_similarity_matrix()
    













    # Find optimal threshold (expecting 3 clusters)
    print("\n" + "=" * 80)
    print("ANALYZING OPTIMAL CLUSTERING THRESHOLD")
    print("=" * 80)
    threshold_analysis = calculator.find_optimal_threshold(
        min_threshold=0.1,
        max_threshold=0.9,
        step=0.05
        # expected_clusters=3  # You know there should be 3 entity types
    )
    
    # Visualize threshold analysis
    calculator.visualize_threshold_analysis(
        threshold_analysis['all_results'],
        'clustering_results/threshold_analysis.png'
    )
    
    # Use optimal threshold for final clustering
    optimal_threshold = threshold_analysis['optimal_threshold']
    optimal_clusters = threshold_analysis['optimal_clusters']
    
    print(f"\n🎯 Using optimal threshold: {optimal_threshold:.2f}")
    print(f"📊 Number of clusters: {len(optimal_clusters)}")
    
    # Update calculator with optimal clusters
    clusters = optimal_clusters
    











    # 4. Perform clustering with different thresholds
    print("\n🔍 Testing different clustering thresholds:")
    for threshold in [optimal_threshold]:#[0.3, 0.4, 0.5, 0.6, 0.7]:
        clusters = calculator.perform_hierarchical_clustering(threshold=threshold)
        print(f"  Threshold {threshold}: {len(clusters)} clusters")
        
        # Analyze clusters for threshold 0.5 (default)
        if threshold == 0.3:
            analysis = calculator.analyze_clusters(clusters)
            print(f"\n📊 Cluster Analysis (threshold={threshold}):")
            print(f"  Overall quality: {analysis['overall_quality']:.3f} ({analysis.get('quality_rating', 'N/A')})")
            print(f"  Number of clusters: {len(clusters)}")
            
            # Show cluster details
            for cluster_info in analysis['cluster_stats']:
                if cluster_info['size'] > 1:
                    print(f"  Cluster {cluster_info['cluster_id']}: {cluster_info['size']} files, "
                          f"avg similarity: {cluster_info['avg_intra_similarity']:.3f}")
    
    # 5. Visualize results
    calculator.visualize_results('clustering_results')
    
    # 6. Save all results
    calculator.save_results('final_clustering_results.json')
    
    # 7. Generate final report
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    # Get final clusters with threshold 0.5
    final_clusters = calculator.clusters
    
    print(f"\n📁 Total files analyzed: {len(calculator.file_names)}")
    print(f"📊 Number of clusters: {len(final_clusters)}")
    print(f"🎯 Recommended threshold: 0.5")
    
    print("\n📋 Detected Clusters:")
    for i, cluster in enumerate(final_clusters, 1):
        print(f"\nCluster {i} ({len(cluster)} files):")
        for file in sorted(cluster):
            print(f"  - {file}")
    
    print("\n✅ Phase 3 complete! Results saved to:")
    print("   - clustering_results/ (visualizations)")
    print("   - final_clustering_results.json (detailed results)")
    print("=" * 80)