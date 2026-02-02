import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

class ColumnClusteringMapper:
    def __init__(self, enhanced_schemas: Dict[str, Any]):
        """
        Apply SAME clustering algorithm to columns as we did for files.
        """
        self.enhanced_schemas = enhanced_schemas
        self.word_embeddings = None
        self._load_word_embeddings()
        
    def map_cluster(self, cluster_files: List[str]) -> Dict[str, Any]:
        """
        Map cluster using same similarity approach as file clustering.
        """
        print(f"\n🔗 Processing {len(cluster_files)} files with column-level clustering")
        
        # Step 1: Extract column features (SAME as file features but for columns)
        column_features = self._extract_column_features(cluster_files)
        
        # Step 2: Calculate similarity matrix (SAME algorithm as file clustering)
        similarity_matrix = self._calculate_column_similarity_matrix(column_features)
        
        # Step 3: Cluster columns (SAME hierarchical clustering)
        column_clusters = self._cluster_columns_hierarchical(similarity_matrix, column_features)
        
        # Step 4: Create golden schema from clusters
        golden_schema = self._create_golden_schema_from_clusters(column_clusters, column_features)
        
        # Step 5: Create mappings
        mappings = self._create_mappings_from_clusters(column_clusters, column_features, golden_schema)
        
        return {
            'cluster_files': cluster_files,
            'column_features': {f['full_key']: f for f in column_features},
            'column_clusters': column_clusters,
            'golden_schema': golden_schema,
            'mappings': mappings
        }
    
    def _extract_column_features(self, cluster_files: List[str]) -> List[Dict[str, Any]]:
        """
        Extract features for each column - SIMILAR to how we extracted file features.
        """
        column_features = []
        
        for filename in cluster_files:
            if filename not in self.enhanced_schemas:
                continue
                
            schema = self.enhanced_schemas[filename]
            
            for col_idx, col_name in enumerate(schema.get('columns', [])):
                # Get data from enhanced_schemas
                col_type = schema.get('column_types', {}).get(col_name, 'unknown')
                patterns = schema.get('value_patterns', {}).get(col_name, {})
                
                # Create feature vector SIMILAR to file features
                features = {
                    'full_key': f"{filename}__{col_name}",
                    'file': filename,
                    'column': col_name,
                    'position': col_idx,
                    
                    # Criterion 1: Name features (like file name similarity)
                    'name_features': {
                        'original': col_name,
                        'lowercase': col_name.lower(),
                        'tokens': col_name.lower().split('_') if '_' in col_name else [col_name.lower()]
                    },
                    
                    # Criterion 2: Type features (like schema length similarity)
                    'type_features': {
                        'inferred_type': col_type,
                        'is_numeric': col_type in ['integer', 'float'],
                        'is_string': col_type == 'string',
                        'is_datetime': col_type == 'datetime'
                    },
                    
                    # Criterion 3: Pattern features (like value pattern similarity)
                    'pattern_features': {
                        'digit_ratio': patterns.get('digit_ratio', 0),
                        'unique_ratio': patterns.get('unique_ratio', 0),
                        'char_distribution': patterns.get('char_distribution', {}),
                        'length_stats': patterns.get('length_stats', {})
                    },
                    
                    # Criterion 4: Statistical features
                    'statistical_features': self._extract_statistical_features(patterns),
                    
                    # Additional context
                    'total_columns_in_file': len(schema.get('columns', [])),
                    'column_samples': schema.get('value_samples', {}).get(col_name, [])[:5]
                }
                
                column_features.append(features)
        
        print(f"  Extracted features for {len(column_features)} columns")
        return column_features
    
    def _extract_statistical_features(self, patterns: Dict) -> Dict[str, float]:
        """Extract statistical features from patterns."""
        stats = {}
        
        if patterns:
            stats['digit_ratio'] = patterns.get('digit_ratio', 0)
            stats['unique_ratio'] = patterns.get('unique_ratio', 0)
            
            char_dist = patterns.get('char_distribution', {})
            if char_dist:
                stats['alpha_ratio'] = char_dist.get('alpha', 0)
                stats['special_ratio'] = char_dist.get('special', 0)
            
            length_stats = patterns.get('length_stats', {})
            if length_stats:
                stats['mean_length'] = length_stats.get('mean', 0)
                stats['length_variation'] = length_stats.get('std', 0) / max(length_stats.get('mean', 1), 1)
        
        return stats
    
    def _calculate_column_similarity_matrix(self, column_features: List[Dict]) -> np.ndarray:
        """
        Calculate similarity matrix using SAME weighted approach as file clustering.
        """
        n = len(column_features)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
                
                # Calculate similarity using SAME multi-criteria approach
                similarity = self._calculate_column_pair_similarity(
                    column_features[i], 
                    column_features[j]
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _calculate_column_pair_similarity(self, col1: Dict, col2: Dict) -> float:
        """
        Calculate similarity between two columns using SAME weighted criteria as files.
        
        Files used: file_name(0.15), schema_length(0.15), column_names(0.30), 
                   data_types(0.20), value_patterns(0.20)
        
        For columns we'll use similar criteria with similar weights.
        """
        # Criterion 1: Name similarity (30% weight - like column_names for files)
        name_similarity = self._calculate_name_similarity_score(
            col1['name_features'], 
            col2['name_features']
        )
        
        # Criterion 2: Type compatibility (20% weight - like data_types for files)
        type_similarity = self._calculate_type_similarity_score(
            col1['type_features'], 
            col2['type_features']
        )
        
        # Criterion 3: Pattern similarity (25% weight - like value_patterns for files)
        pattern_similarity = self._calculate_pattern_similarity_score(
            col1['pattern_features'], 
            col2['pattern_features']
        )
        
        # Criterion 4: Statistical similarity (15% weight)
        statistical_similarity = self._calculate_statistical_similarity_score(
            col1['statistical_features'], 
            col2['statistical_features']
        )
        
        # Criterion 5: Context similarity (10% weight - like positional context)
        context_similarity = self._calculate_context_similarity_score(col1, col2)
        
        # SAME weighted combination approach as file clustering
        weights = {
            'name': 0.30,      # Like column_names for files
            'type': 0.20,      # Like data_types for files  
            'pattern': 0.25,   # Like value_patterns for files
            'statistical': 0.15,
            'context': 0.10
        }
        
        total_similarity = (
            weights['name'] * name_similarity +
            weights['type'] * type_similarity +
            weights['pattern'] * pattern_similarity +
            weights['statistical'] * statistical_similarity +
            weights['context'] * context_similarity
        )
        
        return max(0.0, min(1.0, total_similarity))
    
    def _calculate_name_similarity_score(self, name1: Dict, name2: Dict) -> float:
        """Calculate name similarity (like file name similarity for files)."""
        # Similar to how we calculated file name similarity
        name1_lower = name1['lowercase']
        name2_lower = name2['lowercase']
        
        if name1_lower == name2_lower:
            return 1.0
        
        # Edit distance (like we used for files)
        edit_sim = SequenceMatcher(None, name1_lower, name2_lower).ratio()
        
        # Token overlap (like we used for files)
        tokens1 = name1['tokens']
        tokens2 = name2['tokens']
        
        if tokens1 and tokens2:
            set1 = set(tokens1)
            set2 = set(tokens2)
            token_sim = len(set1.intersection(set2)) / len(set1.union(set2))
        else:
            token_sim = 0.0
        
        # Combined score (similar weighting as file name similarity)
        return 0.7 * edit_sim + 0.3 * token_sim
    
    def _calculate_type_similarity_score(self, type1: Dict, type2: Dict) -> float:
        """Calculate type compatibility (like data_type similarity for files)."""
        if type1['inferred_type'] == type2['inferred_type']:
            return 1.0
        
        # Numeric types are compatible
        if type1['is_numeric'] and type2['is_numeric']:
            return 0.8
        
        # String is somewhat compatible with other types
        if type1['is_string'] or type2['is_string']:
            return 0.5
        
        return 0.3
    
    def _calculate_pattern_similarity_score(self, patterns1: Dict, patterns2: Dict) -> float:
        """Calculate pattern similarity (like value_pattern similarity for files)."""
        if not patterns1 or not patterns2:
            return 0.5
        
        similarities = []
        
        # Compare digit ratios (SAME as file pattern comparison)
        digit1 = patterns1.get('digit_ratio', 0.5)
        digit2 = patterns2.get('digit_ratio', 0.5)
        similarities.append(1 - abs(digit1 - digit2))
        
        # Compare unique ratios (SAME as file pattern comparison)
        unique1 = patterns1.get('unique_ratio', 0.5)
        unique2 = patterns2.get('unique_ratio', 0.5)
        similarities.append(1 - abs(unique1 - unique2))
        
        # Compare character distributions (similar approach as files)
        char_dist1 = patterns1.get('char_distribution', {})
        char_dist2 = patterns2.get('char_distribution', {})
        
        if char_dist1 and char_dist2:
            for char_type in ['alpha', 'digit', 'special', 'space']:
                val1 = char_dist1.get(char_type, 0)
                val2 = char_dist2.get(char_type, 0)
                similarities.append(1 - abs(val1 - val2))
        
        # Compare length statistics (similar approach as files)
        len_stats1 = patterns1.get('length_stats', {})
        len_stats2 = patterns2.get('length_stats', {})
        
        if len_stats1 and len_stats2:
            mean1 = len_stats1.get('mean', 0)
            mean2 = len_stats2.get('mean', 0)
            if mean1 > 0 or mean2 > 0:
                len_sim = 1 - abs(mean1 - mean2) / max(mean1, mean2, 1)
                similarities.append(len_sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_statistical_similarity_score(self, stats1: Dict, stats2: Dict) -> float:
        """Calculate statistical similarity."""
        if not stats1 or not stats2:
            return 0.5
        
        similarities = []
        
        # Compare all available statistics
        all_keys = set(stats1.keys()).union(set(stats2.keys()))
        for key in all_keys:
            val1 = stats1.get(key, 0)
            val2 = stats2.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if max(abs(val1), abs(val2)) > 0:
                    similarity = 1 - abs(val1 - val2) / max(abs(val1), abs(val2), 1)
                else:
                    similarity = 1.0
                
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_context_similarity_score(self, col1: Dict, col2: Dict) -> float:
        """Calculate context similarity (position, file context)."""
        similarities = []
        
        # Position similarity (columns in similar relative positions might be related)
        if col1['total_columns_in_file'] > 0 and col2['total_columns_in_file'] > 0:
            pos1_norm = col1['position'] / col1['total_columns_in_file']
            pos2_norm = col2['position'] / col2['total_columns_in_file']
            pos_similarity = 1 - abs(pos1_norm - pos2_norm)
            similarities.append(pos_similarity)
        
        # Same file bonus (columns from same file might have different relationships)
        if col1['file'] == col2['file']:
            similarities.append(0.3)  # Small bonus for same file
        
        return np.mean(similarities) if similarities else 0.5
    
    def _cluster_columns_hierarchical(self, similarity_matrix: np.ndarray,
                                     column_features: List[Dict]) -> List[List[int]]:
        """
        Cluster columns using SAME hierarchical clustering as files.
        """
        if len(column_features) <= 1:
            return [[0]] if column_features else []
        
        # Convert similarity to distance (SAME as files)
        distance_matrix = 1 - similarity_matrix
        
        # Perform hierarchical clustering (SAME as files)
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        
        try:
            linkage_matrix = linkage(squareform(distance_matrix), method='average')
            
            # Use SAME threshold logic as files (0.5 threshold = 0.5 distance)
            clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
            
            # Group by cluster
            cluster_dict = defaultdict(list)
            for idx, cluster_id in enumerate(clusters):
                cluster_dict[cluster_id].append(idx)
            
            result = list(cluster_dict.values())
            
            print(f"  Created {len(result)} column clusters (threshold: 0.5)")
            return result
            
        except Exception as e:
            print(f"  Clustering failed: {e}")
            # Fallback: group by high similarity
            return self._fallback_similarity_clustering(similarity_matrix)
    
    def _fallback_similarity_clustering(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Fallback clustering based on similarity matrix."""
        n = similarity_matrix.shape[0]
        clusters = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            current_cluster = [i]
            assigned.add(i)
            
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                if similarity_matrix[i, j] >= 0.5:  # SAME threshold as files
                    current_cluster.append(j)
                    assigned.add(j)
            
            clusters.append(current_cluster)
        
        return clusters
    
    def _create_golden_schema_from_clusters(self, column_clusters: List[List[int]],
                                           column_features: List[Dict]) -> Dict[str, Any]:
        """
        Create golden schema from column clusters.
        """
        golden_columns = []
        column_details = {}
        
        for cluster_idx, cluster_indices in enumerate(column_clusters, 1):
            if not cluster_indices:
                continue
            
            # Get all columns in this cluster
            cluster_cols = [column_features[i] for i in cluster_indices]
            
            # Create golden column
            golden_col = self._create_golden_column(cluster_cols, cluster_idx)
            
            if golden_col:
                golden_columns.append(golden_col['name'])
                column_details[golden_col['name']] = golden_col
        
        # Create golden schema
        golden_schema = {
            'name': f"Golden_Schema_{len(column_clusters)}",
            'total_columns': len(golden_columns),
            'columns': golden_columns,
            'column_details': column_details,
            'clusters_found': len(column_clusters)
        }
        
        print(f"  Created golden schema with {len(golden_columns)} columns")
        return golden_schema
    
    def _create_golden_column(self, cluster_cols: List[Dict], cluster_idx: int) -> Dict[str, Any]:
        """Create a golden column from a cluster."""
        if not cluster_cols:
            return None
        
        # Choose name: most common name or most representative
        column_names = [col['column'] for col in cluster_cols]
        
        # Find most common structural pattern
        def get_simple_pattern(name):
            # Simple pattern: count underscores
            return name.count('_')
        
        # Group by simple pattern
        pattern_groups = defaultdict(list)
        for name in column_names:
            pattern = get_simple_pattern(name)
            pattern_groups[pattern].append(name)
        
        # Choose from largest group
        largest_group = max(pattern_groups.values(), key=len, default=column_names)
        chosen_name = min(largest_group, key=len)  # Shortest in largest group
        
        # Calculate cluster statistics
        digit_ratios = [col['pattern_features'].get('digit_ratio', 0) for col in cluster_cols]
        unique_ratios = [col['pattern_features'].get('unique_ratio', 0) for col in cluster_cols]
        types = [col['type_features']['inferred_type'] for col in cluster_cols]
        
        # Most common type
        from collections import Counter
        type_counts = Counter(types)
        most_common_type = type_counts.most_common(1)[0][0] if type_counts else 'unknown'
        
        golden_col = {
            'name': chosen_name,
            'cluster_id': cluster_idx,
            'cluster_size': len(cluster_cols),
            'inferred_type': most_common_type,
            'statistics': {
                'avg_digit_ratio': np.mean(digit_ratios) if digit_ratios else 0,
                'avg_unique_ratio': np.mean(unique_ratios) if unique_ratios else 0,
                'digit_ratio_std': np.std(digit_ratios) if len(digit_ratios) > 1 else 0,
                'type_distribution': dict(type_counts)
            },
            'source_columns': [
                {
                    'file': col['file'],
                    'column': col['column'],
                    'type': col['type_features']['inferred_type'],
                    'digit_ratio': col['pattern_features'].get('digit_ratio', 0)
                }
                for col in cluster_cols
            ]
        }
        
        return golden_col
    
    def _create_mappings_from_clusters(self, column_clusters: List[List[int]],
                                      column_features: List[Dict],
                                      golden_schema: Dict[str, Any]) -> Dict[str, List]:
        """
        Create mappings from clusters.
        """
        mappings = {}
        
        # For each golden column
        for golden_col_name, golden_col_details in golden_schema['column_details'].items():
            cluster_idx = golden_col_details['cluster_id']
            
            if cluster_idx <= len(column_clusters):
                cluster_indices = column_clusters[cluster_idx - 1]
                mappings[golden_col_name] = []
                
                for idx in cluster_indices:
                    source_col = column_features[idx]
                    
                    # Calculate confidence based on similarity to cluster center
                    confidence = self._calculate_mapping_confidence(
                        source_col, 
                        golden_col_details,
                        column_features,
                        cluster_indices
                    )
                    
                    mapping = {
                        'source_file': source_col['file'],
                        'source_column': source_col['column'],
                        'golden_column': golden_col_name,
                        'confidence': confidence,
                        'confidence_level': 'HIGH' if confidence >= 0.7 else 
                                          'MEDIUM' if confidence >= 0.5 else 'LOW'
                    }
                    
                    mappings[golden_col_name].append(mapping)
        
        return mappings
    
    def _calculate_mapping_confidence(self, source_col: Dict, 
                                     golden_col: Dict,
                                     all_features: List[Dict],
                                     cluster_indices: List[int]) -> float:
        """Calculate mapping confidence."""
        # Calculate average similarity to other columns in cluster
        similarities = []
        source_idx = None
        
        # Find source column index
        for idx, col in enumerate(all_features):
            if (col['file'] == source_col['file'] and 
                col['column'] == source_col['column']):
                source_idx = idx
                break
        
        if source_idx is None:
            return 0.5
        
        # Calculate similarity to other columns in same cluster
        for other_idx in cluster_indices:
            if other_idx != source_idx and other_idx < len(all_features):
                other_col = all_features[other_idx]
                similarity = self._calculate_column_pair_similarity(source_col, other_col)
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            # Confidence is based on how similar this column is to others in its cluster
            confidence = min(1.0, avg_similarity * 1.2)  # Slight boost
        else:
            confidence = 0.5
        
        return confidence
    
    def map_all_clusters(self, clusters: List[List[str]]) -> Dict[str, Any]:
        """
        Map all clusters using SAME algorithm as file clustering.
        """
        print("\n" + "=" * 80)
        print("COLUMN-LEVEL CLUSTERING (SAME ALGORITHM AS FILES)")
        print("=" * 80)
        
        all_results = {}
        
        for cluster_idx, cluster_files in enumerate(clusters, 1):
            print(f"\n📊 Cluster {cluster_idx}/{len(clusters)}: {len(cluster_files)} files")
            
            result = self.map_cluster(cluster_files)
            all_results[f"Cluster_{cluster_idx}"] = result
        
        # Summary
        total_mappings = 0
        high_conf = 0
        
        for cluster_name, result in all_results.items():
            mappings = result['mappings']
            for mapping_list in mappings.values():
                total_mappings += len(mapping_list)
                high_conf += sum(1 for m in mapping_list 
                               if m.get('confidence_level') == 'HIGH')
        
        print(f"\n✅ Column clustering complete!")
        print(f"📊 Total mappings: {total_mappings}")
        print(f"🎯 High confidence mappings: {high_conf} ({high_conf/total_mappings:.1%})")
        
        return all_results
    





































    def _load_word_embeddings(self):
        """Load pre-trained word embeddings for semantic similarity."""
        try:
            print("📥 Loading word embeddings for semantic similarity...")
            self.word_embeddings = api.load('glove-wiki-gigaword-100')
            print("✅ Word embeddings loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load word embeddings: {e}")
            print("   Using only structural similarity")
            self.word_embeddings = None
    
    def _calculate_semantic_similarity(self, token1: str, token2: str) -> float:
        """Calculate semantic similarity between two tokens using word embeddings."""
        if not self.word_embeddings:
            return 0.5  # Neutral if no embeddings available
        
        if token1 in self.word_embeddings and token2 in self.word_embeddings:
            try:
                return float(self.word_embeddings.similarity(token1, token2))
            except:
                return 0.5
        return 0.5
    
    def _extract_tokens(self, column_name: str) -> List[str]:
        """Extract meaningful tokens from column name."""
        # Remove common suffixes
        suffixes = ['_id', '_name', '_date', '_time', '_email', '_phone', 
                   '_address', '_city', '_state', '_zip', '_country', '_code',
                   '_price', '_amount', '_total', '_quantity', '_status']
        
        name = column_name.lower()
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        # Split by common separators
        tokens = []
        for part in name.replace('_', ' ').replace('-', ' ').replace('.', ' ').split():
            if part and len(part) > 1:  # Ignore single letters
                tokens.append(part)
        
        return tokens
    
    def _calculate_name_similarity_score(self, name1: Dict, name2: Dict) -> float:
        """Calculate name similarity with SEMANTIC understanding."""
        # Get the original column names
        col1_name = name1['original']
        col2_name = name2['original']
        
        if col1_name.lower() == col2_name.lower():
            return 1.0
        
        # 1. Structural similarity (existing approach - 60% weight)
        structural_sim = self._calculate_structural_similarity(name1, name2)
        
        # 2. Semantic similarity (new - 40% weight)
        semantic_sim = self._calculate_semantic_component_similarity(col1_name, col2_name)
        
        # Combined score
        return 0.6 * structural_sim + 0.4 * semantic_sim
    
    def _calculate_structural_similarity(self, name1: Dict, name2: Dict) -> float:
        """Calculate structural similarity (existing approach)."""
        name1_lower = name1['lowercase']
        name2_lower = name2['lowercase']
        
        # Edit distance
        edit_sim = SequenceMatcher(None, name1_lower, name2_lower).ratio()
        
        # Token overlap
        tokens1 = name1['tokens']
        tokens2 = name2['tokens']
        
        if tokens1 and tokens2:
            set1 = set(tokens1)
            set2 = set(tokens2)
            token_sim = len(set1.intersection(set2)) / len(set1.union(set2))
        else:
            token_sim = 0.0
        
        return 0.7 * edit_sim + 0.3 * token_sim
    
    def _calculate_semantic_component_similarity(self, col1_name: str, col2_name: str) -> float:
        """Calculate semantic similarity between column names."""
        if not self.word_embeddings:
            return 0.5
        
        # Extract meaningful tokens
        tokens1 = self._extract_tokens(col1_name)
        tokens2 = self._extract_tokens(col2_name)
        
        if not tokens1 or not tokens2:
            return 0.5
        
        # Calculate best semantic matches between tokens
        similarities = []
        
        for token1 in tokens1:
            for token2 in tokens2:
                semantic_sim = self._calculate_semantic_similarity(token1, token2)
                similarities.append(semantic_sim)
        
        if similarities:
            # Take average of semantic similarities
            return float(np.mean(similarities))
        
        return 0.5
    def _extract_column_features(self, cluster_files: List[str]) -> List[Dict[str, Any]]:
        """Extract features for each column - SIMILAR to how we extracted file features."""
        column_features = []
        
        for filename in cluster_files:
            if filename not in self.enhanced_schemas:
                continue
                
            schema = self.enhanced_schemas[filename]
            
            for col_idx, col_name in enumerate(schema.get('columns', [])):
                # Get data from enhanced_schemas
                col_type = schema.get('column_types', {}).get(col_name, 'unknown')
                patterns = schema.get('value_patterns', {}).get(col_name, {})
                
                # Create feature vector SIMILAR to file features
                features = {
                    'full_key': f"{filename}__{col_name}",
                    'file': filename,
                    'column': col_name,
                    'position': col_idx,
                    
                    # Criterion 1: Name features (like file name similarity)
                    'name_features': {
                        'original': col_name,
                        'lowercase': col_name.lower(),
                        'tokens': col_name.lower().split('_') if '_' in col_name else [col_name.lower()]
                    },
                    
                    # Criterion 2: Type features (like schema length similarity)
                    'type_features': {
                        'inferred_type': col_type,
                        'is_numeric': col_type in ['integer', 'float'],
                        'is_string': col_type == 'string',
                        'is_datetime': col_type == 'datetime'
                    },
                    
                    # Criterion 3: Pattern features (like value pattern similarity)
                    'pattern_features': {
                        'digit_ratio': patterns.get('digit_ratio', 0),
                        'unique_ratio': patterns.get('unique_ratio', 0),
                        'char_distribution': patterns.get('char_distribution', {}),
                        'length_stats': patterns.get('length_stats', {})
                    },
                    
                    # Criterion 4: Statistical features
                    'statistical_features': self._extract_statistical_features(patterns),
                    
                    # Additional context
                    'total_columns_in_file': len(schema.get('columns', [])),
                    'column_samples': schema.get('value_samples', {}).get(col_name, [])[:5]
                }
                
                column_features.append(features)
        
        print(f"  Extracted features for {len(column_features)} columns")
        return column_features
    
    def _extract_statistical_features(self, patterns: Dict) -> Dict[str, float]:
        """Extract statistical features from patterns."""
        stats = {}
        
        if patterns:
            stats['digit_ratio'] = patterns.get('digit_ratio', 0)
            stats['unique_ratio'] = patterns.get('unique_ratio', 0)
            
            char_dist = patterns.get('char_distribution', {})
            if char_dist:
                stats['alpha_ratio'] = char_dist.get('alpha', 0)
                stats['special_ratio'] = char_dist.get('special', 0)
            
            length_stats = patterns.get('length_stats', {})
            if length_stats:
                stats['mean_length'] = length_stats.get('mean', 0)
                stats['length_variation'] = length_stats.get('std', 0) / max(length_stats.get('mean', 1), 1)
        
        return stats
    
    def _calculate_column_similarity_matrix(self, column_features: List[Dict]) -> np.ndarray:
        """
        Calculate similarity matrix using SAME weighted approach as file clustering.
        """
        n = len(column_features)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
                
                # Calculate similarity using SAME multi-criteria approach
                similarity = self._calculate_column_pair_similarity(
                    column_features[i], 
                    column_features[j]
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _calculate_column_pair_similarity(self, col1: Dict, col2: Dict) -> float:
        """
        Calculate similarity between two columns using SAME weighted criteria as files.
        """
        # Criterion 1: Enhanced Name similarity with semantics (30% weight)
        name_similarity = self._calculate_name_similarity_score(
            col1['name_features'], 
            col2['name_features']
        )
        
        # Criterion 2: Type compatibility (20% weight - like data_types for files)
        type_similarity = self._calculate_type_similarity_score(
            col1['type_features'], 
            col2['type_features']
        )
        
        # Criterion 3: Pattern similarity (25% weight - like value_patterns for files)
        pattern_similarity = self._calculate_pattern_similarity_score(
            col1['pattern_features'], 
            col2['pattern_features']
        )
        
        # Criterion 4: Statistical similarity (15% weight)
        statistical_similarity = self._calculate_statistical_similarity_score(
            col1['statistical_features'], 
            col2['statistical_features']
        )
        
        # Criterion 5: Context similarity (10% weight - like positional context)
        context_similarity = self._calculate_context_similarity_score(col1, col2)
        
        # SAME weighted combination approach as file clustering
        weights = {
            'name': 0.30,      # Like column_names for files
            'type': 0.20,      # Like data_types for files  
            'pattern': 0.25,   # Like value_patterns for files
            'statistical': 0.15,
            'context': 0.10
        }
        
        total_similarity = (
            weights['name'] * name_similarity +
            weights['type'] * type_similarity +
            weights['pattern'] * pattern_similarity +
            weights['statistical'] * statistical_similarity +
            weights['context'] * context_similarity
        )
        
        return max(0.0, min(1.0, total_similarity))
    
    def _calculate_type_similarity_score(self, type1: Dict, type2: Dict) -> float:
        """Calculate type compatibility (like data_type similarity for files)."""
        if type1['inferred_type'] == type2['inferred_type']:
            return 1.0
        
        # Numeric types are compatible
        if type1['is_numeric'] and type2['is_numeric']:
            return 0.8
        
        # String is somewhat compatible with other types
        if type1['is_string'] or type2['is_string']:
            return 0.5
        
        return 0.3
    
    def _calculate_pattern_similarity_score(self, patterns1: Dict, patterns2: Dict) -> float:
        """Calculate pattern similarity (like value_pattern similarity for files)."""
        if not patterns1 or not patterns2:
            return 0.5
        
        similarities = []
        
        # Compare digit ratios (SAME as file pattern comparison)
        digit1 = patterns1.get('digit_ratio', 0.5)
        digit2 = patterns2.get('digit_ratio', 0.5)
        similarities.append(1 - abs(digit1 - digit2))
        
        # Compare unique ratios (SAME as file pattern comparison)
        unique1 = patterns1.get('unique_ratio', 0.5)
        unique2 = patterns2.get('unique_ratio', 0.5)
        similarities.append(1 - abs(unique1 - unique2))
        
        # Compare character distributions (similar approach as files)
        char_dist1 = patterns1.get('char_distribution', {})
        char_dist2 = patterns2.get('char_distribution', {})
        
        if char_dist1 and char_dist2:
            for char_type in ['alpha', 'digit', 'special', 'space']:
                val1 = char_dist1.get(char_type, 0)
                val2 = char_dist2.get(char_type, 0)
                similarities.append(1 - abs(val1 - val2))
        
        # Compare length statistics (similar approach as files)
        len_stats1 = patterns1.get('length_stats', {})
        len_stats2 = patterns2.get('length_stats', {})
        
        if len_stats1 and len_stats2:
            mean1 = len_stats1.get('mean', 0)
            mean2 = len_stats2.get('mean', 0)
            if mean1 > 0 or mean2 > 0:
                len_sim = 1 - abs(mean1 - mean2) / max(mean1, mean2, 1)
                similarities.append(len_sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_statistical_similarity_score(self, stats1: Dict, stats2: Dict) -> float:
        """Calculate statistical similarity."""
        if not stats1 or not stats2:
            return 0.5
        
        similarities = []
        
        # Compare all available statistics
        all_keys = set(stats1.keys()).union(set(stats2.keys()))
        for key in all_keys:
            val1 = stats1.get(key, 0)
            val2 = stats2.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if max(abs(val1), abs(val2)) > 0:
                    similarity = 1 - abs(val1 - val2) / max(abs(val1), abs(val2), 1)
                else:
                    similarity = 1.0
                
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_context_similarity_score(self, col1: Dict, col2: Dict) -> float:
        """Calculate context similarity (position, file context)."""
        similarities = []
        
        # Position similarity (columns in similar relative positions might be related)
        if col1['total_columns_in_file'] > 0 and col2['total_columns_in_file'] > 0:
            pos1_norm = col1['position'] / col1['total_columns_in_file']
            pos2_norm = col2['position'] / col2['total_columns_in_file']
            pos_similarity = 1 - abs(pos1_norm - pos2_norm)
            similarities.append(pos_similarity)
        
        # Same file bonus (columns from same file might have different relationships)
        if col1['file'] == col2['file']:
            similarities.append(0.3)  # Small bonus for same file
        
        return np.mean(similarities) if similarities else 0.5
    
    def _cluster_columns_hierarchical(self, similarity_matrix: np.ndarray,
                                     column_features: List[Dict]) -> List[List[int]]:
        """
        Cluster columns using SAME hierarchical clustering as files.
        """
        if len(column_features) <= 1:
            return [[0]] if column_features else []
        
        # Convert similarity to distance (SAME as files)
        distance_matrix = 1 - similarity_matrix
        
        # Perform hierarchical clustering (SAME as files)
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        
        try:
            linkage_matrix = linkage(squareform(distance_matrix), method='average')
            
            # Use SAME threshold logic as files (0.5 threshold = 0.5 distance)
            clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
            
            # Group by cluster
            cluster_dict = defaultdict(list)
            for idx, cluster_id in enumerate(clusters):
                cluster_dict[cluster_id].append(idx)
            
            result = list(cluster_dict.values())
            
            print(f"  Created {len(result)} column clusters (threshold: 0.5)")
            return result
            
        except Exception as e:
            print(f"  Clustering failed: {e}")
            # Fallback: group by high similarity
            return self._fallback_similarity_clustering(similarity_matrix)
    
    def _fallback_similarity_clustering(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Fallback clustering based on similarity matrix."""
        n = similarity_matrix.shape[0]
        clusters = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            current_cluster = [i]
            assigned.add(i)
            
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                if similarity_matrix[i, j] >= 0.5:  # SAME threshold as files
                    current_cluster.append(j)
                    assigned.add(j)
            
            clusters.append(current_cluster)
        
        return clusters
    
    def _create_golden_schema_from_clusters(self, column_clusters: List[List[int]],
                                           column_features: List[Dict]) -> Dict[str, Any]:
        """
        Create golden schema from column clusters.
        """
        golden_columns = []
        column_details = {}
        
        for cluster_idx, cluster_indices in enumerate(column_clusters, 1):
            if not cluster_indices:
                continue
            
            # Get all columns in this cluster
            cluster_cols = [column_features[i] for i in cluster_indices]
            
            # Create golden column
            golden_col = self._create_golden_column(cluster_cols, cluster_idx)
            
            if golden_col:
                golden_columns.append(golden_col['name'])
                column_details[golden_col['name']] = golden_col
        
        # Create golden schema
        golden_schema = {
            'name': f"Golden_Schema_{len(column_clusters)}",
            'total_columns': len(golden_columns),
            'columns': golden_columns,
            'column_details': column_details,
            'clusters_found': len(column_clusters)
        }
        
        print(f"  Created golden schema with {len(golden_columns)} columns")
        return golden_schema
    
    def _create_golden_column(self, cluster_cols: List[Dict], cluster_idx: int) -> Dict[str, Any]:
        """Create a golden column from a cluster."""
        if not cluster_cols:
            return None
        
        # Choose name: most common name or most representative
        column_names = [col['column'] for col in cluster_cols]
        
        # Find most common structural pattern
        def get_simple_pattern(name):
            # Simple pattern: count underscores
            return name.count('_')
        
        # Group by simple pattern
        pattern_groups = defaultdict(list)
        for name in column_names:
            pattern = get_simple_pattern(name)
            pattern_groups[pattern].append(name)
        
        # Choose from largest group
        largest_group = max(pattern_groups.values(), key=len, default=column_names)
        chosen_name = min(largest_group, key=len)  # Shortest in largest group
        
        # Calculate cluster statistics
        digit_ratios = [col['pattern_features'].get('digit_ratio', 0) for col in cluster_cols]
        unique_ratios = [col['pattern_features'].get('unique_ratio', 0) for col in cluster_cols]
        types = [col['type_features']['inferred_type'] for col in cluster_cols]
        
        # Most common type
        from collections import Counter
        type_counts = Counter(types)
        most_common_type = type_counts.most_common(1)[0][0] if type_counts else 'unknown'
        
        golden_col = {
            'name': chosen_name,
            'cluster_id': cluster_idx,
            'cluster_size': len(cluster_cols),
            'inferred_type': most_common_type,
            'statistics': {
                'avg_digit_ratio': np.mean(digit_ratios) if digit_ratios else 0,
                'avg_unique_ratio': np.mean(unique_ratios) if unique_ratios else 0,
                'digit_ratio_std': np.std(digit_ratios) if len(digit_ratios) > 1 else 0,
                'type_distribution': dict(type_counts)
            },
            'source_columns': [
                {
                    'file': col['file'],
                    'column': col['column'],
                    'type': col['type_features']['inferred_type'],
                    'digit_ratio': col['pattern_features'].get('digit_ratio', 0)
                }
                for col in cluster_cols
            ]
        }
        
        return golden_col
    
    def _create_mappings_from_clusters(self, column_clusters: List[List[int]],
                                      column_features: List[Dict],
                                      golden_schema: Dict[str, Any]) -> Dict[str, List]:
        """
        Create mappings from clusters.
        """
        mappings = {}
        
        # For each golden column
        for golden_col_name, golden_col_details in golden_schema['column_details'].items():
            cluster_idx = golden_col_details['cluster_id']
            
            if cluster_idx <= len(column_clusters):
                cluster_indices = column_clusters[cluster_idx - 1]
                mappings[golden_col_name] = []
                
                for idx in cluster_indices:
                    source_col = column_features[idx]
                    
                    # Calculate confidence based on similarity to cluster center
                    confidence = self._calculate_mapping_confidence(
                        source_col, 
                        golden_col_details,
                        column_features,
                        cluster_indices
                    )
                    
                    mapping = {
                        'source_file': source_col['file'],
                        'source_column': source_col['column'],
                        'golden_column': golden_col_name,
                        'confidence': confidence,
                        'confidence_level': 'HIGH' if confidence >= 0.7 else 
                                          'MEDIUM' if confidence >= 0.5 else 'LOW'
                    }
                    
                    mappings[golden_col_name].append(mapping)
        
        return mappings
    
    def _calculate_mapping_confidence(self, source_col: Dict, 
                                     golden_col: Dict,
                                     all_features: List[Dict],
                                     cluster_indices: List[int]) -> float:
        """Calculate mapping confidence."""
        # Calculate average similarity to other columns in cluster
        similarities = []
        source_idx = None
        
        # Find source column index
        for idx, col in enumerate(all_features):
            if (col['file'] == source_col['file'] and 
                col['column'] == source_col['column']):
                source_idx = idx
                break
        
        if source_idx is None:
            return 0.5
        
        # Calculate similarity to other columns in same cluster
        for other_idx in cluster_indices:
            if other_idx != source_idx and other_idx < len(all_features):
                other_col = all_features[other_idx]
                similarity = self._calculate_column_pair_similarity(source_col, other_col)
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            # Confidence is based on how similar this column is to others in its cluster
            confidence = min(1.0, avg_similarity * 1.2)  # Slight boost
        else:
            confidence = 0.5
        
        return confidence
    
    def map_cluster(self, cluster_files: List[str]) -> Dict[str, Any]:
        """
        Map cluster using same similarity approach as file clustering.
        """
        print(f"\n🔗 Processing {len(cluster_files)} files with column-level clustering")
        
        # Step 1: Extract column features (SAME as file features but for columns)
        column_features = self._extract_column_features(cluster_files)
        
        # Step 2: Calculate similarity matrix (SAME algorithm as file clustering)
        similarity_matrix = self._calculate_column_similarity_matrix(column_features)
        
        # Step 3: Cluster columns (SAME hierarchical clustering)
        column_clusters = self._cluster_columns_hierarchical(similarity_matrix, column_features)
        
        # Step 4: Create golden schema from clusters
        golden_schema = self._create_golden_schema_from_clusters(column_clusters, column_features)
        
        # Step 5: Create mappings
        mappings = self._create_mappings_from_clusters(column_clusters, column_features, golden_schema)
        
        return {
            'cluster_files': cluster_files,
            'column_features': {f['full_key']: f for f in column_features},
            'column_clusters': column_clusters,
            'golden_schema': golden_schema,
            'mappings': mappings
        }
    
    def map_all_clusters(self, clusters: List[List[str]]) -> Dict[str, Any]:
        """
        Map all clusters using SAME algorithm as file clustering.
        """
        print("\n" + "=" * 80)
        print("COLUMN-LEVEL CLUSTERING WITH SEMANTIC SIMILARITY")
        print("=" * 80)
        
        all_results = {}
        
        for cluster_idx, cluster_files in enumerate(clusters, 1):
            print(f"\n📊 Cluster {cluster_idx}/{len(clusters)}: {len(cluster_files)} files")
            
            result = self.map_cluster(cluster_files)
            all_results[f"Cluster_{cluster_idx}"] = result
        
        # Summary
        total_mappings = 0
        high_conf = 0
        
        for cluster_name, result in all_results.items():
            mappings = result['mappings']
            for mapping_list in mappings.values():
                total_mappings += len(mapping_list)
                high_conf += sum(1 for m in mapping_list 
                               if m.get('confidence_level') == 'HIGH')
        
        print(f"\n✅ Column clustering complete!")
        print(f"📊 Total mappings: {total_mappings}")
        print(f"🎯 High confidence mappings: {high_conf} ({high_conf/total_mappings:.1%})")
        
        return all_results
    






    























def print_golden_schemas_with_mappings(mapping_results: Dict[str, Any]):
    """
    Print golden schemas and their column mappings for each cluster.
    """
    print("\n" + "=" * 100)
    print("GOLDEN SCHEMAS WITH COLUMN MAPPINGS")
    print("=" * 100)
    
    for cluster_name, result in mapping_results.items():
        if 'golden_schema' not in result:
            continue
            
        golden_schema = result['golden_schema']
        mappings = result.get('mappings', {})
        
        print(f"\n{'━' * 80}")
        print(f"📦 {cluster_name}: {golden_schema['name']}")
        print(f"{'━' * 80}")
        
        # Print cluster files
        print(f"\n📁 Files in cluster:")
        for file in result.get('cluster_files', []):
            print(f"   • {file}")
        
        print(f"\n📊 Golden Schema has {len(golden_schema['columns'])} columns:")
        print(f"{'─' * 60}")
        
        # Print each golden column with its mappings
        for col_idx, col_name in enumerate(golden_schema['columns'], 1):
            col_details = golden_schema['column_details'].get(col_name, {})
            
            print(f"\n{col_idx:2d}. 🎯 {col_name}")
            print(f"    Type: {col_details.get('inferred_type', 'unknown')}")
            print(f"    Cluster size: {col_details.get('cluster_size', 0)} columns")
            
            # Print statistics if available
            stats = col_details.get('statistics', {})
            if stats:
                print(f"    Statistics: digit_ratio={stats.get('avg_digit_ratio', 0):.2f}, "
                      f"unique_ratio={stats.get('avg_unique_ratio', 0):.2f}")
            
            # Print mappings for this column
            col_mappings = mappings.get(col_name, [])
            if col_mappings:
                print(f"    🔗 Mapped from {len(col_mappings)} source columns:")
                
                # Group by confidence level for better display
                high_conf = [m for m in col_mappings if m.get('confidence_level') == 'HIGH']
                med_conf = [m for m in col_mappings if m.get('confidence_level') == 'MEDIUM']
                low_conf = [m for m in col_mappings if m.get('confidence_level') == 'LOW']
                
                if high_conf:
                    print(f"      🟢 HIGH confidence ({len(high_conf)}):")
                    for mapping in high_conf[:3]:  # Show first 3
                        print(f"        • {mapping['source_file']}.{mapping['source_column']} "
                              f"(score: {mapping['confidence']:.2f})")
                    if len(high_conf) > 3:
                        print(f"          ... and {len(high_conf) - 3} more")
                
                if med_conf:
                    print(f"      🟡 MEDIUM confidence ({len(med_conf)}):")
                    for mapping in med_conf[:2]:
                        print(f"        • {mapping['source_file']}.{mapping['source_column']} "
                              f"(score: {mapping['confidence']:.2f})")
                    if len(med_conf) > 2:
                        print(f"          ... and {len(med_conf) - 2} more")
                
                if low_conf:
                    print(f"      🔴 LOW confidence ({len(low_conf)}):")
                    for mapping in low_conf[:2]:
                        print(f"        • {mapping['source_file']}.{mapping['source_column']} "
                              f"(score: {mapping['confidence']:.2f})")
                    if len(low_conf) > 2:
                        print(f"          ... and {len(low_conf) - 2} more")
            else:
                print(f"    ⚠️  No mappings found for this column")
            
            # Print type distribution if available
            type_dist = col_details.get('statistics', {}).get('type_distribution', {})
            if type_dist and len(type_dist) > 1:
                print(f"    📋 Type variations: {', '.join([f'{k}({v})' for k, v in type_dist.items()])}")
        
        # Print summary for this cluster
        print(f"\n{'─' * 60}")
        
        total_mappings = sum(len(m) for m in mappings.values())
        high_mappings = sum(1 for m_list in mappings.values() for m in m_list 
                          if m.get('confidence_level') == 'HIGH')
        
        if total_mappings > 0:
            print(f"📈 Mapping Summary:")
            print(f"   • Total mappings: {total_mappings}")
            print(f"   • High confidence: {high_mappings} ({high_mappings/total_mappings:.1%})")
            print(f"   • Golden columns: {len(golden_schema['columns'])}")
        
        print(f"{'━' * 80}")

# Alternative: More detailed version with source column features
def print_detailed_golden_schemas(mapping_results: Dict[str, Any]):
    """
    Print detailed golden schemas with source column features.
    """
    print("\n" + "=" * 100)
    print("DETAILED GOLDEN SCHEMAS WITH SOURCE COLUMN ANALYSIS")
    print("=" * 100)
    
    for cluster_name, result in mapping_results.items():
        if 'golden_schema' not in result:
            continue
            
        golden_schema = result['golden_schema']
        column_features = result.get('column_features', {})
        mappings = result.get('mappings', {})
        
        print(f"\n{'=' * 80}")
        print(f"🏆 {cluster_name}: {golden_schema['name']}")
        print(f"{'=' * 80}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  • Files: {len(result.get('cluster_files', []))}")
        print(f"  • Golden columns: {len(golden_schema['columns'])}")
        print(f"  • Column clusters: {golden_schema.get('clusters_found', 0)}")
        
        # Print each golden column in detail
        for col_name in golden_schema['columns']:
            col_details = golden_schema['column_details'].get(col_name, {})
            col_mappings = mappings.get(col_name, [])
            
            print(f"\n{'─' * 60}")
            print(f"🎯 GOLDEN COLUMN: {col_name}")
            print(f"{'─' * 60}")
            
            # Golden column info
            print(f"📋 Golden Column Details:")
            print(f"  • Type: {col_details.get('inferred_type', 'unknown')}")
            print(f"  • Source columns: {col_details.get('cluster_size', 0)}")
            
            stats = col_details.get('statistics', {})
            if stats:
                print(f"  • Avg digit ratio: {stats.get('avg_digit_ratio', 0):.2f}")
                print(f"  • Avg unique ratio: {stats.get('avg_unique_ratio', 0):.2f}")
            
            # Mapped source columns
            if col_mappings:
                print(f"\n🔗 Mapped Source Columns:")
                
                # Sort by confidence
                col_mappings.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                
                for mapping in col_mappings:
                    source_key = f"{mapping['source_file']}__{mapping['source_column']}"
                    source_features = column_features.get(source_key, {})
                    
                    confidence_icon = "🟢" if mapping.get('confidence_level') == 'HIGH' else \
                                     "🟡" if mapping.get('confidence_level') == 'MEDIUM' else "🔴"
                    
                    print(f"\n  {confidence_icon} {mapping['source_file']}.{mapping['source_column']}")
                    print(f"    Confidence: {mapping['confidence']:.2f} ({mapping.get('confidence_level', 'UNKNOWN')})")
                    
                    # Show source column features if available
                    if source_features:
                        source_type = source_features.get('type_features', {}).get('inferred_type', 'unknown')
                        source_digit = source_features.get('pattern_features', {}).get('digit_ratio', 0)
                        
                        print(f"    Source type: {source_type}")
                        print(f"    Source digit ratio: {source_digit:.2f}")
                        
                        # Compare with golden
                        golden_digit = stats.get('avg_digit_ratio', 0)
                        if abs(source_digit - golden_digit) > 0.2:
                            print(f"    ⚠️  Digit ratio diff: {abs(source_digit - golden_digit):.2f}")
            
            # Source column details from cluster
            source_cols = col_details.get('source_columns', [])
            if source_cols:
                print(f"\n📋 Source Column Analysis:")
                
                # Group by file
                by_file = defaultdict(list)
                for src in source_cols:
                    by_file[src['file']].append(src)
                
                for file, cols in by_file.items():
                    print(f"  📁 {file}:")
                    for src in cols:
                        print(f"    • {src['column']} (type: {src.get('type', 'unknown')}, "
                              f"digit: {src.get('digit_ratio', 0):.2f})")
        
        print(f"\n{'=' * 80}")

# Alternative: Compact version for quick overview
def print_compact_golden_schemas(mapping_results: Dict[str, Any]):
    """
    Print compact overview of golden schemas and mappings.
    """
    print("\n" + "=" * 80)
    print("COMPACT GOLDEN SCHEMA OVERVIEW")
    print("=" * 80)
    
    for cluster_name, result in mapping_results.items():
        if 'golden_schema' not in result:
            continue
            
        golden_schema = result['golden_schema']
        mappings = result.get('mappings', {})
        
        print(f"\n📦 {cluster_name}: {golden_schema['name']}")
        print(f"   Files: {len(result.get('cluster_files', []))}")
        print(f"   Golden columns: {len(golden_schema['columns'])}")
        
        # Show top columns with most mappings
        columns_by_mappings = sorted(
            [(col, len(mappings.get(col, []))) for col in golden_schema['columns']],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5
        
        print(f"\n   Top columns by mappings:")
        for col_name, map_count in columns_by_mappings:
            col_details = golden_schema['column_details'].get(col_name, {})
            col_type = col_details.get('inferred_type', 'unknown')
            
            # Count confidence levels
            col_mappings = mappings.get(col_name, [])
            high = sum(1 for m in col_mappings if m.get('confidence_level') == 'HIGH')
            med = sum(1 for m in col_mappings if m.get('confidence_level') == 'MEDIUM')
            
            print(f"   • {col_name} ({col_type}): {map_count} mappings "
                  f"({high}🟢 {med}🟡 {map_count-high-med}🔴)")
        
        # Quick mapping examples
        print(f"\n   Example mappings:")
        for col_name in golden_schema['columns'][:3]:  # First 3 columns
            col_mappings = mappings.get(col_name, [])
            if col_mappings:
                # Get highest confidence mapping
                best_mapping = max(col_mappings, key=lambda x: x.get('confidence', 0))
                confidence_icon = "🟢" if best_mapping.get('confidence_level') == 'HIGH' else \
                                 "🟡" if best_mapping.get('confidence_level') == 'MEDIUM' else "🔴"
                
                print(f"   {confidence_icon} {col_name} ← "
                      f"{best_mapping['source_file']}.{best_mapping['source_column']} "
                      f"({best_mapping['confidence']:.2f})")
        
        print(f"{'─' * 60}")

# Usage
if __name__ == "__main__":
    import json
    
    # Load data
    with open('schema_data.json', 'r') as f:
        enhanced_schemas = json.load(f)
    
    with open('final_clustering_results.json', 'r') as f:
        clustering_results = json.load(f)
    
    clusters = clustering_results.get('clusters', [])
    
    # Initialize mapper
    mapper = ColumnClusteringMapper(enhanced_schemas)
    
    # Map all clusters
    results = mapper.map_all_clusters(clusters)
    print_golden_schemas_with_mappings(results)
    # Save results
    with open('column_clustered_mappings.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: column_clustered_mappings.json")
    print("=" * 80)
























































































# import json
# import numpy as np
# from typing import Dict, List, Any, Tuple
# from collections import defaultdict
# import pandas as pd

# class SchemaMapper:
#     def __init__(self, enhanced_schemas: Dict[str, Any], clustering_results: Dict[str, Any]):
#         """
#         Initialize schema mapper.
        
#         Args:
#             enhanced_schemas: From Phase 1 (complete schema data)
#             clustering_results: From Phase 3 (clustering results)
#         """
#         self.enhanced_schemas = enhanced_schemas
#         self.clustering_results = clustering_results
#         self.clusters = clustering_results.get('clusters', [])
#         self.file_names = clustering_results.get('file_names', [])
        
#         # Store mapping results
#         self.schema_mappings = {}
#         self.golden_schemas = {}
        
#     def generate_mappings_for_cluster(self, cluster: List[str]) -> Dict[str, Any]:
#         """
#         Generate schema mappings for a cluster of similar files.
        
#         Returns:
#             Dictionary with mapping rules and analysis
#         """
#         if len(cluster) < 2:
#             return {'error': 'Cluster must contain at least 2 files'}
        
#         print(f"\n🔧 Generating mappings for cluster with files: {cluster}")
        
#         # Get schema information for all files in cluster
#         cluster_schemas = {}
#         for filename in cluster:
#             if filename in self.enhanced_schemas:
#                 cluster_schemas[filename] = self.enhanced_schemas[filename]
        
#         # Step 1: Find the most representative schema (golden schema candidate)
#         golden_schema_candidate = self._find_golden_schema_candidate(cluster_schemas)
        
#         # Step 2: Create pairwise column mappings
#         pairwise_mappings = self._create_pairwise_mappings(cluster_schemas)
        
#         # Step 3: Create unified column mapping to golden schema
#         unified_mapping = self._create_unified_mapping(cluster_schemas, golden_schema_candidate)
        
#         # Step 4: Analyze schema variations
#         variations = self._analyze_schema_variations(cluster_schemas, unified_mapping)
        
#         # Step 5: Generate transformation rules
#         transformation_rules = self._generate_transformation_rules(cluster_schemas, unified_mapping)
        
#         # Step 6: Create golden schema definition
#         golden_schema = self._create_golden_schema(cluster_schemas, golden_schema_candidate, unified_mapping)
        
#         result = {
#             'cluster_files': cluster,
#             'golden_schema_candidate': golden_schema_candidate,
#             'pairwise_mappings': pairwise_mappings,
#             'unified_mapping': unified_mapping,
#             'schema_variations': variations,
#             'transformation_rules': transformation_rules,
#             'golden_schema': golden_schema,
#             'summary_stats': {
#                 'num_files': len(cluster),
#                 'num_columns_in_golden': len(golden_schema['columns']),
#                 'mapping_coverage': self._calculate_mapping_coverage(unified_mapping)
#             }
#         }
        
#         return result
    
#     def _find_golden_schema_candidate(self, cluster_schemas: Dict[str, Any]) -> str:
#         """
#         Find the most representative schema in the cluster.
        
#         Criteria:
#         1. Most common column count
#         2. Most complete data (fewest missing values)
#         3. Most descriptive column names
#         """
#         # Analyze each schema
#         schema_scores = {}
        
#         for filename, schema in cluster_schemas.items():
#             if 'error' in schema:
#                 continue
            
#             score = 0
#             columns = schema.get('columns', [])
            
#             # Criterion 1: Column count similarity to others
#             col_count = len(columns)
#             other_counts = [len(s.get('columns', [])) for s in cluster_schemas.values() 
#                           if 'error' not in s and s != schema]
            
#             if other_counts:
#                 avg_other = np.mean(other_counts)
#                 count_similarity = 1 - abs(col_count - avg_other) / max(col_count, avg_other, 1)
#                 score += 0.4 * count_similarity
            
#             # Criterion 2: Column name descriptiveness
#             descriptive_score = 0
#             for col in columns:
#                 # More tokens = more descriptive
#                 tokens = len(str(col).split('_'))
#                 descriptive_score += min(tokens / 3, 1.0)  # Cap at 3 tokens
            
#             if columns:
#                 score += 0.3 * (descriptive_score / len(columns))
            
#             # Criterion 3: Data completeness (based on inferred types)
#             type_score = 0
#             column_types = schema.get('column_types', {})
#             for col_type in column_types.values():
#                 if col_type != 'unknown':
#                     type_score += 1
            
#             if column_types:
#                 score += 0.3 * (type_score / len(column_types))
            
#             schema_scores[filename] = score
        
#         # Return schema with highest score
#         if schema_scores:
#             return max(schema_scores.items(), key=lambda x: x[1])[0]
        
#         # Fallback: first schema
#         return list(cluster_schemas.keys())[0]
    
#     def _create_pairwise_mappings(self, cluster_schemas: Dict[str, Any]) -> Dict[str, List]:
#         """
#         Create pairwise column mappings between all files in cluster.
#         """
#         pairwise_mappings = {}
#         filenames = list(cluster_schemas.keys())
        
#         for i in range(len(filenames)):
#             for j in range(i + 1, len(filenames)):
#                 file1 = filenames[i]
#                 file2 = filenames[j]
                
#                 if file1 not in cluster_schemas or file2 not in cluster_schemas:
#                     continue
                
#                 mapping = self._map_columns_between_files(
#                     cluster_schemas[file1], 
#                     cluster_schemas[file2]
#                 )
                
#                 pairwise_mappings[f"{file1}__{file2}"] = mapping
        
#         return pairwise_mappings
    
#     def _map_columns_between_files(self, schema1: Dict, schema2: Dict) -> List[Dict]:
#         """
#         Map columns between two schemas.
#         """
#         columns1 = schema1.get('columns', [])
#         columns2 = schema2.get('columns', [])
        
#         if not columns1 or not columns2:
#             return []
        
#         mappings = []
#         used_cols2 = set()
        
#         for col1 in columns1:
#             best_match = None
#             best_score = 0
            
#             for col2 in columns2:
#                 if col2 in used_cols2:
#                     continue
                
#                 # Calculate similarity score
#                 score = self._calculate_column_match_score(
#                     col1, col2, 
#                     schema1.get('column_types', {}).get(col1, 'unknown'),
#                     schema2.get('column_types', {}).get(col2, 'unknown'),
#                     schema1.get('value_patterns', {}).get(col1, {}),
#                     schema2.get('value_patterns', {}).get(col2, {})
#                 )
                
#                 if score > best_score and score > 0.3:  # Threshold
#                     best_score = score
#                     best_match = col2
            
#             if best_match:
#                 mappings.append({
#                     'source_column': col1,
#                     'target_column': best_match,
#                     'similarity_score': best_score,
#                     'confidence': self._score_to_confidence(best_score)
#                 })
#                 used_cols2.add(best_match)
        
#         return mappings
    
#     def _calculate_column_match_score(self, col1: str, col2: str, 
#                                      type1: str, type2: str,
#                                      patterns1: Dict, patterns2: Dict) -> float:
#         """
#         Calculate match score between two columns.
#         """
#         # 1. Name similarity (40%)
#         name_score = self._calculate_column_name_similarity(col1, col2)
        
#         # 2. Type compatibility (30%)
#         type_score = self._calculate_type_compatibility(type1, type2)
        
#         # 3. Pattern similarity (30%)
#         pattern_score = self._calculate_pattern_similarity(patterns1, patterns2)
        
#         # Weighted combination
#         return 0.4 * name_score + 0.3 * type_score + 0.3 * pattern_score
    
#     def _calculate_column_name_similarity(self, col1: str, col2: str) -> float:
#         """Calculate similarity between column names."""
#         col1_lower = col1.lower()
#         col2_lower = col2.lower()
        
#         if col1_lower == col2_lower:
#             return 1.0
        
#         # Edit distance
#         from difflib import SequenceMatcher
#         edit_sim = SequenceMatcher(None, col1_lower, col2_lower).ratio()
        
#         # Token overlap
#         tokens1 = set(col1_lower.split('_'))
#         tokens2 = set(col2_lower.split('_'))
        
#         if tokens1 and tokens2:
#             token_sim = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
#         else:
#             token_sim = 0.0
        
#         return 0.7 * edit_sim + 0.3 * token_sim
    
#     def _calculate_type_compatibility(self, type1: str, type2: str) -> float:
#         """Calculate type compatibility score."""
#         if type1 == type2:
#             return 1.0
#         elif (type1 in ['integer', 'float'] and type2 in ['integer', 'float']):
#             return 0.8
#         elif (type1 == 'string' and type2 != 'string') or (type2 == 'string' and type1 != 'string'):
#             return 0.3
#         else:
#             return 0.5
    
#     def _calculate_pattern_similarity(self, patterns1: Dict, patterns2: Dict) -> float:
#         """Calculate pattern similarity score."""
#         if not patterns1 or not patterns2:
#             return 0.5
        
#         similarities = []
        
#         # Compare digit ratio
#         digit1 = patterns1.get('digit_ratio', 0.5)
#         digit2 = patterns2.get('digit_ratio', 0.5)
#         similarities.append(1 - abs(digit1 - digit2))
        
#         # Compare unique ratio
#         unique1 = patterns1.get('unique_ratio', 0.5)
#         unique2 = patterns2.get('unique_ratio', 0.5)
#         similarities.append(1 - abs(unique1 - unique2))
        
#         # Compare length stats if available
#         len_stats1 = patterns1.get('length_stats', {})
#         len_stats2 = patterns2.get('length_stats', {})
        
#         if len_stats1 and len_stats2:
#             mean1 = len_stats1.get('mean', 0)
#             mean2 = len_stats2.get('mean', 0)
#             if mean1 > 0 or mean2 > 0:
#                 len_sim = 1 - abs(mean1 - mean2) / max(mean1, mean2, 1)
#                 similarities.append(len_sim)
        
#         return np.mean(similarities) if similarities else 0.5
    
#     def _score_to_confidence(self, score: float) -> str:
#         """Convert score to confidence level."""
#         if score >= 0.8:
#             return 'High'
#         elif score >= 0.6:
#             return 'Medium'
#         elif score >= 0.4:
#             return 'Low'
#         else:
#             return 'Very Low'
    
#     def _create_unified_mapping(self, cluster_schemas: Dict[str, Any], 
#                                golden_candidate: str) -> Dict[str, List]:
#         """
#         Create unified mapping from all schemas to golden schema.
#         """
#         if golden_candidate not in cluster_schemas:
#             return {}
        
#         golden_schema = cluster_schemas[golden_candidate]
#         golden_columns = golden_schema.get('columns', [])
        
#         unified_mapping = {col: [] for col in golden_columns}
        
#         for filename, schema in cluster_schemas.items():
#             if filename == golden_candidate or 'error' in schema:
#                 continue
            
#             # Map this schema's columns to golden schema
#             mapping = self._map_columns_between_files(schema, golden_schema)
            
#             for map_item in mapping:
#                 golden_col = map_item['target_column']
#                 if golden_col in unified_mapping:
#                     unified_mapping[golden_col].append({
#                         'source_file': filename,
#                         'source_column': map_item['source_column'],
#                         'similarity_score': map_item['similarity_score'],
#                         'confidence': map_item['confidence']
#                     })
        
#         return unified_mapping
    
#     def _analyze_schema_variations(self, cluster_schemas: Dict[str, Any],
#                                   unified_mapping: Dict[str, List]) -> Dict[str, Any]:
#         """
#         Analyze variations between schemas in the cluster.
#         """
#         variations = {
#             'naming_variations': defaultdict(list),
#             'type_variations': defaultdict(list),
#             'missing_columns': defaultdict(list),
#             'extra_columns': defaultdict(list)
#         }
        
#         golden_candidate = self._find_golden_schema_candidate(cluster_schemas)
#         golden_schema = cluster_schemas[golden_candidate]
#         golden_columns = golden_schema.get('columns', [])
        
#         for filename, schema in cluster_schemas.items():
#             if filename == golden_candidate or 'error' in schema:
#                 continue
            
#             schema_columns = set(schema.get('columns', []))
#             golden_columns_set = set(golden_columns)
            
#             # Find missing columns (in golden but not in this schema)
#             missing = golden_columns_set - schema_columns
#             if missing:
#                 variations['missing_columns'][filename] = list(missing)
            
#             # Find extra columns (in this schema but not in golden)
#             extra = schema_columns - golden_columns_set
#             if extra:
#                 variations['extra_columns'][filename] = list(extra)
            
#             # Analyze naming variations for mapped columns
#             for golden_col, mappings in unified_mapping.items():
#                 for mapping in mappings:
#                     if mapping['source_file'] == filename:
#                         source_col = mapping['source_column']
#                         if source_col != golden_col:
#                             variations['naming_variations'][golden_col].append({
#                                 'file': filename,
#                                 'source_name': source_col,
#                                 'similarity': mapping['similarity_score']
#                             })
            
#             # Analyze type variations
#             schema_types = schema.get('column_types', {})
#             golden_types = golden_schema.get('column_types', {})
            
#             for col in schema_columns.intersection(golden_columns_set):
#                 type1 = schema_types.get(col, 'unknown')
#                 type2 = golden_types.get(col, 'unknown')
                
#                 if type1 != type2:
#                     variations['type_variations'][col].append({
#                         'file': filename,
#                         'type': type1,
#                         'golden_type': type2
#                     })
        
#         return variations
    
#     def _generate_transformation_rules(self, cluster_schemas: Dict[str, Any],
#                                       unified_mapping: Dict[str, List]) -> Dict[str, Any]:
#         """
#         Generate transformation rules for data unification.
#         """
#         transformation_rules = {
#             'column_mappings': {},
#             'type_conversions': {},
#             'value_transformations': {},
#             'default_values': {}
#         }
        
#         golden_candidate = self._find_golden_schema_candidate(cluster_schemas)
#         golden_schema = cluster_schemas[golden_candidate]
        
#         for golden_col, mappings in unified_mapping.items():
#             transformation_rules['column_mappings'][golden_col] = []
            
#             for mapping in mappings:
#                 source_file = mapping['source_file']
#                 source_col = mapping['source_column']
                
#                 rule = {
#                     'source_file': source_file,
#                     'source_column': source_col,
#                     'target_column': golden_col,
#                     'confidence': mapping['confidence'],
#                     'transformation_type': 'direct_mapping'
#                 }
                
#                 # Check if type conversion is needed
#                 source_schema = cluster_schemas[source_file]
#                 source_type = source_schema.get('column_types', {}).get(source_col, 'unknown')
#                 golden_type = golden_schema.get('column_types', {}).get(golden_col, 'unknown')
                
#                 if source_type != golden_type:
#                     rule['transformation_type'] = 'type_conversion'
#                     rule['type_conversion'] = f"{source_type} → {golden_type}"
                
#                 # Check for naming pattern that might need transformation
#                 source_patterns = source_schema.get('value_patterns', {}).get(source_col, {})
#                 golden_patterns = golden_schema.get('value_patterns', {}).get(golden_col, {})
                
#                 if source_patterns and golden_patterns:
#                     # Check if pattern suggests format conversion
#                     source_sep_patterns = source_patterns.get('separator_patterns', {})
#                     golden_sep_patterns = golden_patterns.get('separator_patterns', {})
                    
#                     if (source_sep_patterns and golden_sep_patterns and 
#                         source_sep_patterns != golden_sep_patterns):
#                         rule['notes'] = 'May require format conversion'
                
#                 transformation_rules['column_mappings'][golden_col].append(rule)
        
#         return transformation_rules
    
#     def _create_golden_schema(self, cluster_schemas: Dict[str, Any],
#                              golden_candidate: str, 
#                              unified_mapping: Dict[str, List]) -> Dict[str, Any]:
#         """
#         Create golden schema definition for the cluster.
#         """
#         golden_schema = cluster_schemas[golden_candidate]
        
#         # Enhanced golden schema with metadata
#         enhanced_golden = {
#             'name': f"Golden_Schema_Cluster_{len(self.schema_mappings) + 1}",
#             'source_files': list(cluster_schemas.keys()),
#             'representative_file': golden_candidate,
#             'columns': [],
#             'column_metadata': {},
#             'coverage_stats': {}
#         }
        
#         # Add column information
#         for col in golden_schema.get('columns', []):
#             col_info = {
#                 'name': col,
#                 'data_type': golden_schema.get('column_types', {}).get(col, 'unknown'),
#                 'value_patterns': golden_schema.get('value_patterns', {}).get(col, {}),
#                 'mapped_from': unified_mapping.get(col, [])
#             }
            
#             # Calculate coverage statistics
#             if col in unified_mapping:
#                 mappings = unified_mapping[col]
#                 coverage = len(mappings) / (len(cluster_schemas) - 1)  # Exclude golden
#                 confidence_scores = [m['similarity_score'] for m in mappings]
                
#                 col_info['coverage'] = {
#                     'percentage': coverage,
#                     'num_sources': len(mappings),
#                     'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
#                     'is_common': coverage > 0.5  # Appears in >50% of sources
#                 }
            
#             enhanced_golden['columns'].append(col)
#             enhanced_golden['column_metadata'][col] = col_info
        
#         # Calculate overall statistics
#         total_cols = len(enhanced_golden['columns'])
#         if total_cols > 0:
#             common_cols = sum(1 for col_info in enhanced_golden['column_metadata'].values()
#                             if col_info.get('coverage', {}).get('is_common', False))
            
#             enhanced_golden['coverage_stats'] = {
#                 'total_columns': total_cols,
#                 'common_columns': common_cols,
#                 'common_percentage': common_cols / total_cols if total_cols > 0 else 0,
#                 'unique_columns_per_source': total_cols - common_cols
#             }
        
#         return enhanced_golden
    
#     def _calculate_mapping_coverage(self, unified_mapping: Dict[str, List]) -> Dict[str, float]:
#         """
#         Calculate mapping coverage statistics.
#         """
#         if not unified_mapping:
#             return {}
        
#         total_mappings = sum(len(mappings) for mappings in unified_mapping.values())
#         total_possible = len(unified_mapping) * (len(self.clusters) - 1)  # Rough estimate
        
#         return {
#             'mapping_density': total_mappings / total_possible if total_possible > 0 else 0,
#             'avg_mappings_per_column': total_mappings / len(unified_mapping) if unified_mapping else 0
#         }
    
#     def generate_all_mappings(self) -> Dict[str, Any]:
#         """
#         Generate schema mappings for all clusters.
#         """
#         print("\n" + "=" * 80)
#         print("GENERATING SCHEMA MAPPINGS")
#         print("=" * 80)
        
#         all_mappings = {}
        
#         for cluster_idx, cluster in enumerate(self.clusters, 1):
#             print(f"\n📋 Processing Cluster {cluster_idx}/{len(self.clusters)}...")
            
#             cluster_mappings = self.generate_mappings_for_cluster(cluster)
#             all_mappings[f"Cluster_{cluster_idx}"] = cluster_mappings
            
#             # Store in instance
#             self.schema_mappings[f"Cluster_{cluster_idx}"] = cluster_mappings
            
#             # Store golden schema
#             if 'golden_schema' in cluster_mappings:
#                 self.golden_schemas[f"Cluster_{cluster_idx}"] = cluster_mappings['golden_schema']
        
#         # Generate overall summary
#         summary = self._generate_overall_summary(all_mappings)
        
#         result = {
#             'all_mappings': all_mappings,
#             'golden_schemas': self.golden_schemas,
#             'summary': summary
#         }
        
#         print("\n✅ Schema mapping generation complete!")
#         print(f"📊 Generated mappings for {len(self.clusters)} clusters")
#         print(f"🎯 Created {len(self.golden_schemas)} golden schemas")
        
#         return result
    
#     def _generate_overall_summary(self, all_mappings: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Generate overall summary of schema unification.
#         """
#         total_files = len(self.file_names)
#         total_clusters = len(all_mappings)
        
#         # Calculate statistics
#         total_columns_in_golden = 0
#         common_columns_total = 0
#         high_confidence_mappings = 0
#         total_mappings = 0
        
#         for cluster_name, mappings in all_mappings.items():
#             golden_schema = mappings.get('golden_schema', {})
#             total_columns_in_golden += len(golden_schema.get('columns', []))
            
#             coverage_stats = golden_schema.get('coverage_stats', {})
#             common_columns_total += coverage_stats.get('common_columns', 0)
            
#             # Count mappings by confidence
#             unified_mapping = mappings.get('unified_mapping', {})
#             for col_mappings in unified_mapping.values():
#                 for mapping in col_mappings:
#                     total_mappings += 1
#                     if mapping.get('confidence') == 'High':
#                         high_confidence_mappings += 1
        
#         summary = {
#             'overall_stats': {
#                 'total_files': total_files,
#                 'total_clusters': total_clusters,
#                 'total_golden_columns': total_columns_in_golden,
#                 'common_columns_across_sources': common_columns_total,
#                 'mapping_quality': {
#                     'total_mappings': total_mappings,
#                     'high_confidence_mappings': high_confidence_mappings,
#                     'high_confidence_percentage': high_confidence_mappings / total_mappings if total_mappings > 0 else 0
#                 }
#             },
#             'cluster_summary': {},
#             'recommendations': self._generate_recommendations(all_mappings)
#         }
        
#         # Add per-cluster summary
#         for cluster_name, mappings in all_mappings.items():
#             golden_schema = mappings.get('golden_schema', {})
#             summary['cluster_summary'][cluster_name] = {
#                 'num_files': len(mappings.get('cluster_files', [])),
#                 'num_columns': len(golden_schema.get('columns', [])),
#                 'common_columns': golden_schema.get('coverage_stats', {}).get('common_columns', 0),
#                 'representative_file': mappings.get('golden_schema_candidate', 'Unknown')
#             }
        
#         return summary
    
#     def _generate_recommendations(self, all_mappings: Dict[str, Any]) -> List[str]:
#         """
#         Generate recommendations based on schema analysis.
#         """
#         recommendations = []
        
#         for cluster_name, mappings in all_mappings.items():
#             variations = mappings.get('schema_variations', {})
#             cluster_files = mappings.get('cluster_files', [])
            
#             # Check for naming variations
#             naming_vars = variations.get('naming_variations', {})
#             if naming_vars:
#                 for col, var_list in naming_vars.items():
#                     if len(var_list) > 1:
#                         recommendations.append(
#                             f"Cluster {cluster_name}: Standardize column naming for '{col}' "
#                             f"across {len(var_list)} variations"
#                         )
            
#             # Check for type variations
#             type_vars = variations.get('type_variations', {})
#             if type_vars:
#                 for col, var_list in type_vars.items():
#                     if len(var_list) > 0:
#                         recommendations.append(
#                             f"Cluster {cluster_name}: Resolve data type inconsistencies for '{col}'"
#                         )
            
#             # Check for missing columns
#             missing_cols = variations.get('missing_columns', {})
#             for file, missing in missing_cols.items():
#                 if missing:
#                     recommendations.append(
#                         f"Cluster {cluster_name}: File '{file}' is missing columns: {missing[:3]}..."
#                     )
        
#         # Add general recommendations
#         if len(all_mappings) > 0:
#             recommendations.append(
#                 f"Consider creating {len(all_mappings)} unified tables for the identified entities"
#             )
#             recommendations.append(
#                 "Implement data validation rules based on discovered patterns"
#             )
        
#         return recommendations[:10]  # Return top 10 recommendations
    
#     def save_mappings(self, output_file: str = 'schema_mappings.json'):
#         """
#         Save schema mappings to JSON file.
#         """
#         results = {
#             'schema_mappings': self.schema_mappings,
#             'golden_schemas': self.golden_schemas,
#             'metadata': {
#                 'total_files': len(self.file_names),
#                 'total_clusters': len(self.clusters),
#                 'generation_timestamp': pd.Timestamp.now().isoformat()
#             }
#         }
        
#         with open(output_file, 'w') as f:
#             json.dump(results, f, indent=2, default=str)
        
#         print(f"💾 Schema mappings saved to {output_file}")
    
#     def generate_report(self, output_file: str = 'schema_unification_report.txt'):
#         """
#         Generate human-readable report.
#         """
#         with open(output_file, 'w') as f:
#             f.write("=" * 80 + "\n")
#             f.write("SCHEMA UNIFICATION REPORT\n")
#             f.write("=" * 80 + "\n\n")
            
#             f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Total Files Analyzed: {len(self.file_names)}\n")
#             f.write(f"Identified Clusters: {len(self.clusters)}\n\n")
            
#             # Summary by cluster
#             f.write("CLUSTER SUMMARY:\n")
#             f.write("-" * 40 + "\n")
            
#             for cluster_idx, cluster in enumerate(self.clusters, 1):
#                 f.write(f"\nCluster {cluster_idx}:\n")
#                 f.write(f"  Files: {len(cluster)}\n")
#                 for file in sorted(cluster):
#                     f.write(f"    - {file}\n")
                
#                 if f"Cluster_{cluster_idx}" in self.schema_mappings:
#                     mappings = self.schema_mappings[f"Cluster_{cluster_idx}"]
#                     golden_schema = mappings.get('golden_schema', {})
                    
#                     f.write(f"  Golden Schema: {golden_schema.get('name', 'N/A')}\n")
#                     f.write(f"  Representative: {mappings.get('golden_schema_candidate', 'N/A')}\n")
#                     f.write(f"  Columns: {len(golden_schema.get('columns', []))}\n")
                    
#                     # Show common columns
#                     coverage_stats = golden_schema.get('coverage_stats', {})
#                     common_cols = coverage_stats.get('common_columns', 0)
#                     total_cols = coverage_stats.get('total_columns', 0)
                    
#                     if total_cols > 0:
#                         f.write(f"  Common Columns: {common_cols}/{total_cols} ({common_cols/total_cols:.1%})\n")
            
#             # Mapping details
#             f.write("\n\nDETAILED MAPPINGS:\n")
#             f.write("=" * 40 + "\n")
            
#             for cluster_name, mappings in self.schema_mappings.items():
#                 f.write(f"\n{cluster_name}:\n")
                
#                 golden_schema = mappings.get('golden_schema', {})
#                 unified_mapping = mappings.get('unified_mapping', {})
                
#                 for golden_col, col_info in golden_schema.get('column_metadata', {}).items():
#                     f.write(f"\n  {golden_col}:\n")
#                     f.write(f"    Type: {col_info.get('data_type', 'unknown')}\n")
                    
#                     mappings_list = unified_mapping.get(golden_col, [])
#                     if mappings_list:
#                         f.write(f"    Mapped from:\n")
#                         for mapping in mappings_list:
#                             f.write(f"      - {mapping['source_file']}.{mapping['source_column']} "
#                                    f"(confidence: {mapping['confidence']})\n")
            
#             # Recommendations
#             f.write("\n\nRECOMMENDATIONS:\n")
#             f.write("=" * 40 + "\n")
            
#             for cluster_name, mappings in self.schema_mappings.items():
#                 variations = mappings.get('schema_variations', {})
                
#                 # Naming variations
#                 naming_vars = variations.get('naming_variations', {})
#                 if naming_vars:
#                     f.write(f"\n{cluster_name} - Naming Standardizations:\n")
#                     for col, var_list in naming_vars.items():
#                         if var_list:
#                             f.write(f"  - {col}: {len(var_list)} variations\n")
                
#                 # Type variations
#                 type_vars = variations.get('type_variations', {})
#                 if type_vars:
#                     f.write(f"\n{cluster_name} - Type Resolutions:\n")
#                     for col, var_list in type_vars.items():
#                         if var_list:
#                             types = set(v['type'] for v in var_list)
#                             f.write(f"  - {col}: {', '.join(types)} → {var_list[0]['golden_type']}\n")
        
#         print(f"📄 Report generated: {output_file}")

#     def print_golden_schemas_detailed(self,golden_schemas: Dict[str, Any]):
#             """Print detailed golden schemas with mapped columns."""
#             print("\n" + "=" * 80)
#             print("DETAILED GOLDEN SCHEMAS")
#             print("=" * 80)
            
#             for cluster_name, golden_schema in golden_schemas.items():
#                 print(f"\n{'━' * 60}")
#                 print(f"📁 {golden_schema['name']}")
#                 print(f"{'━' * 60}")
                
#                 print(f"\n📊 Overview:")
#                 print(f"  • Representative: {golden_schema['representative_file']}")
#                 print(f"  • Total source files: {len(golden_schema['source_files'])}")
#                 print(f"  • Total columns: {len(golden_schema['columns'])}")
                
#                 coverage_stats = golden_schema.get('coverage_stats', {})
#                 if coverage_stats:
#                     print(f"  • Common columns: {coverage_stats.get('common_columns', 0)}")
#                     print(f"  • Common percentage: {coverage_stats.get('common_percentage', 0):.1%}")
                
#                 print(f"\n📋 Column Details:")
#                 print(f"{'─' * 60}")
                
#                 for col_name, col_info in golden_schema['column_metadata'].items():
#                     print(f"\n  ▸ {col_name}")
#                     print(f"    Type: {col_info['data_type']}")
                    
#                     # Coverage info
#                     coverage = col_info.get('coverage', {})
#                     if coverage:
#                         status = "✓ COMMON" if coverage.get('is_common', False) else "⚠️ PARTIAL"
#                         print(f"    Status: {status} ({coverage.get('num_sources', 0)} sources)")
#                         print(f"    Avg confidence: {coverage.get('avg_confidence', 0):.2f}")
                    
#                     # Mapped columns
#                     mapped_from = col_info.get('mapped_from', [])
#                     if mapped_from:
#                         print(f"    Mapped from:")
#                         for mapping in mapped_from:
#                             confidence_icon = "🟢" if mapping['confidence'] == 'High' else \
#                                             "🟡" if mapping['confidence'] == 'Medium' else \
#                                             "🔴"
#                             print(f"      {confidence_icon} {mapping['source_file']}.{mapping['source_column']} "
#                                 f"(score: {mapping['similarity_score']:.2f})")
#                     else:
#                         print(f"    ❗ Unique column - no direct mappings found")
                    
#                     # Value patterns if available
#                     patterns = col_info.get('value_patterns', {})
#                     if patterns and 'digit_ratio' in patterns:
#                         print(f"    Pattern: digit ratio={patterns['digit_ratio']:.2f}, "
#                             f"unique ratio={patterns.get('unique_ratio', 0):.2f}")
                
#                 print(f"\n📎 Source files in this cluster:")
#                 for file in sorted(golden_schema['source_files']):
#                     print(f"   • {file}")
            
#             print("\n" + "=" * 80)

# def main_schema_mapper(clustering_results):

#     # Load data from previous phases
#     print("📂 Loading data from previous phases...")
    
#     # Load enhanced schemas (Phase 1)
#     with open('schema_data.json', 'r') as f:
#         enhanced_schemas = json.load(f)
    
#     # Load clustering results (Phase 3)
#     with open('final_clustering_results.json', 'r') as f:
#         clustering_results = json.load(f)
    
#     # Initialize schema mapper
#     mapper = SchemaMapper(enhanced_schemas, clustering_results)
    
#     # Generate all mappings
#     mapping_results = mapper.generate_all_mappings()
    
#     # Save results
#     mapper.save_mappings('schema_mappings.json')
#     mapper.generate_report('schema_unification_report.txt')
    
#     # # Print summary
#     # print("\n" + "=" * 80)
#     # print("SCHEMA MAPPING SUMMARY")
#     # print("=" * 80)
    
#     # summary = mapping_results['summary']
#     # overall_stats = summary['overall_stats']
    
#     # print(f"\n📊 Overall Statistics:")
#     # print(f"   Total files: {overall_stats['total_files']}")
#     # print(f"   Clusters identified: {overall_stats['total_clusters']}")
#     # print(f"   Total golden columns: {overall_stats['total_golden_columns']}")
    
#     # mapping_quality = overall_stats['mapping_quality']
#     # print(f"   Mapping quality: {mapping_quality['high_confidence_percentage']:.1%} high confidence")
    
#     # print(f"\n🏆 Golden Schemas Created:")
#     # for cluster_name, golden_schema in mapper.golden_schemas.items():
#     #     print(f"   {cluster_name}: {golden_schema['name']}")
#     #     print(f"     Columns: {len(golden_schema['columns'])}")
#     #     print(f"     Source files: {len(golden_schema['source_files'])}")
    
#     # print(f"\n💡 Key Recommendations:")
#     # recommendations = summary['recommendations']
#     # for i, rec in enumerate(recommendations[:3], 1):
#     #     print(f"   {i}. {rec}")
    
#     # print("\n✅ Phase 4 complete! Files generated:")
#     # print("   - schema_mappings.json (detailed mappings)")
#     # print("   - schema_unification_report.txt (human-readable report)")
#     # print("=" * 80)

# # Main execution
# if __name__ == "__main__":
#      # Load data from previous phases
#     print("📂 Loading data from previous phases...")
    
#     # Load enhanced schemas (Phase 1)
#     with open('schema_data.json', 'r') as f:
#         enhanced_schemas = json.load(f)
    
#     # Load clustering results (Phase 3)
#     with open('final_clustering_results.json', 'r') as f:
#         clustering_results = json.load(f)
    
#     # Initialize schema mapper
#     mapper = SchemaMapper(enhanced_schemas, clustering_results)
    
#     # Generate all mappings
#     mapping_results = mapper.generate_all_mappings()
    
#     # Save results
#     mapper.save_mappings('schema_mappings.json')
#     mapper.generate_report('schema_unification_report.txt')
    
#     mapper.print_golden_schemas_detailed(mapper.golden_schemas)