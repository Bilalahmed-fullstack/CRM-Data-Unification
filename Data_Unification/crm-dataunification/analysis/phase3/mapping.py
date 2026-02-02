import numpy as np
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict
from difflib import SequenceMatcher
import json
import fasttext
import fasttext.util
import numpy as np
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class DataDrivenColumnMapper:
    def __init__(self, enhanced_schemas: Dict[str, Any]):
        """
        Completely data-driven column mapping with NO hardcoded assumptions.
        """
        self.enhanced_schemas = enhanced_schemas
        
        # Learn patterns from the data itself
        self.learned_patterns = self._learn_patterns_from_data()
        
        # Optional: Load word embeddings if available
        self.word_embeddings = None
        try:
            # import gensim.downloader as api
            # print("📥 Loading word embeddings for semantic similarity...")
            # self.word_embeddings = api.load('glove-wiki-gigaword-50')
            # print(f"✅ Word embeddings loaded!")

            # fasttext.util.download_model('en', if_exists='ignore')
            # self.word_embeddings = fasttext.load_model('cc.en.300.bin')

            model_name='all-MiniLM-L6-v2'
            self.word_embeddings = SentenceTransformer(model_name)
        except:
            print("⚠️ Word embeddings not available")
    
    def _learn_patterns_from_data(self) -> Dict[str, Any]:
        """
        Learn patterns from ALL schemas to understand what's typical.
        """
        print("🧠 Learning patterns from data...")
        
        all_column_names = []
        all_column_types = []
        all_patterns = []
        common_suffixes = {}
        
        for filename, schema in self.enhanced_schemas.items():
            columns = schema.get('columns', [])
            column_types = schema.get('column_types', {})
            value_patterns = schema.get('value_patterns', {})
            
            all_column_names.extend(columns)
            suffixes = self._learn_common_suffixes(columns)
            for key in suffixes.keys():
                if key not in common_suffixes.keys():
                    common_suffixes[key] = suffixes[key]

            for col_name, col_type in column_types.items():
                all_column_types.append(col_type)
                
                if col_name in value_patterns:
                    pattern = value_patterns[col_name]
                    pattern['name'] = col_name
                    pattern['type'] = col_type
                    all_patterns.append(pattern)
        
        # Learn common suffixes from column names
        # common_suffixes = self._learn_common_suffixes(all_column_names)
        
        # Learn typical patterns for each type
        type_patterns = self._learn_type_patterns(all_patterns)
        
        # Learn what makes a column "ID-like" from data
        id_like_patterns = self._learn_id_patterns(all_patterns)
        
        return {
            'common_suffixes': common_suffixes,
            'type_patterns': type_patterns,
            'id_like_patterns': id_like_patterns,
            'column_count': len(all_column_names),
            'type_distribution': dict(Counter(all_column_types))
        }
    
    def _learn_common_suffixes(self, column_names: List[str]) -> Dict[str, float]:
        """
        Learn common suffixes from column names in the data.
        """
        suffix_counter = Counter()
        total_names = len(column_names)
        
        for name in column_names:
            # Split by common separators
            parts = name.lower().replace('_', ' ').replace('-', ' ').split()
            
            # The last part is often the suffix
            if len(parts) > 1:
                suffix = parts[-1]
                if len(suffix) > 1:  # Ignore single letters
                    suffix_counter[suffix] += 1
        
        # Convert to frequencies
        common_suffixes = {}
        for suffix, count in suffix_counter.items():
            if count / total_names >= 0.1:  # At least 5% frequency
                common_suffixes[suffix] = count / total_names
        
        print(f"  Learned {len(common_suffixes)} common suffixes")
        print(common_suffixes)
        return common_suffixes
    
    def _learn_type_patterns(self, all_patterns: List[Dict]) -> Dict[str, Dict]:
        """
        Learn typical patterns for each data type from actual data.
        """
        type_patterns = defaultdict(list)
        
        for pattern in all_patterns:
            col_type = pattern.get('type', 'unknown')
            type_patterns[col_type].append(pattern)
        
        # Calculate statistics for each type
        learned_patterns = {}
        for col_type, patterns in type_patterns.items():
            if patterns:
                digit_ratios = [p.get('digit_ratio', 0) for p in patterns]
                unique_ratios = [p.get('unique_ratio', 0) for p in patterns]
                
                learned_patterns[col_type] = {
                    'avg_digit_ratio': np.mean(digit_ratios),
                    'std_digit_ratio': np.std(digit_ratios) if len(digit_ratios) > 1 else 0,
                    'avg_unique_ratio': np.mean(unique_ratios),
                    'std_unique_ratio': np.std(unique_ratios) if len(unique_ratios) > 1 else 0,
                    'sample_count': len(patterns)
                }
        
        return learned_patterns
    
    def _learn_id_patterns(self, all_patterns: List[Dict]) -> Dict[str, Any]:
        """
        Learn what patterns indicate an ID-like column from the data.
        """
        if not all_patterns:
            return {}
        
        # Analyze patterns to learn what makes a column ID-like
        digit_ratios = [p.get('digit_ratio', 0) for p in all_patterns]
        unique_ratios = [p.get('unique_ratio', 0) for p in all_patterns]
        
        # Learn thresholds from data distribution
        digit_threshold = np.percentile(digit_ratios, 75)  # Top 25%
        unique_threshold = np.percentile(unique_ratios, 75)  # Top 25%
        
        return {
            'digit_threshold': float(digit_threshold),
            'unique_threshold': float(unique_threshold),
            'avg_digit_ratio': float(np.mean(digit_ratios)),
            'avg_unique_ratio': float(np.mean(unique_ratios))
        }
    
    def map_all_clusters(self, clusters: List[List[str]]) -> Dict[str, Any]:
        """
        Create data-driven one-to-one column mappings.
        """
        print("\n" + "=" * 80)
        print("DATA-DRIVEN COLUMN MAPPING")
        print("=" * 80)
        
        all_results = {}
        
        for cluster_idx, cluster_files in enumerate(clusters, 1):
            print(f"\n📊 Processing Cluster {cluster_idx}: {len(cluster_files)} schemas")
            
            result = self._create_data_driven_mappings(cluster_files)
            all_results[f"Cluster_{cluster_idx}"] = result
            
            # # Print immediate feedback
            # self._print_data_driven_summary(result, cluster_idx)
        
        print(f"\n{'=' * 80}")
        print("✅ DATA-DRIVEN MAPPING COMPLETE")
        print(f"{'=' * 80}")
        
        return all_results
    
    def _create_data_driven_mappings(self, cluster_files: List[str]) -> Dict[str, Any]:
        """
        Create mappings using only learned patterns from data.
        """
        if len(cluster_files) < 2:
            return self._handle_single_schema_data_driven(cluster_files)
        
        # Extract data-driven features
        schema_features = {}
        for filename in cluster_files:
            if filename in self.enhanced_schemas:
                schema_features[filename] = self._extract_data_driven_features(filename)
        
        schemas = list(schema_features.keys())
        
        # Calculate similarity matrices
        similarity_matrices = {}
        for i in range(len(schemas)):
            for j in range(i + 1, len(schemas)):
                schema1 = schemas[i]
                schema2 = schemas[j]
                
                sim_matrix = self._calculate_data_driven_similarity(
                    schema_features[schema1], 
                    schema_features[schema2]
                )
                
                similarity_matrices[(schema1, schema2)] = sim_matrix
                similarity_matrices[(schema2, schema1)] = sim_matrix.T
        
        # Find the most representative schema as reference
        reference_schema = self._find_reference_schema(schema_features)
        
        # Create mappings starting from reference
        column_groups = self._create_column_groups_from_reference(
            reference_schema, schema_features, similarity_matrices
        )
        
        # Validate and refine groups
        validated_groups = self._validate_column_groups(
            column_groups, schema_features, similarity_matrices
        )
        
        # Create golden schema
        golden_schema = self._create_data_driven_golden_schema(validated_groups, schema_features)
        
        # Create mapping matrix
        mapping_matrix = self._create_data_driven_mapping_matrix(
            validated_groups, schema_features, golden_schema
        )
        
        return {
            'cluster_files': cluster_files,
            'reference_schema': reference_schema,
            'validated_groups': validated_groups,
            'golden_schema': golden_schema,
            'mapping_matrix': mapping_matrix,
            'quality_metrics': self._calculate_data_driven_quality(validated_groups, schema_features)
        }
    
    def _extract_data_driven_features(self, filename: str) -> Dict[str, Dict]:
        """
        Extract features using only learned patterns from data.
        """
        if filename not in self.enhanced_schemas:
            return {}
        
        schema = self.enhanced_schemas[filename]
        columns = schema.get('columns', [])
        column_types = schema.get('column_types', {})
        value_patterns = schema.get('value_patterns', {})
        value_samples = schema.get('value_samples', {})
        
        column_features = {}
        
        for col_name in columns:
            col_type = column_types.get(col_name, 'unknown')
            patterns = value_patterns.get(col_name, {})
            samples = value_samples.get(col_name, [])
            
            # Analyze the column name based on learned patterns
            name_analysis = self._analyze_column_name(col_name)
            
            # Analyze data patterns
            data_analysis = self._analyze_data_patterns(patterns, samples, col_type)
            
            features = {
                'name': col_name,
                'type': col_type,
                'patterns': patterns,
                'samples': samples[:5],
                'name_analysis': name_analysis,
                'data_analysis': data_analysis,
                'position_index': columns.index(col_name),
                'total_columns': len(columns),
                'similarity_confidence': 1.0  # Will be updated during matching
            }
            
            column_features[col_name] = features
        
        return column_features
    
    def _analyze_column_name(self, column_name: str) -> Dict[str, Any]:
        """
        Analyze column name using learned patterns.
        """
        name_lower = column_name.lower()
        
        # Tokenize
        tokens = name_lower.replace('_', ' ').replace('-', ' ').split()
        
        # Check for learned suffixes
        suffix_score = 0
        if tokens:
            last_token = tokens[-1]
            suffix_frequency = self.learned_patterns['common_suffixes'].get(last_token, 0)
            suffix_score = suffix_frequency
        
        # Calculate name complexity
        token_count = len(tokens)
        avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
        
        # Name similarity to other names in the learned data
        # (We'll calculate this during comparison)
        
        return {
            'tokens': tokens,
            'token_count': token_count,
            'avg_token_length': avg_token_length,
            'suffix_score': suffix_score,
            'has_separators': '_' in column_name or '-' in column_name or ' ' in column_name
        }
    
    def _analyze_data_patterns(self, patterns: Dict, samples: List, col_type: str) -> Dict[str, Any]:
        """
        Analyze data patterns using learned type patterns.
        """
        digit_ratio = patterns.get('digit_ratio', 0)
        unique_ratio = patterns.get('unique_ratio', 0)
        char_dist = patterns.get('char_distribution', {})
        length_stats = patterns.get('length_stats', {})
        
        # Compare with learned type patterns
        type_patterns = self.learned_patterns['type_patterns'].get(col_type, {})
        
        type_consistency = 0.5
        if type_patterns:
            # Check how well this column matches its type's typical patterns
            avg_digit = type_patterns.get('avg_digit_ratio', 0)
            std_digit = type_patterns.get('std_digit_ratio', 0)
            
            avg_unique = type_patterns.get('avg_unique_ratio', 0)
            std_unique = type_patterns.get('std_unique_ratio', 0)
            
            if std_digit > 0:
                digit_zscore = abs(digit_ratio - avg_digit) / std_digit
                digit_consistency = 1.0 / (1.0 + digit_zscore)
            else:
                digit_consistency = 1.0 - abs(digit_ratio - avg_digit)
            
            if std_unique > 0:
                unique_zscore = abs(unique_ratio - avg_unique) / std_unique
                unique_consistency = 1.0 / (1.0 + unique_zscore)
            else:
                unique_consistency = 1.0 - abs(unique_ratio - avg_unique)
            
            type_consistency = (digit_consistency + unique_consistency) / 2
        
        # Check if column might be ID-like based on learned patterns
        id_like_patterns = self.learned_patterns.get('id_like_patterns', {})
        is_id_like = False
        
        if id_like_patterns:
            digit_threshold = id_like_patterns.get('digit_threshold', 0.7)
            unique_threshold = id_like_patterns.get('unique_threshold', 0.9)
            
            if digit_ratio > digit_threshold and unique_ratio > unique_threshold:
                is_id_like = True
        
        # Check if column might be categorical
        is_categorical = unique_ratio < 0.3
        
        return {
            'digit_ratio': digit_ratio,
            'unique_ratio': unique_ratio,
            'type_consistency': type_consistency,
            'is_id_like': is_id_like,
            'is_categorical': is_categorical,
            'length_mean': length_stats.get('mean', 0),
            'length_std': length_stats.get('std', 0)
        }
    
    def _calculate_data_driven_similarity(self, schema1_features: Dict[str, Dict], 
                                         schema2_features: Dict[str, Dict]) -> np.ndarray:
        """
        Calculate similarity matrix using data-driven features only.
        """
        cols1 = list(schema1_features.keys())
        cols2 = list(schema2_features.keys())
        
        n = len(cols1)
        m = len(cols2)
        
        sim_matrix = np.zeros((n, m))
        
        for i, col1_name in enumerate(cols1):
            col1_features = schema1_features[col1_name]
            
            for j, col2_name in enumerate(cols2):
                col2_features = schema2_features[col2_name]
                
                similarity = self._calculate_column_similarity_data_driven(col1_features, col2_features)
                sim_matrix[i, j] = similarity
        
        return sim_matrix
    
    def _calculate_column_similarity_data_driven(self, col1: Dict, col2: Dict) -> float:
        """
        Calculate similarity using data-driven features only.
        """
        # 1. Name similarity (40%)
        name_sim = self._calculate_name_similarity_data_driven(
            col1['name'], col1['name_analysis'],
            col2['name'], col2['name_analysis']
        )
        if col1['name'] == "sale_id" or col2['name'] == "sale_id":
            print("*"*50)
            print(name_sim,col1['name'],col2['name'])
            print("*"*50)
        # 2. Type compatibility (25%)
        type_sim = self._calculate_type_similarity_data_driven(
            col1['type'], col1['data_analysis'],
            col2['type'], col2['data_analysis']
        )
        
        # 3. Pattern similarity (25%)
        pattern_sim = self._calculate_pattern_similarity_data_driven(
            col1['data_analysis'], col2['data_analysis']
        )
        
        # 4. Value sample similarity (10%) - if samples available
        value_sim = self._calculate_value_similarity_data_driven(col1, col2)
        
        # Weighted combination
        return 0.8 * name_sim + 0.05 * type_sim + 0.05 * pattern_sim + 0.1 * value_sim
    
    # def _calculate_name_similarity_data_driven(self, name1: str, analysis1: Dict,
    #                                           name2: str, analysis2: Dict) -> float:
    #     """Calculate name similarity using learned patterns."""
    #     if name1.lower() == name2.lower():
    #         return 1.0
        
    #     similarities = []
        
    #     # 1. Edit distance
    #     edit_sim = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    #     similarities.append(edit_sim)
        
    #     # 2. Token overlap
    #     tokens1 = set(analysis1['tokens'])
    #     tokens2 = set(analysis2['tokens'])
        
    #     if tokens1 and tokens2:
    #         union = tokens1.union(tokens2)
    #         intersection = tokens1.intersection(tokens2)
    #         token_sim = len(intersection) / len(union) if union else 0.0
    #         similarities.append(token_sim)
        
    #     # 3. Semantic similarity (if embeddings available)
    #     if self.word_embeddings:
    #         semantic_sim = self._calculate_semantic_similarity_data_driven(name1, name2)
    #         similarities.append(semantic_sim)
        
    #     # 4. Suffix consistency
    #     suffix_sim = 1.0 - abs(analysis1['suffix_score'] - analysis2['suffix_score'])
    #     similarities.append(suffix_sim)
        
    #     # Return average of available similarities
    #     return np.mean(similarities) if similarities else 0.5
    def _calculate_name_similarity_data_driven(self, name1: str, analysis1: Dict,
                                          name2: str, analysis2: Dict) -> float:
        semantic_sim = 0.0
        if self.word_embeddings:
            semantic_sim = self._calculate_semantic_similarity_aligned(analysis1, analysis2)
        return max(semantic_sim,0.0)
        # """Calculate name similarity with weighted components."""
        # if name1.lower() == name2.lower():
        #     return 1.0
        
        # # Base weights
        # weights = {
        #     'edit': 0.25,
        #     'token': 0.20, 
        #     'semantic': 0.45,
        #     'suffix': 0.10
        # }
        
        # # Adjust based on token count (more tokens = more weight to semantic)
        # token_count1 = len(analysis1['tokens'])
        # token_count2 = len(analysis2['tokens'])
        # avg_tokens = (token_count1 + token_count2) / 2
        
        # if avg_tokens > 2:
        #     # Multi-word names: emphasize semantic similarity more
        #     weights['semantic'] = 0.70
        #     weights['edit'] = 0.10
        #     weights['token'] = 0.10
        #     weights['suffix'] = 0.10
        # elif avg_tokens == 1:
        #     # Single token names: emphasize edit distance more
        #     weights['edit'] = 0.40
        #     weights['semantic'] = 0.35
        #     weights['token'] = 0.20  # Less relevant for single token
        #     weights['suffix'] = 0.05



        
        # weighted_sum = 0.0
        # total_weight = 0.0
        
        # # 1. Edit distance (25%)
        # edit_sim = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        # weighted_sum += weights['edit'] * edit_sim
        # total_weight += weights['edit']
        
        # # 2. Token overlap (20%)
        # tokens1 = set(analysis1['tokens'])
        # tokens2 = set(analysis2['tokens'])
        # token_sim = 0.0
        # if tokens1 and tokens2:
        #     union = tokens1.union(tokens2)
        #     intersection = tokens1.intersection(tokens2)
        #     token_sim = len(intersection) / len(union) if union else 0.0
        # weighted_sum += weights['token'] * token_sim
        # total_weight += weights['token']
        
        # # 3. Semantic similarity (45% - highest)
        # if self.word_embeddings:
        #     semantic_sim = self._calculate_semantic_similarity_aligned(analysis1, analysis2)
        # weighted_sum += weights['semantic'] * semantic_sim
        # total_weight += weights['semantic']
        
        # # 4. Suffix consistency (10%)
        # suffix_sim = 1.0 - abs(analysis1['suffix_score'] - analysis2['suffix_score'])
        # weighted_sum += weights['suffix'] * suffix_sim
        # total_weight += weights['suffix']
        
        # return weighted_sum / total_weight if total_weight > 0 else 0.5





    
    def _calculate_semantic_similarity_data_driven(self, name1: str, name2: str) -> float:
        """Calculate semantic similarity without assumptions."""
        if not self.word_embeddings:
            return 0.5
        
        # Extract meaningful tokens (length > 2)
        tokens1 = [t for t in name1.lower().replace('_', ' ').replace('-', ' ').split() 
                  if len(t) > 2]
        tokens2 = [t for t in name2.lower().replace('_', ' ').replace('-', ' ').split() 
                  if len(t) > 2]
        
        if not tokens1 or not tokens2:
            return 0.5
        




        phrase1 = ' '.join(tokens1)
        phrase2 = ' '.join(tokens2)
        embeddings = self.word_embeddings.encode([phrase1, phrase2])
        sentence_transformer_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        # Ensure it's between 0-1
        sentence_transformer_similarity = max(0.0, min(1.0, sentence_transformer_similarity))
        return sentence_transformer_similarity
        # vecs1 = []
        # for token in tokens1:
        #             if token in self.word_embeddings:
        #                 vecs1.append(self.word_embeddings[token])
                
        # vecs2 = []
        # for token in tokens2:
        #             if token in self.word_embeddings:
        #                 vecs2.append(self.word_embeddings[token])
                
        # if vecs1 and vecs2:
        #             avg_vec1 = np.mean(vecs1, axis=0)
        #             avg_vec2 = np.mean(vecs2, axis=0)
        #             embedding_similarity = cosine_similarity([avg_vec1], [avg_vec2])[0][0]
        #             # Normalize to 0-1
        #             embedding_similarity = (embedding_similarity + 1) / 2
        # return embedding_similarity if embedding_similarity else 0.5 
        



    # def _calculate_semantic_similarity_aligned(self, features1: Dict, features2: Dict) -> float:
    #     """Find best alignment between tokens."""
    #     if not self.word_embeddings:
    #         return 0.5
        
    #     tokens1 = list(set(features1['tokens']))
    #     tokens2 = list(set(features2['tokens']))
        
    #     if not tokens1 or not tokens2:
    #         return 0.5
        
    #     try:
    #         # Get embeddings for valid tokens
    #         vecs1 = {}
    #         valid_tokens1 = []
    #         for token in tokens1:
    #             if token in self.word_embeddings:
    #                 vecs1[token] = self.word_embeddings[token]
    #                 valid_tokens1.append(token)
    #             else:
    #                 print(token , "<-Not in Embedding!")
            
    #         vecs2 = {}
    #         valid_tokens2 = []
    #         for token in tokens2:
    #             if token in self.word_embeddings:
    #                 vecs2[token] = self.word_embeddings[token]
    #                 valid_tokens2.append(token)
    #             else:
    #                 print(token , "<-Not in Embedding!")
            
    #         if not vecs1 or not vecs2:
    #             return 0.5
            
    #         # Create similarity matrix
    #         similarity_matrix = np.zeros((len(valid_tokens1), len(valid_tokens2)))
            
    #         for i, token1 in enumerate(valid_tokens1):
    #             vec1 = vecs1[token1]
    #             for j, token2 in enumerate(valid_tokens2):
    #                 vec2 = vecs2[token2]
    #                 cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    #                 similarity_matrix[i, j] = cos_sim
            
    #         # Find best alignment (greedy or Hungarian algorithm)
    #         aligned_similarities = []
    #         used_j = set()
            
    #         # Greedy: for each token in set1, find best unused match in set2
    #         for i in range(len(valid_tokens1)):
    #             best_sim = -1
    #             best_j = -1
                
    #             for j in range(len(valid_tokens2)):
    #                 if j not in used_j and similarity_matrix[i, j] > best_sim:
    #                     best_sim = similarity_matrix[i, j]
    #                     best_j = j
                
    #             if best_j != -1:
    #                 aligned_similarities.append(best_sim)
    #                 used_j.add(best_j)
            
    #         # Add penalty for unmatched tokens
    #         unmatched_penalty = (abs(len(valid_tokens1) - len(valid_tokens2)) / 
    #                         max(len(valid_tokens1), len(valid_tokens2)))
            
    #         if aligned_similarities:
    #             avg_aligned_sim = np.mean(aligned_similarities)
    #             # Adjust for unmatched tokens
    #             final_sim = avg_aligned_sim * (1 - 0.3 * unmatched_penalty)
    #             return (final_sim + 1) / 2
            
    #     except Exception as e:
    #         print(f"Error in aligned semantic similarity: {e}")
        
    #     return 0.5
    def _calculate_semantic_similarity_aligned(self, features1: Dict, features2: Dict) -> float:
        """Find best alignment between tokens."""
        if not self.word_embeddings:
            return 0.5
        
        tokens1 = list(set(features1['tokens']))
        tokens2 = list(set(features2['tokens']))
        


        phrase1 = ' '.join(tokens1)
        phrase2 = ' '.join(tokens2)
        sentence_embeddings = self.word_embeddings.encode([phrase1, phrase2])
        sentence_similarity = cosine_similarity([sentence_embeddings[0]],[sentence_embeddings[1]])[0][0]# Ensure it's between 0-1
        sentence_similarity = max(0.0, min(1.0, sentence_similarity))
        



        if not tokens1 or not tokens2:
            return 0.0
        
        try:
            
            # Create similarity matrix
            similarity_matrix = np.zeros((len(tokens1), len(tokens2)))
            
            for i, token1 in enumerate(tokens1):
                for j, token2 in enumerate(tokens2):
                    embeddings = self.word_embeddings.encode([token1, token2])
                    sentence_transformer_similarity = cosine_similarity(
                        [embeddings[0]], 
                        [embeddings[1]]
                    )[0][0]
                    # Ensure it's between 0-1
                    sentence_transformer_similarity = max(0.0, min(1.0, sentence_transformer_similarity))
                    similarity_matrix[i, j] = sentence_transformer_similarity
            
            # Find best alignment (greedy or Hungarian algorithm)
            aligned_similarities = []
            used_j = set()
            
            # Greedy: for each token in set1, find best unused match in set2
            for i in range(len(tokens1)):
                best_sim = -1
                best_j = -1
                
                for j in range(len(tokens2)):
                    if j not in used_j and similarity_matrix[i, j] > best_sim:
                        best_sim = similarity_matrix[i, j]
                        best_j = j
                
                if best_j != -1:
                    aligned_similarities.append(best_sim)
                    used_j.add(best_j)
            
            # Add penalty for unmatched tokens
            unmatched_penalty = (abs(len(tokens1) - len(tokens2)) / 
                            max(len(tokens1), len(tokens2)))
            
            if aligned_similarities:
                avg_aligned_sim = np.mean(aligned_similarities)
                # Adjust for unmatched tokens
                final_sim = avg_aligned_sim * (1 - 0.3 * unmatched_penalty)
                token_aligned_similarity = (final_sim + 1) / 2
                token_aligned_similarity = max(0.0, min(1.0, token_aligned_similarity))
            

            return max(0.0,sentence_similarity)
            # num_tokens = max(len(tokens1), len(tokens2))
        
            # if num_tokens == 1:
            #     # Single token: rely more on phrase similarity
            #     weights = {'phrase': 0.8, 'token': 0.2}
            # elif num_tokens == 2:
            #     # Two tokens: balanced
            #     weights = {'phrase': 0.6, 'token': 0.4}
            # else:
            #     # Multiple tokens: emphasize token alignment more
            #     weights = {'phrase': 0.4, 'token': 0.6}
            
            # final_similarity = (
            #     weights['phrase'] * sentence_similarity +
            #     weights['token'] * token_aligned_similarity
            # )
            
            # return float(max(0.0, min(1.0, final_similarity)))
        
        except Exception as e:
            print(f"Error in aligned semantic similarity: {e}")
        
        return 0.0



    
    def _calculate_type_similarity_data_driven(self, type1: str, analysis1: Dict,
                                              type2: str, analysis2: Dict) -> float:
        """Calculate type compatibility using learned patterns."""
        if type1 == type2:
            return 1.0
        
        # Check type consistency scores
        type_consistency1 = analysis1.get('type_consistency', 0.5)
        type_consistency2 = analysis2.get('type_consistency', 0.5)
        
        # If both columns are consistent with their types, but types differ
        # we need to check if they could be compatible
        
        # Learn compatibility from learned type patterns
        type_patterns = self.learned_patterns['type_patterns']
        
        if type1 in type_patterns and type2 in type_patterns:
            # Compare typical patterns of each type
            pattern1 = type_patterns[type1]
            pattern2 = type_patterns[type2]
            
            # Calculate pattern distance
            digit_diff = abs(pattern1.get('avg_digit_ratio', 0) - 
                           pattern2.get('avg_digit_ratio', 0))
            unique_diff = abs(pattern1.get('avg_unique_ratio', 0) - 
                            pattern2.get('avg_unique_ratio', 0))
            
            pattern_distance = (digit_diff + unique_diff) / 2
            pattern_similarity = 1.0 - min(pattern_distance, 1.0)
            
            return pattern_similarity
        
        return 0.3  # Default low compatibility
    
    def _calculate_pattern_similarity_data_driven(self, analysis1: Dict, analysis2: Dict) -> float:
        """Calculate pattern similarity using learned distributions."""
        similarities = []
        
        # Compare digit ratios
        digit_sim = 1.0 - abs(analysis1['digit_ratio'] - analysis2['digit_ratio'])
        similarities.append(digit_sim)
        
        # Compare unique ratios
        unique_sim = 1.0 - abs(analysis1['unique_ratio'] - analysis2['unique_ratio'])
        similarities.append(unique_sim)
        
        # Compare type consistency
        type_consistency_sim = 1.0 - abs(analysis1['type_consistency'] - analysis2['type_consistency'])
        similarities.append(type_consistency_sim)
        
        # Compare ID-like classification
        if analysis1['is_id_like'] == analysis2['is_id_like']:
            similarities.append(1.0)
        else:
            similarities.append(0.5)
        
        # Compare categorical classification
        if analysis1['is_categorical'] == analysis2['is_categorical']:
            similarities.append(1.0)
        else:
            similarities.append(0.5)
        
        return np.mean(similarities)
    
    def _calculate_value_similarity_data_driven(self, col1: Dict, col2: Dict) -> float:
        """Calculate similarity based on actual values if available."""
        samples1 = col1.get('samples', [])
        samples2 = col2.get('samples', [])
        
        if not samples1 or not samples2:
            return 0.5
        
        # Convert to strings
        str_samples1 = [str(s).strip() for s in samples1 if s is not None]
        str_samples2 = [str(s).strip() for s in samples2 if s is not None]
        
        if not str_samples1 or not str_samples2:
            return 0.5
        
        # Simple overlap check
        set1 = set(str_samples1)
        set2 = set(str_samples2)
        
        if set1 and set2:
            overlap = len(set1.intersection(set2))
            min_size = min(len(set1), len(set2))
            
            if min_size > 0:
                return overlap / min_size
        
        return 0.0
    
    def _find_reference_schema(self, schema_features: Dict[str, Dict]) -> str:
        """
        Find the most representative schema to use as reference.
        """
        schemas = list(schema_features.keys())
        
        if not schemas:
            return ""
        
        # Score each schema based on:
        # 1. Number of columns (more is often better)
        # 2. Column naming consistency
        # 3. Type diversity
        
        schema_scores = {}
        
        for schema_name, features in schema_features.items():
            columns = list(features.keys())
            score = 0.0
            
            # 1. Column count (normalized)
            max_cols = max(len(list(f.keys())) for f in schema_features.values())
            if max_cols > 0:
                score += len(columns) / max_cols
            
            # 2. Naming consistency (based on suffix scores)
            suffix_scores = [f['name_analysis']['suffix_score'] for f in features.values()]
            if suffix_scores:
                score += np.mean(suffix_scores)
            
            # 3. Type diversity (different types is good)
            types = [f['type'] for f in features.values()]
            type_diversity = len(set(types)) / len(types) if types else 0
            score += type_diversity
            
            schema_scores[schema_name] = score
        
        # Return schema with highest score
        return max(schema_scores.items(), key=lambda x: x[1])[0]
    
    def _create_column_groups_from_reference(self, reference_schema: str,
                                            schema_features: Dict[str, Dict],
                                            similarity_matrices: Dict) -> List[Dict]:
        """
        Create column groups starting from the reference schema.
        """
        schemas = list(schema_features.keys())
        reference_features = schema_features[reference_schema]
        
        column_groups = []
        
        # For each column in reference schema
        for ref_col_name, ref_features in reference_features.items():
            group = {
                'reference_schema': reference_schema,
                'reference_column': ref_col_name,
                'reference_features': ref_features,
                'matches': {}
            }
            
            # Find best match in each other schema
            for other_schema in schemas:
                if other_schema == reference_schema:
                    continue
                
                other_features = schema_features[other_schema]
                
                # Get similarity matrix
                sim_matrix_key = (reference_schema, other_schema)
                if sim_matrix_key not in similarity_matrices:
                    continue
                
                sim_matrix = similarity_matrices[sim_matrix_key]
                
                # Find ref column index
                ref_cols = list(reference_features.keys())
                other_cols = list(other_features.keys())
                
                if ref_col_name not in ref_cols:
                    continue
                
                ref_idx = ref_cols.index(ref_col_name)
                
                # Find best match in other schema
                similarities = sim_matrix[ref_idx]
                if len(similarities) == 0:
                    continue
                
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                if best_similarity >= 0.4:  # Minimum similarity threshold
                    best_match_col = other_cols[best_match_idx]
                    group['matches'][other_schema] = {
                        'column': best_match_col,
                        'similarity': float(best_similarity),
                        'features': other_features[best_match_col]
                    }
            
            column_groups.append(group)
        
        return column_groups
    
    def _validate_column_groups(self, column_groups: List[Dict],
                               schema_features: Dict[str, Dict],
                               similarity_matrices: Dict) -> List[Dict]:
        """
        Validate and refine column groups to ensure one-to-one mapping.
        """
        schemas = list(schema_features.keys())
        
        # Track which columns have been assigned
        assigned_columns = {schema: set() for schema in schemas}
        
        validated_groups = []
        
        # Sort groups by average similarity (highest first)
        sorted_groups = sorted(
            column_groups,
            key=lambda g: np.mean([m['similarity'] for m in g['matches'].values()]) 
            if g['matches'] else 0,
            reverse=True
        )
        
        for group in sorted_groups:
            reference_schema = group['reference_schema']
            ref_col = group['reference_column']
            
            # Check if reference column is already assigned (shouldn't happen)
            if ref_col in assigned_columns[reference_schema]:
                continue
            
            # Validate each match
            valid_matches = {}
            for other_schema, match in group['matches'].items():
                other_col = match['column']
                
                # Check if this column is already assigned to another group
                if other_col in assigned_columns[other_schema]:
                    # Conflict resolution: keep the better match
                    continue
                
                # Check if match is reciprocally good
                if self._is_reciprocal_match(
                    reference_schema, ref_col,
                    other_schema, other_col,
                    schema_features, similarity_matrices
                ):
                    valid_matches[other_schema] = match
            
            # Only keep group if it has at least one valid match
            if valid_matches:
                group['matches'] = valid_matches
                
                # Mark columns as assigned
                assigned_columns[reference_schema].add(ref_col)
                for other_schema, match in valid_matches.items():
                    assigned_columns[other_schema].add(match['column'])
                
                validated_groups.append(group)
        
        return validated_groups
    
    def _is_reciprocal_match(self, schema1: str, col1: str,
                            schema2: str, col2: str,
                            schema_features: Dict,
                            similarity_matrices: Dict) -> bool:
        """
        Check if the match is reciprocal (col2 also considers col1 its best match).
        """
        # Get similarity matrix
        sim_matrix_key = (schema1, schema2)
        if sim_matrix_key not in similarity_matrices:
            return False
        
        sim_matrix = similarity_matrices[sim_matrix_key]
        
        # Get column indices
        cols1 = list(schema_features[schema1].keys())
        cols2 = list(schema_features[schema2].keys())
        
        if col1 not in cols1 or col2 not in cols2:
            return False
        
        idx1 = cols1.index(col1)
        idx2 = cols2.index(col2)
        
        # Check if col1 is the best match for col2
        col2_similarities = sim_matrix[:, idx2]
        best_match_for_col2_idx = np.argmax(col2_similarities)
        
        return best_match_for_col2_idx == idx1
    
    def _create_data_driven_golden_schema(self, column_groups: List[Dict],
                                         schema_features: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create golden schema from validated column groups.
        """
        golden_columns = []
        column_details = {}
        
        for group_idx, group in enumerate(column_groups, 1):
            ref_col = group['reference_column']
            ref_features = group['reference_features']
            
            # Collect all columns in this group
            all_columns = [(group['reference_schema'], ref_col)]
            all_columns.extend([(schema, match['column']) 
                              for schema, match in group['matches'].items()])
            
            # Choose name based on consensus
            column_names = [col[1] for col in all_columns]
            golden_name = self._choose_golden_name_data_driven(column_names)
            
            # Collect statistics
            all_types = []
            all_similarities = []
            
            for schema_name, col_name in all_columns:
                if schema_name in schema_features and col_name in schema_features[schema_name]:
                    features = schema_features[schema_name][col_name]
                    all_types.append(features['type'])
            
            # Add match similarities
            for match in group['matches'].values():
                all_similarities.append(match['similarity'])
            
            # Determine consensus type
            from collections import Counter
            type_counts = Counter(all_types)
            consensus_type = type_counts.most_common(1)[0][0] if type_counts else 'unknown'
            
            # Calculate confidence
            avg_similarity = np.mean(all_similarities) if all_similarities else 0.5
            
            golden_col = {
                'name': golden_name,
                'group_id': group_idx,
                'consensus_type': consensus_type,
                'confidence': float(avg_similarity),
                'mapped_schemas': len(all_columns),
                'source_mappings': [
                    {
                        'schema': schema_name,
                        'column': col_name,
                        'type': schema_features[schema_name][col_name]['type']
                        if schema_name in schema_features and col_name in schema_features[schema_name]
                        else 'unknown'
                    }
                    for schema_name, col_name in all_columns
                ]
            }
            
            golden_columns.append(golden_name)
            column_details[golden_name] = golden_col
        
        return {
            'name': f"Golden_Schema_{len(column_groups)}",
            'total_columns': len(golden_columns),
            'columns': golden_columns,
            'column_details': column_details,
            'total_groups': len(column_groups)
        }
    
    def _choose_golden_name_data_driven(self, column_names: List[str]) -> str:
        """
        Choose golden column name based on data-driven consensus.
        """
        if not column_names:
            return "unknown_column"
        
        # Count frequencies
        name_counts = Counter(column_names)
        
        # If one name dominates, use it
        most_common_name, most_common_count = name_counts.most_common(1)[0]
        if most_common_count / len(column_names) >= 0.5:  # Majority
            return most_common_name
        
        # Otherwise, choose the shortest name with highest frequency
        candidate_names = []
        for name, count in name_counts.items():
            frequency = count / len(column_names)
            length_score = 1.0 / (len(name) + 1)  # Prefer shorter names
            score = frequency * 0.7 + length_score * 0.3
            candidate_names.append((name, score))
        
        candidate_names.sort(key=lambda x: x[1], reverse=True)
        return candidate_names[0][0]
    
    def _create_data_driven_mapping_matrix(self, column_groups: List[Dict],
                                          schema_features: Dict[str, Dict],
                                          golden_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create mapping matrix showing one-to-one correspondences.
        """
        schemas = list(schema_features.keys())
        
        mapping_matrix = {
            'schemas': schemas,
            'golden_columns': golden_schema['columns'],
            'mappings': []
        }
        
        for golden_col_name in golden_schema['columns']:
            golden_col_details = golden_schema['column_details'].get(golden_col_name, {})
            source_mappings = golden_col_details.get('source_mappings', [])
            
            mapping = {
                'golden_column': golden_col_name,
                'consensus_type': golden_col_details.get('consensus_type', 'unknown'),
                'confidence': golden_col_details.get('confidence', 0.5),
                'schema_mappings': {}
            }
            
            for schema_name in schemas:
                # Find mapping for this schema
                schema_mapping = next(
                    (m for m in source_mappings if m['schema'] == schema_name), 
                    None
                )
                
                if schema_mapping:
                    mapping['schema_mappings'][schema_name] = {
                        'column': schema_mapping['column'],
                        'type': schema_mapping['type'],
                        'mapped': True
                    }
                else:
                    mapping['schema_mappings'][schema_name] = {
                        'column': None,
                        'type': None,
                        'mapped': False
                    }
            
            mapping_matrix['mappings'].append(mapping)
        
        return mapping_matrix
    # def print_unmapped_fields_simple(self, enhanced_schemas: Dict[str, Any], 
    #                             mapping_results: Dict[str, Any]):
    #     """
    #     Simply print all fields that are not mapped to any cluster.
    #     """
    #     print("\n" + "=" * 100)
    #     print("🔍 UNMAPPED FIELDS - COLUMN NAMES ONLY")
    #     print("=" * 100)
        
    #     # First, collect all mapped columns from all clusters
    #     mapped_columns = defaultdict(set)
        
    #     for cluster_name, result in mapping_results.items():
    #         if 'mapping_matrix' not in result:
    #             continue
            
    #         mapping_matrix = result['mapping_matrix']
            
    #         for mapping in mapping_matrix.get('mappings', []):
    #             schema_mappings = mapping.get('schema_mappings', {})
                
    #             for schema_name, schema_map in schema_mappings.items():
    #                 if schema_map.get('mapped', False) and schema_map.get('column'):
    #                     mapped_columns[schema_name].add(schema_map['column'])
        
    #     # Now, find and print unmapped columns for each schema
    #     total_unmapped = 0
        
    #     for schema_name, schema_data in enhanced_schemas.items():
    #         all_columns = set(schema_data.get('columns', []))
    #         mapped_in_schema = mapped_columns.get(schema_name, set())
    #         unmapped_in_schema = all_columns - mapped_in_schema
            
    #         if unmapped_in_schema:
    #             total_unmapped += len(unmapped_in_schema)
    #             print(f"\n📁 {schema_name} - {len(unmapped_in_schema)} unmapped columns:")
                
    #             # Sort columns alphabetically and print them
    #             for column in sorted(unmapped_in_schema):
    #                 print(f"  • {column}")
        
    #     # Print summary
    #     print(f"\n{'=' * 80}")
    #     print(f"📊 Total unmapped columns across all schemas: {total_unmapped}")
    #     print(f"{'=' * 80}")
    # def print_unmapped_fields_with_closest_matches(self,enhanced_schemas, 
    #                                           mapping_results,
    #                                           mapper):
    #     """
    #     Print unmapped fields with their closest matches and confidence scores.
    #     """
    #     print("\n" + "=" * 100)
    #     print("🔍 UNMAPPED FIELDS WITH CLOSEST MATCHES")
    #     print("=" * 100)
        
    #     # First, collect all mapped columns from all clusters
    #     mapped_columns_info = defaultdict(dict)  # schema_name -> {column_name: cluster_info}
        
    #     for cluster_name, result in mapping_results.items():
    #         if 'mapping_matrix' not in result:
    #             continue
            
    #         mapping_matrix = result['mapping_matrix']
    #         golden_schema = result.get('golden_schema', {})
            
    #         for mapping in mapping_matrix.get('mappings', []):
    #             golden_col = mapping.get('golden_column', '')
    #             confidence = mapping.get('confidence', 0)
    #             consensus_type = mapping.get('consensus_type', '')
    #             schema_mappings = mapping.get('schema_mappings', {})
                
    #             for schema_name, schema_map in schema_mappings.items():
    #                 if schema_map.get('mapped', False) and schema_map.get('column'):
    #                     column_name = schema_map['column']
    #                     mapped_columns_info[schema_name][column_name] = {
    #                         'cluster': cluster_name,
    #                         'golden_column': golden_col,
    #                         'confidence': confidence,
    #                         'type': consensus_type,
    #                         'schema_type': schema_map.get('type', '')
    #                     }
        
    #     # Now, find unmapped columns and their closest matches
    #     total_unmapped = 0
    #     all_closest_matches = []
        
    #     for schema_name, schema_data in enhanced_schemas.items():
    #         all_columns = set(schema_data.get('columns', []))
    #         mapped_in_schema = set(mapped_columns_info.get(schema_name, {}).keys())
    #         unmapped_in_schema = all_columns - mapped_in_schema
            
    #         if unmapped_in_schema:
    #             total_unmapped += len(unmapped_in_schema)
    #             print(f"\n📁 {schema_name} - {len(unmapped_in_schema)} unmapped columns:")
                
    #             # For each unmapped column, find closest matches in the same schema
    #             schema_features = mapper._extract_data_driven_features(schema_name)
                
    #             for unmapped_column in sorted(unmapped_in_schema):
    #                 print(f"\n  • {unmapped_column}")
                    
    #                 if unmapped_column not in schema_features:
    #                     print(f"    No feature information available")
    #                     continue
                    
    #                 unmapped_features = schema_features[unmapped_column]
                    
    #                 # Find closest mapped columns in the same schema
    #                 closest_matches = []
    #                 for mapped_column, cluster_info in mapped_columns_info.get(schema_name, {}).items():
    #                     if mapped_column in schema_features:
    #                         mapped_features = schema_features[mapped_column]
                            
    #                         # Calculate similarity
    #                         similarity = mapper._calculate_column_similarity_data_driven(
    #                             unmapped_features, 
    #                             mapped_features
    #                         )
                            
    #                         closest_matches.append({
    #                             'mapped_column': mapped_column,
    #                             'similarity': similarity,
    #                             'cluster': cluster_info['cluster'],
    #                             'golden_column': cluster_info['golden_column'],
    #                             'cluster_confidence': cluster_info['confidence'],
    #                             'type': cluster_info['type']
    #                         })
                    
    #                 # Sort by similarity (descending)
    #                 closest_matches.sort(key=lambda x: x['similarity'], reverse=True)
                    
    #                 # Show top 3 closest matches
    #                 if closest_matches:
    #                     print(f"    Closest mapped columns in this schema:")
    #                     for i, match in enumerate(closest_matches[:3]):
    #                         similarity_color = "🟢" if match['similarity'] >= 0.6 else "🟡" if match['similarity'] >= 0.4 else "🔴"
    #                         print(f"    {i+1}. {similarity_color} {match['mapped_column']} "
    #                             f"(sim={match['similarity']:.2f})")
    #                         print(f"       → Mapped to: {match['golden_column']} in {match['cluster']}")
    #                         print(f"       → Cluster confidence: {match['cluster_confidence']:.2f}")
    #                         print(f"       → Type: {match['type']}")
    #                 else:
    #                     print(f"    No mapped columns found in this schema for comparison")
                    
    #                 # Also check if there are similar columns in other schemas
    #                 other_schema_matches = []
    #                 for other_schema_name, other_mapped in mapped_columns_info.items():
    #                     if other_schema_name == schema_name:
    #                         continue
                        
    #                     # Get features for other schema
    #                     other_features = mapper._extract_data_driven_features(other_schema_name)
                        
    #                     for other_column, cluster_info in other_mapped.items():
    #                         if other_column in other_features:
    #                             other_column_features = other_features[other_column]
                                
    #                             similarity = mapper._calculate_column_similarity_data_driven(
    #                                 unmapped_features,
    #                                 other_column_features
    #                             )
                                
    #                             if similarity >= 0.4:  # Threshold for showing
    #                                 other_schema_matches.append({
    #                                     'schema': other_schema_name,
    #                                     'column': other_column,
    #                                     'similarity': similarity,
    #                                     'cluster': cluster_info['cluster'],
    #                                     'golden_column': cluster_info['golden_column'],
    #                                     'cluster_confidence': cluster_info['confidence']
    #                                 })
                    
    #                 # Show top cross-schema matches
    #                 if other_schema_matches:
    #                     other_schema_matches.sort(key=lambda x: x['similarity'], reverse=True)
    #                     print(f"    Similar columns in other schemas (top 2):")
    #                     for i, match in enumerate(other_schema_matches[:2]):
    #                         similarity_color = "🟢" if match['similarity'] >= 0.6 else "🟡" if match['similarity'] >= 0.4 else "🔴"
    #                         print(f"    {i+1}. {similarity_color} {match['schema']}.{match['column']} "
    #                             f"(sim={match['similarity']:.2f})")
    #                         print(f"       → Mapped to: {match['golden_column']} in {match['cluster']}")
    #                         print(f"       → Cluster confidence: {match['cluster_confidence']:.2f}")
                    
    #                 all_closest_matches.append({
    #                     'schema': schema_name,
    #                     'column': unmapped_column,
    #                     'closest_matches': closest_matches[:3] if closest_matches else [],
    #                     'cross_schema_matches': other_schema_matches[:2] if other_schema_matches else []
    #                 })
        
    #     # Print summary statistics
    #     print(f"\n{'=' * 80}")
    #     print(f"📊 UNMAPPED FIELDS ANALYSIS")
    #     print(f"{'=' * 80}")
        
    #     # Calculate statistics
    #     columns_with_same_schema_matches = sum(1 for cm in all_closest_matches if cm['closest_matches'])
    #     columns_with_cross_schema_matches = sum(1 for cm in all_closest_matches if cm['cross_schema_matches'])
        
    #     print(f"Total unmapped columns: {total_unmapped}")
    #     print(f"Columns with similar mapped columns in same schema: {columns_with_same_schema_matches}")
    #     print(f"Columns with similar columns in other schemas: {columns_with_cross_schema_matches}")
        
    #     # Show top 10 most similar unmapped columns (those that were almost mapped)
    #     if all_closest_matches:
    #         print(f"\n🔝 Top 10 unmapped columns closest to being mapped:")
    #         sorted_by_closest = sorted(all_closest_matches, 
    #                                 key=lambda x: max([m['similarity'] for m in x['closest_matches']] 
    #                                                 if x['closest_matches'] else [0]), 
    #                                 reverse=True)
            
    #         for i, item in enumerate(sorted_by_closest[:10]):
    #             if item['closest_matches']:
    #                 best_match = item['closest_matches'][0]
    #                 print(f"{i+1}. {item['schema']}.{item['column']}")
    #                 print(f"   Similarity to {best_match['mapped_column']}: {best_match['similarity']:.2f}")
    #                 print(f"   Missed threshold by: {0.4 - best_match['similarity']:.2f}")
        
    #     print(f"{'=' * 80}")
    def print_unmapped_fields_within_cluster_matches(self, enhanced_schemas: Dict[str, Any], 
                                                mapping_results: Dict[str, Any],
                                                mapper,
                                                clusters: List[List[str]]):
        """
        Print unmapped fields with their closest matches within the same cluster.
        """
        print("\n" + "=" * 100)
        print("🔍 UNMAPPED FIELDS WITHIN CLUSTER CLOSEST MATCHES")
        print("=" * 100)
        
        # Create a mapping of schema -> cluster
        schema_to_cluster = {}
        for cluster_idx, cluster_files in enumerate(clusters, 1):
            cluster_name = f"Cluster_{cluster_idx}"
            for schema_name in cluster_files:
                schema_to_cluster[schema_name] = cluster_name
        
        # First, collect all mapped columns with their cluster info
        mapped_columns_info = defaultdict(lambda: defaultdict(dict))  # cluster -> schema -> {column: info}
        
        for cluster_name, result in mapping_results.items():
            if 'mapping_matrix' not in result:
                continue
            
            mapping_matrix = result['mapping_matrix']
            
            for mapping in mapping_matrix.get('mappings', []):
                golden_col = mapping.get('golden_column', '')
                confidence = mapping.get('confidence', 0)
                consensus_type = mapping.get('consensus_type', '')
                schema_mappings = mapping.get('schema_mappings', {})
                
                for schema_name, schema_map in schema_mappings.items():
                    if schema_map.get('mapped', False) and schema_map.get('column'):
                        column_name = schema_map['column']
                        mapped_columns_info[cluster_name][schema_name][column_name] = {
                            'golden_column': golden_col,
                            'confidence': confidence,
                            'type': consensus_type,
                            'schema_type': schema_map.get('type', '')
                        }
        
        # Now, find unmapped columns and their closest matches within the same cluster
        total_unmapped = 0
        all_closest_matches = []
        
        for schema_name, schema_data in enhanced_schemas.items():
            # Skip if schema is not in any cluster
            if schema_name not in schema_to_cluster:
                continue
                
            cluster_name = schema_to_cluster[schema_name]
            all_columns = set(schema_data.get('columns', []))
            
            # Get mapped columns in this schema from this cluster
            mapped_in_this_schema = set(mapped_columns_info.get(cluster_name, {}).get(schema_name, {}).keys())
            unmapped_in_schema = all_columns - mapped_in_this_schema
            
            if unmapped_in_schema:
                total_unmapped += len(unmapped_in_schema)
                print(f"\n📁 {schema_name} ({cluster_name}) - {len(unmapped_in_schema)} unmapped columns:")
                
                # Get features for this schema
                schema_features = mapper._extract_data_driven_features(schema_name)
                
                for unmapped_column in sorted(unmapped_in_schema):
                    print(f"\n  • {unmapped_column}")
                    
                    if unmapped_column not in schema_features:
                        print(f"    No feature information available")
                        continue
                    
                    unmapped_features = schema_features[unmapped_column]
                    
                    # Find closest mapped columns in the SAME SCHEMA (within same cluster)
                    closest_matches_same_schema = []
                    for mapped_column, cluster_info in mapped_columns_info.get(cluster_name, {}).get(schema_name, {}).items():
                        if mapped_column in schema_features:
                            mapped_features = schema_features[mapped_column]
                            
                            # Calculate similarity
                            similarity = mapper._calculate_column_similarity_data_driven(
                                unmapped_features, 
                                mapped_features
                            )
                            
                            closest_matches_same_schema.append({
                                'mapped_column': mapped_column,
                                'similarity': similarity,
                                'golden_column': cluster_info['golden_column'],
                                'cluster_confidence': cluster_info['confidence'],
                                'type': cluster_info['type']
                            })
                    
                    # Find closest mapped columns in OTHER SCHEMAS within SAME CLUSTER
                    closest_matches_same_cluster = []
                    
                    # Get all schemas in this cluster
                    cluster_idx = int(cluster_name.split('_')[1]) - 1
                    if cluster_idx < len(clusters):
                        cluster_schemas = clusters[cluster_idx]
                        
                        for other_schema in cluster_schemas:
                            if other_schema == schema_name:
                                continue  # Skip same schema (handled above)
                            
                            # Get mapped columns in this other schema
                            other_mapped_columns = mapped_columns_info.get(cluster_name, {}).get(other_schema, {})
                            
                            if not other_mapped_columns:
                                continue
                            
                            # Get features for other schema
                            other_features = mapper._extract_data_driven_features(other_schema)
                            
                            for mapped_column, cluster_info in other_mapped_columns.items():
                                if mapped_column in other_features:
                                    mapped_features = other_features[mapped_column]
                                    
                                    similarity = mapper._calculate_column_similarity_data_driven(
                                        unmapped_features,
                                        mapped_features
                                    )
                                    
                                    if similarity >= 0.3:  # Lower threshold for cross-schema
                                        closest_matches_same_cluster.append({
                                            'schema': other_schema,
                                            'column': mapped_column,
                                            'similarity': similarity,
                                            'golden_column': cluster_info['golden_column'],
                                            'cluster_confidence': cluster_info['confidence'],
                                            'type': cluster_info['type']
                                        })
                    
                    # Sort by similarity
                    closest_matches_same_schema.sort(key=lambda x: x['similarity'], reverse=True)
                    closest_matches_same_cluster.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Show closest matches in same schema
                    if closest_matches_same_schema:
                        print(f"    Closest mapped columns in SAME SCHEMA:")
                        for i, match in enumerate(closest_matches_same_schema[:3]):  # Top 3
                            similarity_color = "🟢" if match['similarity'] >= 0.6 else "🟡" if match['similarity'] >= 0.4 else "🔴"
                            print(f"    {i+1}. {similarity_color} {match['mapped_column']} "
                                f"(sim={match['similarity']:.2f})")
                            print(f"       → Mapped to golden column: {match['golden_column']}")
                            print(f"       → Cluster confidence: {match['cluster_confidence']:.2f}")
                            print(f"       → Type: {match['type']}")
                    else:
                        print(f"    No mapped columns found in same schema")
                    
                    # Show closest matches in same cluster (other schemas)
                    if closest_matches_same_cluster:
                        print(f"    Closest mapped columns in SAME CLUSTER (other schemas):")
                        for i, match in enumerate(closest_matches_same_cluster[:2]):  # Top 2
                            similarity_color = "🟢" if match['similarity'] >= 0.6 else "🟡" if match['similarity'] >= 0.4 else "🔴"
                            print(f"    {i+1}. {similarity_color} {match['schema']}.{match['column']} "
                                f"(sim={match['similarity']:.2f})")
                            print(f"       → Mapped to golden column: {match['golden_column']}")
                            print(f"       → Cluster confidence: {match['cluster_confidence']:.2f}")
                            print(f"       → Type: {match['type']}")
                    else:
                        print(f"    No similar mapped columns in other schemas of this cluster")
                    

                    same_schema_sims = [m['similarity'] for m in closest_matches_same_schema] if closest_matches_same_schema else [0]
                    same_cluster_sims = [m['similarity'] for m in closest_matches_same_cluster] if closest_matches_same_cluster else [0]
                    # Get the single maximum value
                    all_sims = same_schema_sims + same_cluster_sims
                    best = max(all_sims) if all_sims else 0.0


                    # Store for summary
                    all_closest_matches.append({
                        'cluster': cluster_name,
                        'schema': schema_name,
                        'column': unmapped_column,
                        'same_schema_matches': closest_matches_same_schema[:3] if closest_matches_same_schema else [],
                        'same_cluster_matches': closest_matches_same_cluster[:2] if closest_matches_same_cluster else [],
                        'best_similarity': best
                    })
                    # max(
                    #         [m['similarity'] for m in closest_matches_same_schema] if closest_matches_same_schema else [0],
                    #         [m['similarity'] for m in closest_matches_same_cluster] if closest_matches_same_cluster else [0]
                    #     )
        
        # Print summary statistics
        print(f"\n{'=' * 80}")
        print(f"📊 UNMAPPED FIELDS ANALYSIS (WITHIN CLUSTER)")
        print(f"{'=' * 80}")
        
        # Calculate statistics
        columns_with_same_schema_matches = sum(1 for cm in all_closest_matches if cm['same_schema_matches'])
        columns_with_same_cluster_matches = sum(1 for cm in all_closest_matches if cm['same_cluster_matches'])
        columns_with_any_matches = sum(1 for cm in all_closest_matches if cm['same_schema_matches'] or cm['same_cluster_matches'])
        
        print(f"Total unmapped columns: {total_unmapped}")
        print(f"Columns with similar mapped columns in same schema: {columns_with_same_schema_matches}")
        print(f"Columns with similar mapped columns in same cluster: {columns_with_same_cluster_matches}")
        print(f"Columns with any similar mapped columns: {columns_with_any_matches}")
        
        # Show top 10 most similar unmapped columns
        if all_closest_matches:
            print(f"\n🔝 Top 10 unmapped columns closest to being mapped:")
            sorted_by_similarity = sorted(all_closest_matches, 
                                        key=lambda x: x['best_similarity'], 
                                        reverse=True)
            
            for i, item in enumerate(sorted_by_similarity[:10]):
                print(f"{i+1}. {item['cluster']} - {item['schema']}.{item['column']}")
                print(f"   Best similarity: {item['best_similarity']:.2f}")
                
                # Show the best match info
                all_matches = item['same_schema_matches'] + item['same_cluster_matches']
                if all_matches:
                    best_match = max(all_matches, key=lambda x: x['similarity'])
                    if 'schema' in best_match:
                        print(f"   Best match: {best_match['schema']}.{best_match['column']} "
                            f"(sim={best_match['similarity']:.2f})")
                    else:
                        print(f"   Best match: {best_match['mapped_column']} "
                            f"(sim={best_match['similarity']:.2f})")
                
                # Show mapping threshold gap
                threshold_gap = 0.4 - item['best_similarity']
                if threshold_gap > 0:
                    print(f"   Missed mapping threshold by: {threshold_gap:.2f}")
        
        # Group by cluster for cluster-level analysis
        print(f"\n📊 UNMAPPED FIELDS BY CLUSTER:")
        unmapped_by_cluster = defaultdict(list)
        for item in all_closest_matches:
            unmapped_by_cluster[item['cluster']].append(item)
        
        for cluster_name, cluster_items in sorted(unmapped_by_cluster.items()):
            print(f"\n  {cluster_name}:")
            print(f"    Total unmapped: {len(cluster_items)}")
            
            # Calculate average best similarity in this cluster
            avg_similarity = np.mean([item['best_similarity'] for item in cluster_items]) if cluster_items else 0
            print(f"    Average best similarity: {avg_similarity:.2f}")
            
            # Count by similarity ranges
            high_sim = sum(1 for item in cluster_items if item['best_similarity'] >= 0.6)
            med_sim = sum(1 for item in cluster_items if 0.4 <= item['best_similarity'] < 0.6)
            low_sim = sum(1 for item in cluster_items if item['best_similarity'] < 0.4)
            
            print(f"    High similarity (≥0.6): {high_sim}")
            print(f"    Medium similarity (0.4-0.6): {med_sim}")
            print(f"    Low similarity (<0.4): {low_sim}")
        
        print(f"{'=' * 80}")

    def _calculate_data_driven_quality(self, column_groups: List[Dict],
                                      schema_features: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate quality metrics for the mapping.
        """
        if not column_groups:
            return {}
        
        all_similarities = []
        schema_coverage = defaultdict(int)
        total_columns = 0
        
        for schema_name, features in schema_features.items():
            total_columns += len(features)
        
        for group in column_groups:
            # Add reference schema
            schema_coverage[group['reference_schema']] += 1
            
            # Add matches
            for other_schema, match in group['matches'].items():
                schema_coverage[other_schema] += 1
                all_similarities.append(match['similarity'])
        
        # Calculate coverage per schema
        coverage_ratios = {}
        for schema_name, features in schema_features.items():
            total_in_schema = len(features)
            mapped_in_schema = schema_coverage.get(schema_name, 0)
            coverage_ratios[schema_name] = mapped_in_schema / total_in_schema if total_in_schema > 0 else 0
        
        return {
            'average_similarity': float(np.mean(all_similarities)) if all_similarities else 0,
            'median_similarity': float(np.median(all_similarities)) if all_similarities else 0,
            'coverage_ratios': coverage_ratios,
            'total_mapped_columns': sum(schema_coverage.values()),
            'total_possible_columns': total_columns,
            'overall_coverage': sum(schema_coverage.values()) / total_columns if total_columns > 0 else 0
        }
    
    def _handle_single_schema_data_driven(self, cluster_files: List[str]) -> Dict[str, Any]:
        """Handle single schema case."""
        # Similar to previous implementation but data-driven
        # ... (implementation omitted for brevity)
        pass
    
import json
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# Copy the complete DataDrivenColumnMapper class here
# (The entire class code from above should be placed here)

def print_data_driven_mappings(mapping_results: Dict[str, Any]):
    """
    Print data-driven one-to-one column mappings.
    """
    print("\n" + "=" * 100)
    print("DATA-DRIVEN ONE-TO-ONE COLUMN MAPPINGS")
    print("=" * 100)
    
    for cluster_name, result in mapping_results.items():
        if 'mapping_matrix' not in result:
            continue
        
        mapping_matrix = result['mapping_matrix']
        golden_schema = result.get('golden_schema', {})
        cluster_files = result.get('cluster_files', [])
        quality_metrics = result.get('quality_metrics', {})
        
        print(f"\n{'━' * 80}")
        print(f"📦 {cluster_name}")
        print(f"{'━' * 80}")
        
        # Print schemas
        print(f"\n📁 Schemas ({len(cluster_files)}):")
        for file in cluster_files:
            print(f"   • {file}")
        
        # Print reference schema
        if 'reference_schema' in result:
            print(f"\n🎯 Reference schema: {result['reference_schema']}")
        
        # Print quality metrics
        if quality_metrics:
            print(f"\n📈 Quality Metrics:")
            print(f"   • Average similarity: {quality_metrics.get('average_similarity', 0):.2f}")
            print(f"   • Coverage: {quality_metrics.get('overall_coverage', 0):.1%}")
            print(f"   • Mapped columns: {quality_metrics.get('total_mapped_columns', 0)}/"
                  f"{quality_metrics.get('total_possible_columns', 0)}")
        
        # Print golden schema
        print(f"\n🎯 Golden Schema: {golden_schema.get('name', 'Unknown')}")
        print(f"   Columns: {len(golden_schema.get('columns', []))}")
        
        # Print mapping table
        print(f"\n🔗 One-to-One Column Mappings:")
        print(f"{'─' * 100}")
        
        schemas = mapping_matrix.get('schemas', [])
        header = f"{'Golden Column':<20} | {'Type':<10} | {'Conf':<5} | "
        header += " | ".join([f"{schema[:12]:<12}" for schema in schemas])
        print(header)
        print(f"{'─' * 100}")
        
        # Print each mapping row
        for mapping in mapping_matrix.get('mappings', []):
            golden_col = mapping.get('golden_column', '')
            consensus_type = mapping.get('consensus_type', '')
            confidence = mapping.get('confidence', 0)
            schema_mappings = mapping.get('schema_mappings', {})
            
            # Truncate long names
            # if len(golden_col) > 20:
            #     golden_col = golden_col[:17] + "..."
            
            row = f"{golden_col:<20} | {consensus_type[:10]:<10} | {confidence:.2f} | "
            
            for schema in schemas:
                schema_map = schema_mappings.get(schema, {})
                if schema_map.get('mapped', False):
                    col_name = schema_map.get('column', '')
                    
                    # Add confidence indicator
                    if confidence >= 0.8:
                        indicator = "🟢 "
                    elif confidence >= 0.6:
                        indicator = "🟡 "
                    else:
                        indicator = "🔴 "
                    
                    # Truncate column name
                    # if len(col_name) > 12:
                    #     col_name = col_name[:9] + "..."
                    
                    row += f"{indicator}{col_name:<12} | "
                else:
                    row += f"{'─':<12} | "
            
            print(row)
        
        # Print confidence distribution
        confidences = [m.get('confidence', 0) for m in mapping_matrix.get('mappings', [])]
        if confidences:
            high_conf = sum(1 for c in confidences if c >= 0.8)
            med_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
            low_conf = sum(1 for c in confidences if c < 0.6)
            
            print(f"\n📊 Confidence Distribution:")
            print(f"   • High (≥0.8): {high_conf} columns")
            print(f"   • Medium (0.6-0.8): {med_conf} columns")
            print(f"   • Low (<0.6): {low_conf} columns")
            print(f"   • Average confidence: {np.mean(confidences):.2f}")
        
        print(f"{'━' * 80}")

def print_detailed_analysis(mapping_results: Dict[str, Any]):
    """
    Print detailed analysis of mappings.
    """
    print("\n" + "=" * 100)
    print("DETAILED MAPPING ANALYSIS")
    print("=" * 100)
    
    for cluster_name, result in mapping_results.items():
        if 'validated_groups' not in result:
            continue
        
        validated_groups = result.get('validated_groups', [])
        schema_features = result.get('schema_features', {})
        
        print(f"\n{'━' * 80}")
        print(f"🔍 {cluster_name} - Detailed Group Analysis")
        print(f"{'━' * 80}")
        
        for group in validated_groups[:10]:  # Show first 10 groups
            ref_col = group.get('reference_column', '')
            ref_schema = group.get('reference_schema', '')
            matches = group.get('matches', {})
            
            print(f"\n📋 Group: {ref_schema}.{ref_col}")
            
            # Calculate group statistics
            similarities = [match['similarity'] for match in matches.values()]
            if similarities:
                avg_sim = np.mean(similarities)
                min_sim = min(similarities)
                max_sim = max(similarities)
                
                print(f"   Matches: {len(matches)} schemas")
                print(f"   Similarity: avg={avg_sim:.2f}, min={min_sim:.2f}, max={max_sim:.2f}")
            
            # Show each match
            for schema, match in matches.items():
                col_name = match['column']
                similarity = match['similarity']
                features = match['features']
                
                confidence = "🟢" if similarity >= 0.8 else "🟡" if similarity >= 0.6 else "🔴"
                
                print(f"   {confidence} {schema}.{col_name} (sim={similarity:.2f}, "
                      f"type={features.get('type', '?')})")
        
        print(f"{'━' * 80}")

def save_results(mapping_results: Dict[str, Any], output_file: str = 'data_driven_mappings.json'):
    """
    Save mapping results to JSON file.
    """
    # Convert numpy arrays and other non-serializable objects
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_for_json(mapping_results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
def column_mappings():
        with open('schema_data.json', 'r') as f:
            enhanced_schemas = json.load(f)
        
        with open('final_clustering_results.json', 'r') as f:
            clustering_results = json.load(f)
        # Assuming you have enhanced_schemas and clustering_results
        mapper = DataDrivenColumnMapper(enhanced_schemas)


        # Get clusters from clustering_results
        clusters = clustering_results.get('clusters', [])

        # Create one-to-one mappings
        results = mapper.map_all_clusters(clusters)
        
        # Print results
        print_data_driven_mappings(results)
        
        # Optional: Print detailed analysis
        # print_detailed_analysis(results)
        
        # Save results
        save_results(results, 'data_driven_column_mappings.json')
        
        # Summary statistics
        print(f"\n📊 FINAL SUMMARY")
        print(f"{'=' * 80}")
        
        total_clusters = len(results)
        total_golden_columns = 0
        total_mappings = 0
        total_high_conf = 0
        
        for cluster_name, result in results.items():
            golden_schema = result.get('golden_schema', {})
            mapping_matrix = result.get('mapping_matrix', {})
            
            total_golden_columns += len(golden_schema.get('columns', []))
            
            for mapping in mapping_matrix.get('mappings', []):
                total_mappings += 1
                if mapping.get('confidence', 0) >= 0.8:
                    total_high_conf += 1
        
        print(f"• Clusters processed: {total_clusters}")
        print(f"• Golden columns created: {total_golden_columns}")
        print(f"• Total one-to-one mappings: {total_mappings}")
        
        if total_mappings > 0:
            print(f"• High confidence mappings: {total_high_conf} ({total_high_conf/total_mappings:.1%})")
        
        print(f"{'=' * 80}")
        print("✅ All done!")



        # # Add this at the end of your main execution, after saving results
        # mapper.print_unmapped_fields_with_closest_matches(enhanced_schemas, results, mapper) #print_unmapped_fields_simple(enhanced_schemas, results)

        # Add this at the end of your main execution, after saving results
        mapper.print_unmapped_fields_within_cluster_matches(enhanced_schemas, results, mapper, clusters)



 
# Main execution
if __name__ == "__main__":
    with open('schema_data.json', 'r') as f:
        enhanced_schemas = json.load(f)
    
    with open('final_clustering_results.json', 'r') as f:
        clustering_results = json.load(f)
    # Assuming you have enhanced_schemas and clustering_results
    mapper = DataDrivenColumnMapper(enhanced_schemas)


    # Get clusters from clustering_results
    clusters = clustering_results.get('clusters', [])

    # Create one-to-one mappings
    results = mapper.map_all_clusters(clusters)
    
    # Print results
    print_data_driven_mappings(results)
    
    # Optional: Print detailed analysis
    # print_detailed_analysis(results)
    
    # Save results
    save_results(results, 'data_driven_column_mappings.json')
    
    # Summary statistics
    print(f"\n📊 FINAL SUMMARY")
    print(f"{'=' * 80}")
    
    total_clusters = len(results)
    total_golden_columns = 0
    total_mappings = 0
    total_high_conf = 0
    
    for cluster_name, result in results.items():
        golden_schema = result.get('golden_schema', {})
        mapping_matrix = result.get('mapping_matrix', {})
        
        total_golden_columns += len(golden_schema.get('columns', []))
        
        for mapping in mapping_matrix.get('mappings', []):
            total_mappings += 1
            if mapping.get('confidence', 0) >= 0.8:
                total_high_conf += 1
    
    print(f"• Clusters processed: {total_clusters}")
    print(f"• Golden columns created: {total_golden_columns}")
    print(f"• Total one-to-one mappings: {total_mappings}")
    
    if total_mappings > 0:
        print(f"• High confidence mappings: {total_high_conf} ({total_high_conf/total_mappings:.1%})")
    
    print(f"{'=' * 80}")
    print("✅ All done!")



    # # Add this at the end of your main execution, after saving results
    # mapper.print_unmapped_fields_with_closest_matches(enhanced_schemas, results, mapper) #print_unmapped_fields_simple(enhanced_schemas, results)

    # Add this at the end of your main execution, after saving results
    mapper.print_unmapped_fields_within_cluster_matches(enhanced_schemas, results, mapper, clusters)



 