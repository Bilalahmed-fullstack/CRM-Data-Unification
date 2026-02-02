import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import fasttext.util
import json
from sentence_transformers import SentenceTransformer
# import SentenceTransformer
# import llm
class FeatureEngineer:
    def __init__(self):
        """
        Initialize feature engineering for schema mapping.
        """
        # Initialize word embeddings (if available)
        self.word_embeddings = None
        self._load_word_embeddings()
        
    def _load_word_embeddings(self):
        """
        Load pre-trained word embeddings for semantic similarity.
        Uses fastText if available, otherwise uses TF-IDF.
        """
        try:
            # # Try to load fastText embeddings
            # fasttext.util.download_model('en', if_exists='ignore')
            # self.word_embeddings = fasttext.load_model('cc.en.300.bin')
            # print("✅ Loaded fastText word embeddings")


            # import gensim.downloader as api
            # # Load smaller model (~80MB)
            # self.word_embeddings = api.load('glove-wiki-gigaword-100')
            # print("✅ Loaded GloVe word embeddings (100-dim)")

            model_name='all-MiniLM-L6-v2'
            self.word_embeddings = SentenceTransformer(model_name)
        except:
            print("⚠️  Could not load fastText, will use TF-IDF for semantic similarity")
            self.word_embeddings = None
    
    def extract_file_name_features(self, enhanced_schemas: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract features for Criterion 1: File Name Similarity.
        
        Returns:
            Dictionary with filename as key and feature vector as value
        """
        file_features = {}
        
        for filename, schema in enhanced_schemas.items():
            # Normalize filename: lowercase, remove extension
            name_without_ext = filename.rsplit('.', 1)[0].lower()
            
            # Tokenize by common separators
            tokens = re.split(r'[_\-. ]+', name_without_ext)
            
            # Create feature vector
            features = {
                'tokens': tokens,
                'token_count': len(tokens),
                'name_length': len(name_without_ext),
                'has_digits': any(char.isdigit() for char in name_without_ext),
                'has_separators': any(char in '_-.' for char in name_without_ext)
            }
            
            file_features[filename] = features
        
        return file_features
    
    def calculate_file_name_similarity(self, file1_features: Dict, file2_features: Dict) -> float:
        """
        Calculate semantic similarity between two file names.
        
        Args:
            file1_features: Features for first file
            file2_features: Features for second file
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get tokens
        tokens1 = file1_features['tokens']
        tokens2 = file2_features['tokens']
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Method 1: Token overlap (Jaccard similarity)
        set1 = set(tokens1)
        set2 = set(tokens2)

        embeddings = self.word_embeddings.encode([set1, set2])
        semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        # Ensure it's between 0-1 (though cosine similarity usually is)
        semantic_similarity = max(0.0, min(1.0, semantic_similarity))
        # return semantic_similarity
        
        if set1 and set2:
            token_overlap = len(set1.intersection(set2)) / len(set1.union(set2))
        else:
            token_overlap = 0.0
        
        # # Method 2: Word embedding similarity (if available)
        # embedding_similarity = 0.0
        # if self.word_embeddings:
        #     try:
        #         # Get embeddings for each token and average
        #         vecs1 = []
        #         for token in tokens1:
        #             if token in self.word_embeddings:
        #                 vecs1.append(self.word_embeddings[token])
                        
                
        #         vecs2 = []
        #         for token in tokens2:
        #             if token in self.word_embeddings:
        #                 vecs2.append(self.word_embeddings[token])
                
        #         if vecs1 and vecs2:
        #             avg_vec1 = np.mean(vecs1, axis=0)
        #             avg_vec2 = np.mean(vecs2, axis=0)
        #             embedding_similarity = cosine_similarity([avg_vec1], [avg_vec2])[0][0]
        #             # Normalize to 0-1
        #             embedding_similarity = (embedding_similarity + 1) / 2
        #     except:
        #         embedding_similarity = 0.0
        
        # Method 3: Length similarity
        len1 = file1_features['name_length']
        len2 = file2_features['name_length']
        length_similarity = 1 - abs(len1 - len2) / max(len1, len2, 1)
        
        # Combine scores (weighted average)
        if self.word_embeddings:
            # Use embeddings if available
            similarity = 0.6 * semantic_similarity + 0.3 * token_overlap + 0.1 * length_similarity
        else:
            # Fallback to token overlap
            similarity = 0.8 * token_overlap + 0.2 * length_similarity
        
        return max(0.0, min(1.0, similarity))
    
    def extract_schema_length_features(self, enhanced_schemas: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract features for Criterion 2: Schema Length.
        
        Returns:
            Dictionary with schema length features for each file
        """
        length_features = {}
        
        for filename, schema in enhanced_schemas.items():
            if 'error' in schema:
                continue
            
            column_count = schema.get('column_count', 0)
            
            # Additional features based on column types
            column_types = schema.get('column_types', {})
            type_counts = defaultdict(int)
            for col_type in column_types.values():
                type_counts[col_type] += 1
            
            features = {
                'column_count': column_count,
                'string_columns': type_counts.get('string', 0),
                'numeric_columns': type_counts.get('integer', 0) + type_counts.get('float', 0),
                'datetime_columns': type_counts.get('datetime', 0),
                'boolean_columns': type_counts.get('boolean', 0)
            }
            
            # Add ratios
            if column_count > 0:
                features.update({
                    'string_ratio': features['string_columns'] / column_count,
                    'numeric_ratio': features['numeric_columns'] / column_count,
                    'datetime_ratio': features['datetime_columns'] / column_count
                })
            else:
                features.update({
                    'string_ratio': 0.0,
                    'numeric_ratio': 0.0,
                    'datetime_ratio': 0.0
                })
            
            length_features[filename] = features
        
        return length_features
    
    def calculate_schema_length_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Calculate similarity based on schema length features.
        
        Returns:
            Similarity score between 0 and 1
        """
        # Column count similarity (normalized difference)
        count1 = features1['column_count']
        count2 = features2['column_count']
        
        if count1 == 0 and count2 == 0:
            count_similarity = 1.0
        elif count1 == 0 or count2 == 0:
            count_similarity = 0.0
        else:
            count_similarity = 1 - abs(count1 - count2) / max(count1, count2)
        
        # Column type distribution similarity
        type_features = ['string_ratio', 'numeric_ratio', 'datetime_ratio']
        type_similarities = []
        
        for feat in type_features:
            val1 = features1.get(feat, 0)
            val2 = features2.get(feat, 0)
            sim = 1 - abs(val1 - val2)
            type_similarities.append(sim)
        
        avg_type_similarity = np.mean(type_similarities) if type_similarities else 0.0
        
        # Combined score (weighted)
        similarity = 0.6 * count_similarity + 0.4 * avg_type_similarity
        
        return max(0.0, min(1.0, similarity))
    
    # def extract_column_name_features(self, enhanced_schemas: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    #     """
    #     Extract features for Criterion 3: Column Name Similarity.
        
    #     Returns:
    #         Dictionary with column name features for each file
    #     """
    #     column_features = {}
        
    #     for filename, schema in enhanced_schemas.items():
    #         if 'error' in schema:
    #             continue
            
    #         columns = schema.get('columns', [])
    #         features = {
    #             'column_names': columns,
    #             'column_name_tokens': [],
    #             'name_patterns': defaultdict(int)
    #         }
            
    #         # Analyze each column name
    #         for col in columns:
    #             # Tokenize column name
    #             col_lower = col.lower()
    #             tokens = re.split(r'[_\-. ]+', col_lower)
    #             features['column_name_tokens'].extend(tokens)
                
    #             # Detect naming patterns
    #             if col_lower.endswith('_id') or col_lower.endswith('id'):
    #                 features['name_patterns']['id_pattern'] += 1
    #             if col_lower.endswith('_name') or col_lower.endswith('name'):
    #                 features['name_patterns']['name_pattern'] += 1
    #             if col_lower.endswith('_date') or col_lower.endswith('date') or 'timestamp' in col_lower:
    #                 features['name_patterns']['date_pattern'] += 1
    #             if col_lower.endswith('_email') or 'email' in col_lower:
    #                 features['name_patterns']['email_pattern'] += 1
    #             if col_lower.endswith('_phone') or 'phone' in col_lower or 'tel' in col_lower:
    #                 features['name_patterns']['phone_pattern'] += 1
                
    #             # Count word types
    #             if any(word in col_lower for word in ['first', 'last', 'full', 'name']):
    #                 features['name_patterns']['name_related'] += 1
    #             if any(word in col_lower for word in ['addr', 'street', 'city', 'state', 'zip']):
    #                 features['name_patterns']['address_related'] += 1
    #             if any(word in col_lower for word in ['price', 'cost', 'amount', 'total', 'fee']):
    #                 features['name_patterns']['price_related'] += 1
            
    #         column_features[filename] = features
        
    #     return column_features
    
    def extract_column_name_features(self, enhanced_schemas: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract features for Criterion 3: Column Name Similarity.
        Generic pattern detection without hardcoded domain knowledge.
        
        Returns:
            Dictionary with column name features for each file
        """
        column_features = {}
        
        for filename, schema in enhanced_schemas.items():
            if 'error' in schema:
                continue
            
            columns = schema.get('columns', [])
            features = {
                'column_names': columns,
                'column_name_tokens': [],
                'token_frequencies': defaultdict(int),
                'suffix_patterns': defaultdict(int),
                'prefix_patterns': defaultdict(int),
                'compound_patterns': defaultdict(int),
                'token_positions': defaultdict(list)
            }
            
            # Analyze each column name
            for idx, col in enumerate(columns):

                col_lower = col.lower()
                
                # Tokenize column name
                tokens = re.split(r'[_\-. ]+', col_lower)
                features['column_name_tokens'].extend(tokens)
                
                # Track token frequencies
                for token in tokens:
                    features['token_frequencies'][token] += 1
                
                # Dynamic pattern discovery
                # 1. Suffix patterns (last token)
                if len(tokens) > 1:
                    suffix = tokens[-1]
                    features['suffix_patterns'][suffix] += 1
                
                # 2. Prefix patterns (first token)
                if tokens:
                    prefix = tokens[0]
                    features['prefix_patterns'][prefix] += 1
                
                # 3. Compound patterns (token pairs)
                for i in range(len(tokens) - 1):
                    compound = f"{tokens[i]}_{tokens[i+1]}"
                    features['compound_patterns'][compound] += 1
                
                # 4. Track token positions for semantic analysis
                for token in tokens:
                    features['token_positions'][token].append(idx)
                
                # 5. Character pattern analysis
                char_pattern = self._extract_char_pattern(col_lower)
                features['compound_patterns'][f"charpat_{char_pattern}"] = \
                    features['compound_patterns'].get(f"charpat_{char_pattern}", 0) + 1
            
            # Calculate token statistics
            total_tokens = len(features['column_name_tokens'])
            if total_tokens > 0:
                features['token_statistics'] = {
                    'unique_tokens': len(set(features['column_name_tokens'])),
                    'token_diversity': len(set(features['column_name_tokens'])) / total_tokens,
                    'avg_tokens_per_column': total_tokens / len(columns) if columns else 0,
                    'most_common_tokens': sorted(
                        features['token_frequencies'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                }
            
            # Discover dominant patterns dynamically
            features['dominant_patterns'] = self._discover_dominant_patterns(features)
            
            column_features[filename] = features
        
        return column_features
    

    def _extract_char_pattern(self, text: str) -> str:
        """
        Extract character pattern from text.
        Example: "customer_id" -> "LLLLLLLLL_LL"
        """
        pattern = []
        for char in text:
            if char.isalpha():
                pattern.append('L')
            elif char.isdigit():
                pattern.append('D')
            elif char in '_-.':
                pattern.append('S')  # Separator
            else:
                pattern.append('O')  # Other
        
        # Simplify pattern by grouping consecutive same characters
        simplified = []
        prev_char = None
        count = 0
        
        for char in pattern:
            if char == prev_char:
                count += 1
            else:
                if prev_char:
                    if count > 1:
                        simplified.append(f"{prev_char}{count}")
                    else:
                        simplified.append(prev_char)
                prev_char = char
                count = 1
        
        # Add last character
        if prev_char:
            if count > 1:
                simplified.append(f"{prev_char}{count}")
            else:
                simplified.append(prev_char)
        
        return ''.join(simplified)

    def _discover_dominant_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover dominant patterns in column names dynamically.
        No hardcoded domain knowledge.
        """
        dominant_patterns = {
            'suffixes': [],
            'prefixes': [],
            'compounds': [],
            'semantic_clusters': []
        }
        
        # 1. Discover significant suffixes (appear in >20% of columns)
        total_columns = len(features['column_names'])
        if total_columns > 0:
            for suffix, count in features['suffix_patterns'].items():
                frequency = count / total_columns
                if frequency > 0.2:  # At least 20% of columns have this suffix
                    dominant_patterns['suffixes'].append({
                        'suffix': suffix,
                        'frequency': frequency,
                        'count': count
                    })
        
        # 2. Discover significant prefixes
        for prefix, count in features['prefix_patterns'].items():
            frequency = count / total_columns
            if frequency > 0.2:
                dominant_patterns['prefixes'].append({
                    'prefix': prefix,
                    'frequency': frequency,
                    'count': count
                })
        
        # 3. Discover significant compound patterns
        for compound, count in features['compound_patterns'].items():
            frequency = count / total_columns
            if frequency > 0.1:  # Lower threshold for compounds
                dominant_patterns['compounds'].append({
                    'compound': compound,
                    'frequency': frequency,
                    'count': count
                })
        
        # 4. Discover semantic token clusters
        token_freq = features['token_frequencies']
        if token_freq:
            # Group tokens by character pattern similarity
            token_patterns = {}
            for token in token_freq.keys():
                pattern = self._extract_char_pattern(token)
                if pattern not in token_patterns:
                    token_patterns[pattern] = []
                token_patterns[pattern].append(token)
            
            # Find patterns with multiple tokens (potential semantic clusters)
            for pattern, tokens in token_patterns.items():
                if len(tokens) > 1:
                    total_count = sum(token_freq[t] for t in tokens)
                    avg_freq = total_count / len(tokens)
                    
                    if avg_freq > 1:  # Each token appears at least once on average
                        dominant_patterns['semantic_clusters'].append({
                            'pattern': pattern,
                            'tokens': tokens,
                            'avg_frequency': avg_freq,
                            'total_occurrences': total_count
                        })
        
        # Sort patterns by frequency/importance
        for key in ['suffixes', 'prefixes', 'compounds']:
            dominant_patterns[key].sort(key=lambda x: x['frequency'], reverse=True)
        
        dominant_patterns['semantic_clusters'].sort(
            key=lambda x: x['total_occurrences'], 
            reverse=True
        )
        
        return dominant_patterns
    









    # def calculate_column_name_similarity(self, file1: str, file2: str, 
    #                                     column_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    #     """
    #     Calculate similarity between column names of two files.
        
    #     Returns:
    #         Tuple of (similarity_score, column_mapping_details)
    #     """
    #     if file1 not in column_features or file2 not in column_features:
    #         return 0.0, {}
        
    #     features1 = column_features[file1]
    #     features2 = column_features[file2]
        
    #     cols1 = features1['column_names']
    #     cols2 = features2['column_names']
        
    #     if not cols1 or not cols2:
    #         return 0.0, {}
        
    #     # Method 1: Exact string matching (case-insensitive)
    #     cols1_lower = [c.lower() for c in cols1]
    #     cols2_lower = [c.lower() for c in cols2]
        
    #     exact_matches = set(cols1_lower).intersection(set(cols2_lower))
    #     exact_similarity = len(exact_matches) / max(len(cols1), len(cols2))
        
    #     # Method 2: Edit distance for partial matches
    #     partial_matches = []
    #     for col1 in cols1_lower:
    #         best_match_score = 0
    #         best_match = None
            
    #         for col2 in cols2_lower:
    #             # Calculate normalized edit distance
    #             from difflib import SequenceMatcher
    #             similarity = SequenceMatcher(None, col1, col2).ratio()
                
    #             if similarity > best_match_score and similarity > 0.7:  # Threshold
    #                 best_match_score = similarity
    #                 best_match = col2
            
    #         if best_match:
    #             partial_matches.append((col1, best_match, best_match_score))
        
    #     partial_similarity = sum(score for _, _, score in partial_matches) / max(len(cols1), len(cols2))
        
    #     # Method 3: Semantic similarity using word embeddings
    #     semantic_similarity = 0.0
    #     if self.word_embeddings:
    #         try:
    #             # Get embeddings for all column name tokens
    #             tokens1 = features1['column_name_tokens']
    #             tokens2 = features2['column_name_tokens']
                
    #             if tokens1 and tokens2:
    #                 # Get unique embeddings
    #                 vecs1 = []
    #                 for token in set(tokens1):
    #                     if token in self.word_embeddings:
    #                         vecs1.append(self.word_embeddings[token])
                    
    #                 vecs2 = []
    #                 for token in set(tokens2):
    #                     if token in self.word_embeddings:
    #                         vecs2.append(self.word_embeddings[token])
                    
    #                 if vecs1 and vecs2:
    #                     avg_vec1 = np.mean(vecs1, axis=0)
    #                     avg_vec2 = np.mean(vecs2, axis=0)
    #                     cos_sim = cosine_similarity([avg_vec1], [avg_vec2])[0][0]
    #                     semantic_similarity = (cos_sim + 1) / 2  # Normalize to 0-1
    #         except:
    #             semantic_similarity = 0.0
        
    #     # Method 4: Pattern similarity
    #     patterns1 = features1['name_patterns']
    #     patterns2 = features2['name_patterns']
        
    #     all_patterns = set(patterns1.keys()).union(set(patterns2.keys()))
    #     pattern_similarities = []
        
    #     for pattern in all_patterns:
    #         count1 = patterns1.get(pattern, 0)
    #         count2 = patterns2.get(pattern, 0)
    #         total_cols1 = len(cols1)
    #         total_cols2 = len(cols2)
            
    #         if total_cols1 > 0 and total_cols2 > 0:
    #             ratio1 = count1 / total_cols1
    #             ratio2 = count2 / total_cols2
    #             pattern_sim = 1 - abs(ratio1 - ratio2)
    #             pattern_similarities.append(pattern_sim)
        
    #     pattern_similarity = np.mean(pattern_similarities) if pattern_similarities else 0.0
        
    #     # Combine all similarity measures
    #     weights = {
    #         'exact': 0.3,
    #         'partial': 0.3,
    #         'semantic': 0.2,
    #         'pattern': 0.2
    #     }
        
    #     total_similarity = (
    #         weights['exact'] * exact_similarity +
    #         weights['partial'] * partial_similarity +
    #         weights['semantic'] * semantic_similarity +
    #         weights['pattern'] * pattern_similarity
    #     )
        
    #     # Create mapping details
    #     mapping_details = {
    #         'exact_matches': list(exact_matches),
    #         'partial_matches': partial_matches,
    #         'pattern_similarity': pattern_similarity,
    #         'column_counts': (len(cols1), len(cols2))
    #     }
        
    #     return max(0.0, min(1.0, total_similarity)), mapping_details
    def calculate_column_name_similarity(self, file1: str, file2: str, 
                                    column_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate similarity between column names of two files.
        Generic version without hardcoded patterns.
        """
        if file1 not in column_features or file2 not in column_features:
            return 0.0, {}
        
        features1 = column_features[file1]
        features2 = column_features[file2]
        
        cols1 = features1['column_names']
        cols2 = features2['column_names']
        
        if not cols1 or not cols2:
            return 0.0, {}
        
        # 1. Exact string matching (case-insensitive)
        cols1_lower = [c.lower() for c in cols1]
        cols2_lower = [c.lower() for c in cols2]
        
        exact_matches = set(cols1_lower).intersection(set(cols2_lower))
        exact_similarity = len(exact_matches) / max(len(cols1), len(cols2))
        
        # 2. Edit distance for partial matches
        partial_matches = []
        for col1 in cols1_lower:
            best_match_score = 0
            best_match = None
            
            for col2 in cols2_lower:
                # Skip if already exact match
                if col1 == col2:
                    continue
                    
                # Calculate normalized edit distance
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, col1, col2).ratio()
                
                if similarity > best_match_score and similarity > 0.7:  # Threshold
                    best_match_score = similarity
                    best_match = col2
            
            if best_match:
                partial_matches.append((col1, best_match, best_match_score))
        
        partial_similarity = sum(score for _, _, score in partial_matches) / max(len(cols1), len(cols2))
        
        # 3. Semantic similarity using word embeddings
        semantic_similarity = self._calculate_semantic_similarity(features1, features2)
        
        # 4. Pattern similarity (dynamic patterns)
        pattern_similarity = self._calculate_pattern_similarity(features1, features2)
        
        # 5. Token distribution similarity
        token_similarity = self._calculate_token_similarity(features1, features2)
        
        # Combine all similarity measures
        weights = {
            'exact': 0.25,
            'partial': 0.25,
            'semantic': 0.20,
            'pattern': 0.15,
            'token': 0.15
        }
        
        total_similarity = (
            weights['exact'] * exact_similarity +
            weights['partial'] * partial_similarity +
            weights['semantic'] * semantic_similarity +
            weights['pattern'] * pattern_similarity +
            weights['token'] * token_similarity
        )
        
        # Create mapping details
        mapping_details = {
            'exact_matches': list(exact_matches),
            'partial_matches': partial_matches,
            'pattern_similarity': pattern_similarity,
            'token_similarity': token_similarity,
            'semantic_similarity': semantic_similarity,
            'column_counts': (len(cols1), len(cols2))
        }
        
        return max(0.0, min(1.0, total_similarity)), mapping_details
    def _calculate_semantic_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate semantic similarity between column name tokens."""
        if not self.word_embeddings:
            return 0.5  # Default middle value if no embeddings
        
        tokens1 = list(set(features1['column_name_tokens']))
        tokens2 = list(set(features2['column_name_tokens']))
        
        if not tokens1 or not tokens2:
            return 0.5
        
        try:
            # Create meaningful phrases from tokens
            phrase1 = ' '.join(tokens1)
            phrase2 = ' '.join(tokens2)
            embeddings = self.word_embeddings.encode([phrase1, phrase2])
            sentence_transformer_similarity = cosine_similarity(
                [embeddings[0]], 
                [embeddings[1]]
            )[0][0]
            # Ensure it's between 0-1
            sentence_transformer_similarity = max(0.0, min(1.0, sentence_transformer_similarity))
            return sentence_transformer_similarity
            # Get embeddings for tokens
            # vecs1 = []
            # for token in tokens1:
            #     if token in self.word_embeddings:
            #         vecs1.append(self.word_embeddings[token])
            
            # vecs2 = []
            # for token in tokens2:
            #     if token in self.word_embeddings:
            #         vecs2.append(self.word_embeddings[token])
            
            # if vecs1 and vecs2:
            #     avg_vec1 = np.mean(vecs1, axis=0)
            #     avg_vec2 = np.mean(vecs2, axis=0)
            #     cos_sim = cosine_similarity([avg_vec1], [avg_vec2])[0][0]
            #     return (cos_sim + 1) / 2  # Normalize to 0-1
        except:
            pass
        
        return 0.5
    def _calculate_pattern_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity based on discovered patterns."""
        patterns1 = features1.get('dominant_patterns', {})
        patterns2 = features2.get('dominant_patterns', {})
        
        if not patterns1 or not patterns2:
            return 0.5
        
        pattern_similarities = []
        
        # Compare suffix patterns
        suffixes1 = {p['suffix']: p['frequency'] for p in patterns1.get('suffixes', [])}
        suffixes2 = {p['suffix']: p['frequency'] for p in patterns2.get('suffixes', [])}
        
        if suffixes1 or suffixes2:
            suffix_sim = self._compare_pattern_dicts(suffixes1, suffixes2)
            pattern_similarities.append(suffix_sim)
        
        # Compare prefix patterns
        prefixes1 = {p['prefix']: p['frequency'] for p in patterns1.get('prefixes', [])}
        prefixes2 = {p['prefix']: p['frequency'] for p in patterns2.get('prefixes', [])}
        
        if prefixes1 or prefixes2:
            prefix_sim = self._compare_pattern_dicts(prefixes1, prefixes2)
            pattern_similarities.append(prefix_sim)
        
        # Compare token statistics
        stats1 = features1.get('token_statistics', {})
        stats2 = features2.get('token_statistics', {})
        
        if stats1 and stats2:
            # Compare token diversity
            div1 = stats1.get('token_diversity', 0.5)
            div2 = stats2.get('token_diversity', 0.5)
            div_sim = 1 - abs(div1 - div2)
            pattern_similarities.append(div_sim)
        
        return np.mean(pattern_similarities) if pattern_similarities else 0.5

    def _compare_pattern_dicts(self, dict1: Dict, dict2: Dict) -> float:
        """Compare two pattern frequency dictionaries."""
        if not dict1 and not dict2:
            return 1.0  # Both empty = perfect match
        if not dict1 or not dict2:
            return 0.0  # One empty = no match
        
        all_patterns = set(dict1.keys()).union(set(dict2.keys()))
        similarities = []
        
        for pattern in all_patterns:
            freq1 = dict1.get(pattern, 0)
            freq2 = dict2.get(pattern, 0)
            sim = 1 - abs(freq1 - freq2)
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    def _calculate_token_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity based on token distributions."""
        token_freq1 = features1.get('token_frequencies', {})
        token_freq2 = features2.get('token_frequencies', {})
        
        if not token_freq1 and not token_freq2:
            return 1.0
        if not token_freq1 or not token_freq2:
            return 0.0
        
        # Normalize frequencies
        total1 = sum(token_freq1.values())
        total2 = sum(token_freq2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        norm_freq1 = {k: v/total1 for k, v in token_freq1.items()}
        norm_freq2 = {k: v/total2 for k, v in token_freq2.items()}
        
        # Calculate Jensen-Shannon divergence (symmetrized KL divergence)
        all_tokens = set(norm_freq1.keys()).union(set(norm_freq2.keys()))
        
        # Convert to probability vectors
        prob1 = np.array([norm_freq1.get(t, 0) for t in all_tokens])
        prob2 = np.array([norm_freq2.get(t, 0) for t in all_tokens])
        
        # Calculate JS divergence
        m = 0.5 * (prob1 + prob2)
        js_div = 0.5 * (self._kl_divergence(prob1, m) + self._kl_divergence(prob2, m))
        
        # Convert divergence to similarity (0=identical, 1=different)
        # JS divergence ranges from 0 to log(2)
        similarity = 1 - (js_div / np.log(2))
        
        return max(0.0, min(1.0, similarity))
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence with smoothing."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_smooth = p + epsilon
        q_smooth = q + epsilon
        
        # Normalize
        p_smooth = p_smooth / np.sum(p_smooth)
        q_smooth = q_smooth / np.sum(q_smooth)
        
        return np.sum(p_smooth * np.log(p_smooth / q_smooth))


























    def extract_data_type_features(self, enhanced_schemas: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract features for Criterion 4: Data Type Compatibility.
        
        Returns:
            Dictionary with data type features for each file
        """
        type_features = {}
        
        for filename, schema in enhanced_schemas.items():
            if 'error' in schema:
                continue
            
            column_types = schema.get('column_types', {})
            value_patterns = schema.get('value_patterns', {})
            
            features = {
                'column_types': column_types,
                'type_distribution': defaultdict(int),
                'pattern_features': {}
            }
            
            # Analyze type distribution
            for col, col_type in column_types.items():
                features['type_distribution'][col_type] += 1
                
                # Extract pattern features for each column
                if col in value_patterns:
                    patterns = value_patterns[col]
                    features['pattern_features'][col] = {
                        'char_distribution': patterns.get('char_distribution', {}),
                        'length_stats': patterns.get('length_stats', {}),
                        'unique_ratio': patterns.get('unique_ratio', 0),
                        'digit_ratio': patterns.get('digit_ratio', 0)
                    }
            
            type_features[filename] = features
        
        return type_features
    
    def calculate_data_type_similarity(self, file1: str, file2: str,
                                      type_features: Dict[str, Any],
                                      column_mapping: List[Tuple[str, str, float]]) -> float:
        """
        Calculate similarity based on data types and value patterns.
        
        Args:
            file1, file2: File names
            type_features: Data type features dictionary
            column_mapping: List of (col1, col2, similarity_score) from column name matching
            
        Returns:
            Similarity score between 0 and 1
        """
        if file1 not in type_features or file2 not in type_features:
            return 0.0
        
        features1 = type_features[file1]
        features2 = type_features[file2]
        
        if not column_mapping:
            return 0.0
        
        # For each mapped column pair, calculate type compatibility
        type_compatibilities = []
        pattern_compatibilities = []
        
        for col1, col2, name_similarity in column_mapping:
            # Type compatibility
            type1 = features1['column_types'].get(col1, 'unknown')
            type2 = features2['column_types'].get(col2, 'unknown')
            
            if type1 == type2:
                type_score = 1.0
            elif (type1 in ['integer', 'float'] and type2 in ['integer', 'float']):
                type_score = 0.8  # Both numeric
            elif (type1 == 'string' and type2 != 'string') or (type2 == 'string' and type1 != 'string'):
                type_score = 0.3  # One is string, other is not
            else:
                type_score = 0.5  # Different but compatible
            
            type_compatibilities.append(type_score)
            
            # Pattern compatibility (if patterns available)
            patterns1 = features1['pattern_features'].get(col1, {})
            patterns2 = features2['pattern_features'].get(col2, {})
            
            if patterns1 and patterns2:
                # Compare character distributions
                char_dist1 = patterns1.get('char_distribution', {})
                char_dist2 = patterns2.get('char_distribution', {})
                
                if char_dist1 and char_dist2:
                    char_similarities = []
                    for char_type in set(char_dist1.keys()).union(set(char_dist2.keys())):
                        val1 = char_dist1.get(char_type, 0)
                        val2 = char_dist2.get(char_type, 0)
                        char_sim = 1 - abs(val1 - val2)
                        char_similarities.append(char_sim)
                    
                    char_similarity = np.mean(char_similarities) if char_similarities else 0.5
                else:
                    char_similarity = 0.5
                
                # Compare length statistics
                len_stats1 = patterns1.get('length_stats', {})
                len_stats2 = patterns2.get('length_stats', {})
                
                if len_stats1 and len_stats2:
                    len_mean1 = len_stats1.get('mean', 0)
                    len_mean2 = len_stats2.get('mean', 0)
                    len_similarity = 1 - abs(len_mean1 - len_mean2) / max(len_mean1, len_mean2, 1)
                else:
                    len_similarity = 0.5
                
                # Compare uniqueness
                unique1 = patterns1.get('unique_ratio', 0.5)
                unique2 = patterns2.get('unique_ratio', 0.5)
                unique_similarity = 1 - abs(unique1 - unique2)
                
                # Combine pattern similarities
                pattern_score = (char_similarity + len_similarity + unique_similarity) / 3
                pattern_compatibilities.append(pattern_score)
        
        # Calculate average compatibilities
        avg_type_compatibility = np.mean(type_compatibilities) if type_compatibilities else 0.0
        avg_pattern_compatibility = np.mean(pattern_compatibilities) if pattern_compatibilities else 0.5
        
        # Combined score (weighted)
        similarity = 0.6 * avg_type_compatibility + 0.4 * avg_pattern_compatibility
        
        return max(0.0, min(1.0, similarity))
    
    def extract_value_pattern_features(self, enhanced_schemas: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract features for Criterion 5: Value Pattern Analysis.
        
        Returns:
            Dictionary with value pattern features for each file
        """
        pattern_features = {}
        
        for filename, schema in enhanced_schemas.items():
            if 'error' in schema:
                continue
            
            value_patterns = schema.get('value_patterns', {})
            
            features = {
                'pattern_summary': {},
                'column_patterns': {}
            }
            
            # Summarize patterns across all columns
            all_char_dist = defaultdict(float)
            all_length_stats = []
            all_digit_ratios = []
            all_unique_ratios = []
            
            pattern_samples = []
            
            for col, patterns in value_patterns.items():
                features['column_patterns'][col] = patterns
                
                # Collect statistics
                char_dist = patterns.get('char_distribution', {})
                for char_type, ratio in char_dist.items():
                    all_char_dist[char_type] += ratio
                
                length_stats = patterns.get('length_stats', {})
                if length_stats:
                    all_length_stats.append(length_stats.get('mean', 0))
                
                all_digit_ratios.append(patterns.get('digit_ratio', 0))
                all_unique_ratios.append(patterns.get('unique_ratio', 0))
                
                # Collect separator pattern samples
                sep_patterns = patterns.get('separator_patterns', {})
                if 'dominant_pattern_groups' in sep_patterns:
                    for group in sep_patterns['dominant_pattern_groups'][:2]:  # Top 2 groups
                        pattern_samples.append({
                            'pattern': group.get('example_pattern', ''),
                            'frequency': group.get('frequency', 0),
                            'value': group.get('example_value', '')
                        })
            
            # Create summary
            if all_char_dist:
                # Normalize character distribution
                total = sum(all_char_dist.values())
                features['pattern_summary']['char_distribution'] = {
                    k: v/total for k, v in all_char_dist.items()
                }
            
            if all_length_stats:
                features['pattern_summary']['avg_length'] = np.mean(all_length_stats)
                features['pattern_summary']['length_variability'] = np.std(all_length_stats)
            
            if all_digit_ratios:
                features['pattern_summary']['avg_digit_ratio'] = np.mean(all_digit_ratios)
            
            if all_unique_ratios:
                features['pattern_summary']['avg_unique_ratio'] = np.mean(all_unique_ratios)
            
            features['pattern_summary']['pattern_samples'] = pattern_samples[:5]  # Top 5 patterns
            
            pattern_features[filename] = features
        
        return pattern_features
    
    def calculate_value_pattern_similarity(self, file1: str, file2: str,
                                         pattern_features: Dict[str, Any],
                                         column_mapping: List[Tuple[str, str, float]]) -> float:
        """
        Calculate similarity based on value patterns.
        
        Returns:
            Similarity score between 0 and 1
        """
        if file1 not in pattern_features or file2 not in pattern_features:
            return 0.0
        
        features1 = pattern_features[file1]
        features2 = pattern_features[file2]
        
        if not column_mapping:
            # Compare overall pattern summaries
            summary1 = features1['pattern_summary']
            summary2 = features2['pattern_summary']
            
            similarities = []
            
            # Compare character distributions
            char_dist1 = summary1.get('char_distribution', {})
            char_dist2 = summary2.get('char_distribution', {})
            
            if char_dist1 and char_dist2:
                char_similarities = []
                for char_type in set(char_dist1.keys()).union(set(char_dist2.keys())):
                    val1 = char_dist1.get(char_type, 0)
                    val2 = char_dist2.get(char_type, 0)
                    char_sim = 1 - abs(val1 - val2)
                    char_similarities.append(char_sim)
                
                similarities.append(np.mean(char_similarities) if char_similarities else 0.5)
            
            # Compare average digit ratio
            digit1 = summary1.get('avg_digit_ratio', 0.5)
            digit2 = summary2.get('avg_digit_ratio', 0.5)
            digit_similarity = 1 - abs(digit1 - digit2)
            similarities.append(digit_similarity)
            
            # Compare average uniqueness
            unique1 = summary1.get('avg_unique_ratio', 0.5)
            unique2 = summary2.get('avg_unique_ratio', 0.5)
            unique_similarity = 1 - abs(unique1 - unique2)
            similarities.append(unique_similarity)
            
            if similarities:
                return np.mean(similarities)
            else:
                return 0.5
        
        else:
            # Compare patterns for mapped columns
            pattern_similarities = []
            
            for col1, col2, _ in column_mapping:
                patterns1 = features1['column_patterns'].get(col1, {})
                patterns2 = features2['column_patterns'].get(col2, {})
                
                if not patterns1 or not patterns2:
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
                
                # Length statistics
                len_stats1 = patterns1.get('length_stats', {})
                len_stats2 = patterns2.get('length_stats', {})
                
                if len_stats1 and len_stats2:
                    mean1 = len_stats1.get('mean', 0)
                    mean2 = len_stats2.get('mean', 0)
                    if mean1 > 0 or mean2 > 0:
                        len_sim = 1 - abs(mean1 - mean2) / max(mean1, mean2, 1)
                        feature_similarities.append(len_sim)
                
                # Digit ratio
                digit1 = patterns1.get('digit_ratio', 0.5)
                digit2 = patterns2.get('digit_ratio', 0.5)
                digit_sim = 1 - abs(digit1 - digit2)
                feature_similarities.append(digit_sim)
                
                # Unique ratio
                unique1 = patterns1.get('unique_ratio', 0.5)
                unique2 = patterns2.get('unique_ratio', 0.5)
                unique_sim = 1 - abs(unique1 - unique2)
                feature_similarities.append(unique_sim)
                
                # Separator patterns
                sep_patterns1 = patterns1.get('separator_patterns', {})
                sep_patterns2 = patterns2.get('separator_patterns', {})
                
                if sep_patterns1 and sep_patterns2:
                    # Compare dominant pattern groups
                    groups1 = sep_patterns1.get('dominant_pattern_groups', [])
                    groups2 = sep_patterns2.get('dominant_pattern_groups', [])
                    
                    if groups1 and groups2:
                        # Compare pattern diversity
                        div1 = sep_patterns1.get('pattern_diversity', 0.5)
                        div2 = sep_patterns2.get('pattern_diversity', 0.5)
                        div_sim = 1 - abs(div1 - div2)
                        feature_similarities.append(div_sim)
                
                if feature_similarities:
                    pattern_similarities.append(np.mean(feature_similarities))
            
            if pattern_similarities:
                return np.mean(pattern_similarities)
            else:
                return 0.5
    def get_descriptions(self,enhanced_schemas):
        descriptions = {}

        for filename, schema in enhanced_schemas.items():
            # Skip files with errors or missing schema
            if not isinstance(schema, dict):
                continue

            file_descriptions = schema.get('description', {})

            # Ensure we always return a dict
            descriptions[filename] = {}

            for column, desc in file_descriptions.items():
                descriptions[filename][column] = desc
        return descriptions
    def extract_all_features(self, enhanced_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features for all criteria.
        
        Returns:
            Dictionary containing all extracted features
        """
        print("\n" + "=" * 80)
        print("EXTRACTING FEATURES FOR ALL CRITERIA")
        print("=" * 80)
        
        all_features = {
            'file_names': self.extract_file_name_features(enhanced_schemas),
            'schema_lengths': self.extract_schema_length_features(enhanced_schemas),
            'column_names': self.extract_column_name_features(enhanced_schemas),
            'data_types': self.extract_data_type_features(enhanced_schemas),
            'description': self.get_descriptions(enhanced_schemas),
            'value_patterns': self.extract_value_pattern_features(enhanced_schemas)
        }
        
        print("✅ Feature extraction complete!")
        print(f"📊 Files processed: {len(enhanced_schemas)}")
        
        return all_features
def feature_engineering(schema_data):
    import pickle
    # with open('schema_data.json', 'r') as f:
    #     enhanced_schemas = json.load(f)
    enhanced_schemas = schema_data
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Extract all features
    all_features = feature_engineer.extract_all_features(enhanced_schemas)
    
    # Save features for Phase 3
    with open('all_features.json', 'w') as f:
        json.dump(all_features, f, indent=2, default=str)
    
    # print("💾 Features saved to: all_features.json")
    return all_features
# Example usage in main:
if __name__ == "__main__":
    # Load enhanced schemas (from Phase 1)
    import pickle
    with open('schema_data.json', 'r') as f:
        enhanced_schemas = json.load(f)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Extract all features
    all_features = feature_engineer.extract_all_features(enhanced_schemas)
    
    # Save features for Phase 3
    with open('all_features.json', 'w') as f:
        json.dump(all_features, f, indent=2, default=str)
    
    print("💾 Features saved to: all_features.json")