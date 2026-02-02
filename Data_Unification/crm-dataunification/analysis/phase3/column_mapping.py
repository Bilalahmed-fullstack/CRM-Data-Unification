# mapping_simplified.py
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
from llm_judge import describe_column

class SimplifiedColumnMapper:
    def __init__(self, enhanced_schemas: Dict[str, Any]):
        """
        Simplified column mapper using only schema_data.json and sentence transformers.
        No pattern learning from data.
        """
        self.enhanced_schemas = enhanced_schemas
        
        # Initialize sentence transformer for semantic similarity
        try:
            print("Loading sentence transformer model...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load sentence transformer: {e}")
            print("Will use basic string similarity only")
            self.semantic_model = None
    
    def map_all_clusters(self, clusters: List[List[str]]) -> Dict[str, Any]:
        """
        Create column mappings for all clusters.
        
        Args:
            clusters: List of clusters from Phase 3
            
        Returns:
            Dictionary with mapping results for each cluster
        """
        print("\n" + "=" * 80)
        print("SIMPLIFIED COLUMN MAPPING")
        print("=" * 80)
        
        all_results = {}
        
        for cluster_idx, cluster_files in enumerate(clusters, 1):
            print(f"\n📊 Processing Cluster {cluster_idx}: {len(cluster_files)} files")
            
            result = self._process_cluster(cluster_files, cluster_idx)
            all_results[f"Cluster_{cluster_idx}"] = result
        
        print(f"\n{'=' * 80}")
        print("✅ COLUMN MAPPING COMPLETE")
        print(f"{'=' * 80}")
        
        return all_results
    
    def _process_cluster(self, cluster_files: List[str], cluster_idx: int) -> Dict[str, Any]:
        """
        Process a single cluster: map columns and create golden schema.
        """
        if len(cluster_files) < 2:
            return self._handle_single_schema(cluster_files, cluster_idx)
        
        # Step 1: Extract column information for all schemas in cluster
        schema_columns = {}
        for filename in cluster_files:
            if filename in self.enhanced_schemas:
                schema_columns[filename] = self._extract_schema_info(filename)
        
        # Step 2: Calculate similarity matrices between all schema pairs
        similarity_matrices = self._calculate_similarity_matrices(schema_columns)
        
        results = []
        all_results = {}
        for schema_name, columns in schema_columns.items():
            reference_schema = schema_name
            all_results["reference_schema"]=reference_schema
            # Step 4: Create column mappings starting from reference
            column_groups = self._create_column_mappings(
                reference_schema, schema_columns, similarity_matrices
            )
            all_results["column_groups"]=column_groups
            # Step 5: Create golden schema from column groups
            golden_schema = self._create_golden_schema(column_groups, schema_columns)
            all_results["golden_schema"]=golden_schema
            # Step 6: Create mapping matrix
            mapping_matrix = self._create_mapping_matrix(golden_schema, schema_columns)
            all_results["mapping_matrix"]=mapping_matrix

            # Step 7: Calculate quality metrics
            all_results["quality_metrics"] = self._calculate_quality_metrics(column_groups, schema_columns)
            results.append(all_results)
        prev = results[0]
        for result in results[1:]:
            quality_metrics = self.pareto_dominance(prev["quality_metrics"], result["quality_metrics"])
            if prev["quality_metrics"] == quality_metrics:
                final = prev
            else:
                final = result
            prev = result

        
        return {
            'cluster_files': cluster_files,
            'reference_schema': final["reference_schema"],
            'golden_schema': final["golden_schema"],
            'mapping_matrix': final["mapping_matrix"],
            'quality_metrics': final["quality_metrics"],
            'column_groups': final["column_groups"]
        }
    def pareto_dominance(self,result1, result2):
        """Check if one result dominates the other (better in all metrics)"""
        
        metrics = ['coverage', 'average_similarity', 'median_similarity', 
                'min_similarity', 'max_similarity']
        
        # Higher is better for all metrics in this case
        result1_better = all(result1[m] >= result2[m] for m in metrics)
        result2_better = all(result2[m] >= result1[m] for m in metrics)
        
        if result1_better and not result2_better:
            return result1
        elif result2_better and not result1_better:
            result2
        elif result1_better and result2_better:
            return False
        else:
            return False


        # # Step 3: Find reference schema (most columns as baseline)
        # reference_schema = self._find_reference_schema(schema_columns)
        
        # # Step 4: Create column mappings starting from reference
        # column_groups = self._create_column_mappings(
        #     reference_schema, schema_columns, similarity_matrices
        # )
        
        # # Step 5: Create golden schema from column groups
        # golden_schema = self._create_golden_schema(column_groups, schema_columns)
        
        # # Step 6: Create mapping matrix
        # mapping_matrix = self._create_mapping_matrix(golden_schema, schema_columns)
        
        # # Step 7: Calculate quality metrics
        # quality_metrics = self._calculate_quality_metrics(column_groups, schema_columns)
        
        # return {
        #     'cluster_files': cluster_files,
        #     'reference_schema': reference_schema,
        #     'golden_schema': golden_schema,
        #     'mapping_matrix': mapping_matrix,
        #     'quality_metrics': quality_metrics,
        #     'column_groups': column_groups
        # }



    
    def _extract_schema_info(self, filename: str) -> Dict[str, Any]:
        """
        Extract column information from schema_data.json.
        
        Returns:
            Dictionary with column name, type, and basic features
        """
        if filename not in self.enhanced_schemas:
            return {}
        
        schema = self.enhanced_schemas[filename]
        columns = schema.get('columns', [])
        description = schema.get('description', [])
        # description_dict = self.enhanced_schemas['description'].get(filename, {})
        column_types = schema.get('column_types', {})
        

        column_info = {}
        for col in columns:
            col_type = column_types.get(col, 'unknown')
            
            # Simple feature extraction
            tokens = self._tokenize_column_name(col)
            
            column_info[col] = {
                'name': col,
                'description': description.get(col, 'no description'),
                'type': col_type,
                'tokens': tokens,
                'token_count': len(tokens),
                'has_separators': '_' in col or '-' in col
            }
        
        return column_info
    
    
    def _tokenize_column_name(self, column_name: str) -> List[str]:
        """Simple tokenization of column names."""
        # Convert to lowercase and split by common separators
        name_lower = column_name.lower()
        tokens = []
        
        # Split by underscores, hyphens, and camelCase
        import re
        # Split by underscores and hyphens
        parts = re.split(r'[_\-. ]+', name_lower)
        
        # Further split camelCase if any
        for part in parts:
            if part:
                # Simple splitting - keep as is for now
                tokens.append(part)
        
        return tokens
    
    def _calculate_similarity_matrices(self, schema_columns: Dict[str, Dict]) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Calculate semantic similarity matrices for all schema pairs.
        """
        similarity_matrices = {}
        schemas = list(schema_columns.keys())
        
        for i in range(len(schemas)):
            for j in range(i + 1, len(schemas)):
                schema1 = schemas[i]
                schema2 = schemas[j]
                
                sim_matrix = self._calculate_pairwise_similarity(
                    schema_columns[schema1], 
                    schema_columns[schema2]
                )
                
                similarity_matrices[(schema1, schema2)] = sim_matrix
                similarity_matrices[(schema2, schema1)] = sim_matrix.T
        
        return similarity_matrices
    
    def _calculate_pairwise_similarity(self, cols1: Dict, cols2: Dict) -> np.ndarray:
        """
        Calculate similarity matrix between two sets of columns.
        """
        col_names1 = list(cols1.keys())
        col_names2 = list(cols2.keys())
        
        n = len(col_names1)
        m = len(col_names2)
        
        sim_matrix = np.zeros((n, m))
        
        for i, col1_name in enumerate(col_names1):
            col1_info = cols1[col1_name]
            
            for j, col2_name in enumerate(col_names2):
                col2_info = cols2[col2_name]
                
                similarity = self._calculate_column_similarity(col1_info, col2_info)
                sim_matrix[i, j] = similarity
        
        return sim_matrix
    def _type_similarity(self, type1: str, type2: str) -> float:
        TYPE_SIMILARITY = {
            # Same types
            ('string', 'string'): 1.0,
            ('integer', 'integer'): 1.0,
            ('float', 'float'): 1.0,
            ('datetime', 'datetime'): 1.0,
            ('boolean', 'boolean'): 1.0,

            # Numeric compatibility
            ('integer', 'float'): 0.85,
            ('float', 'integer'): 0.85,

            # String as weakly compatible fallback
            ('string', 'integer'): 0.2,
            ('integer', 'string'): 0.2,
            ('string', 'float'): 0.2,
            ('float', 'string'): 0.2,
            ('string', 'boolean'): 0.1,
            ('boolean', 'string'): 0.1,
            ('string', 'datetime'): 0.15,
            ('datetime', 'string'): 0.15,

            # Datetime incompatibilities
            ('datetime', 'integer'): 0.0,
            ('integer', 'datetime'): 0.0,
            ('datetime', 'float'): 0.0,
            ('float', 'datetime'): 0.0,
            ('datetime', 'boolean'): 0.0,
            ('boolean', 'datetime'): 0.0,

            # Boolean incompatibilities
            ('boolean', 'integer'): 0.1,
            ('integer', 'boolean'): 0.1,
            ('boolean', 'float'): 0.1,
            ('float', 'boolean'): 0.1,
        }


        if not type1 or not type2:
            return 0.0
        if type1 == type2:
            return 1.0
        return TYPE_SIMILARITY.get((type1, type2), 0.0)

    def _calculate_column_similarity(self, col1: Dict, col2: Dict) -> float:
        """
        Calculate semantic similarity between two columns.
        Primary method: sentence transformer semantic similarity.
        """
        # Use sentence transformer if available
        if self.semantic_model:
            type_sim = self._type_similarity(col1['type'],col2['type'])
            value_pattern_sim = self._calculate_pattern_similarity_data_driven(col1, col2)
            sementic = self._calculate_semantic_similarity(col1, col2)
            similarity = value_pattern_sim*0.1 + 0.1*type_sim + 0.8*sementic
            if (col1['name'] == "customer_name" or col1['name'] == "user_id" or col1['name'] == "customer_id" or col1['name'] == "account_id") and col2['name'] in ["customer_name" , "user_id", "customer_id" , "account_id"]:
                print(col1['type'] , col1['name'] , col2['type'] , col2['name'] , similarity)
                print("Type:",type_sim)
                print("Pattern:",value_pattern_sim)
                print("Sementic:",sementic)
            return similarity
        else:
            # Fallback to basic string similarity
            return self._calculate_basic_similarity(col1, col2)
    
    def _calculate_semantic_similarity(self, col1: Dict, col2: Dict) -> float:
        """
        Calculate semantic similarity using sentence transformers.
        """
        name1 = col1['name']
        name2 = col2['name']
        # description1 = describe_column(str(name1))
        # description2 = describe_column(str(name2))
        description1 = col1['description']
        description2 = col2['description']
        # Exact match
        if name1.lower() == name2.lower():
            return 1.0
        
        try:
            # Encode both names
            name_embeddings = self.semantic_model.encode([name1, name2])
            name_similarity = cosine_similarity([name_embeddings[0]], [name_embeddings[1]])[0][0]
            # Ensure between 0 and 1
            name_similarity = max(0.0, min(1.0, name_similarity))

            # Encode both descriptions
            description_embeddings = self.semantic_model.encode([description1, description2])
            description_similarity = cosine_similarity([description_embeddings[0]], [description_embeddings[1]])[0][0]
            # Ensure between 0 and 1
            description_similarity = max(0.0, min(1.0, description_similarity))


            w_name = 0.4
            w_description = 0.6

            final_similarity = w_name * name_similarity + w_description * description_similarity


            
            return final_similarity
            
        except Exception as e:
            print(f"Error in semantic similarity for {name1} ↔ {name2}: {e}")
            return self._calculate_basic_similarity(col1, col2)
    
    def _calculate_basic_similarity(self, col1: Dict, col2: Dict) -> float:
        """
        Basic similarity calculation without sentence transformers.
        """
        from difflib import SequenceMatcher
        
        name1 = col1['name'].lower()
        name2 = col2['name'].lower()
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        # String similarity
        string_sim = SequenceMatcher(None, name1, name2).ratio()
        
        # Token overlap
        tokens1 = set(col1['tokens'])
        tokens2 = set(col2['tokens'])
        
        if tokens1 and tokens2:
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            token_sim = intersection / union if union > 0 else 0.0
        else:
            token_sim = 0.0
        
        # Type compatibility
        type_sim = 1.0 if col1['type'] == col2['type'] else 0.3
        
        # Weighted combination
        similarity = 0.5 * string_sim + 0.3 * token_sim + 0.2 * type_sim
        
        return similarity
    
    #*******************************************************************************************************************************************************
    def _calculate_pattern_similarity_data_driven(self, col1_info: Dict, col2_info: Dict) -> float:
        """
        Calculate pattern similarity using data-driven features.
        """
        similarities = []
        
        # Get data analysis from enhanced schemas if available
        schema1 = col1_info.get('source_schema', '')
        schema2 = col2_info.get('source_schema', '')
        col1_name = col1_info['name']
        col2_name = col2_info['name']
        
        # Try to get pattern analysis from enhanced schemas
        analysis1 = self._get_column_analysis(schema1, col1_name)
        analysis2 = self._get_column_analysis(schema2, col2_name)
        
        if analysis1 and analysis2:
            # Compare digit ratios
            digit_sim = 1.0 - min(abs(analysis1.get('digit_ratio', 0) - analysis2.get('digit_ratio', 0)), 1.0)
            similarities.append(digit_sim)
            
            # Compare unique ratios
            unique_sim = 1.0 - min(abs(analysis1.get('unique_ratio', 0.5) - analysis2.get('unique_ratio', 0.5)), 1.0)
            similarities.append(unique_sim)
            
            # Compare type consistency
            type_consistency_sim = 1.0 - min(abs(analysis1.get('type_consistency', 1.0) - analysis2.get('type_consistency', 1.0)), 1.0)
            similarities.append(type_consistency_sim)
            
            # Compare ID-like classification
            if analysis1.get('is_id_like', False) == analysis2.get('is_id_like', False):
                similarities.append(1.0)
            else:
                similarities.append(0.5)
            
            # Compare categorical classification
            if analysis1.get('is_categorical', False) == analysis2.get('is_categorical', False):
                similarities.append(1.0)
            else:
                similarities.append(0.5)
            
            # Pattern complexity similarity
            complexity_sim = 1.0 - min(abs(analysis1.get('pattern_complexity', 0) - analysis2.get('pattern_complexity', 0)), 1.0)
            similarities.append(complexity_sim)
            
            return np.mean(similarities)
        
        return 0.0  # Default moderate similarity if no pattern data
    def _get_column_analysis(self, schema_name: str, column_name: str) -> Dict:
        """
        Extract data-driven analysis for a column from enhanced schemas.
        """
        if schema_name not in self.enhanced_schemas:
            return {}
        
        schema = self.enhanced_schemas[schema_name]
        value_patterns = schema.get('value_patterns', {})
        value_samples = schema.get('value_samples', {})
        column_types = schema.get('column_types', {})
        
        if column_name not in value_patterns:
            return {}
        
        patterns = value_patterns[column_name]
        samples = value_samples.get(column_name, [])
        col_type = column_types.get(column_name, 'unknown')
        
        return self._analyze_data_patterns(patterns, samples, col_type)
    def _analyze_data_patterns(self, patterns: Dict, samples: List, col_type: str) -> Dict:
        """
        Analyze data patterns from learned distributions.
        """
        analysis = {
            'digit_ratio': 0.0,
            'unique_ratio': 0.5,  # Default moderate uniqueness
            'type_consistency': 1.0,  # Default high consistency
            'is_id_like': False,
            'is_categorical': False,
            'pattern_complexity': 0.0
        }
        
        if patterns:
            # Extract digit ratio from patterns if available
            if 'digit_ratio' in patterns:
                analysis['digit_ratio'] = patterns['digit_ratio']
            
            # Estimate unique ratio from patterns
            if 'unique_ratio' in patterns:
                analysis['unique_ratio'] = patterns['unique_ratio']
            elif 'unique_count' in patterns and 'total_count' in patterns:
                if patterns['total_count'] > 0:
                    analysis['unique_ratio'] = patterns['unique_count'] / patterns['total_count']
            
            # Check if looks like an ID (high uniqueness, often contains 'id' in name)
            if analysis['unique_ratio'] > 0.9:
                analysis['is_id_like'] = True
            
            # Check if categorical (low to moderate uniqueness)
            if 0.01 <= analysis['unique_ratio'] <= 0.3:
                analysis['is_categorical'] = True
            
            # Type consistency from patterns
            if 'type_consistency' in patterns:
                analysis['type_consistency'] = patterns['type_consistency']
        
        # Additional analysis from samples
        if samples:
            # Check if samples suggest categorical values (limited distinct values)
            if len(samples) > 0:
                distinct_samples = set(str(s).lower() for s in samples)
                if len(distinct_samples) / len(samples) < 0.5:
                    analysis['is_categorical'] = True
        
        return analysis
    #*******************************************************************************************************************************************************
    def _apply_pattern_boosts(self, similarity: float, col1: Dict, col2: Dict) -> float:
        """
        Apply pattern-based boosts to similarity score.
        """
        boosted_similarity = similarity
        
        # Check for common suffixes
        common_suffixes = ['id', 'name', 'date', 'time', 'price', 'amount', 
                          'quantity', 'status', 'type', 'code', 'number']
        
        name1 = col1['name'].lower()
        name2 = col2['name'].lower()
        
        # Boost if both end with same common suffix
        for suffix in common_suffixes:
            if name1.endswith(suffix) and name2.endswith(suffix):
                boosted_similarity = min(1.0, boosted_similarity + 0.1)
                break
        
        # Boost for exact token matches
        tokens1 = set(col1['tokens'])
        tokens2 = set(col2['tokens'])
        
        if tokens1 and tokens2:
            exact_token_matches = len(tokens1.intersection(tokens2))
            if exact_token_matches > 0:
                boost = min(0.2, exact_token_matches * 0.05)
                boosted_similarity = min(1.0, boosted_similarity + boost)
        
        return boosted_similarity
    
    def _find_reference_schema(self, schema_columns: Dict[str, Dict]) -> str:
        """
        Find the most representative schema to use as reference.
        Simple heuristic: schema with most columns.
        """
        if not schema_columns:
            return ""
        
        # Find schema with maximum number of columns
        max_columns = -1
        reference_schema = ""
        
        for schema_name, columns in schema_columns.items():
            num_columns = len(columns)
            if num_columns > max_columns:
                max_columns = num_columns
                reference_schema = schema_name
        
        return reference_schema
    
    # def _create_column_mappings(self, reference_schema: str, 
    #                            schema_columns: Dict[str, Dict],
    #                            similarity_matrices: Dict) -> List[Dict]:
    #     """
    #     Create column mappings starting from reference schema.
    #     """
    #     column_groups = []
    #     schemas = list(schema_columns.keys())
    #     ref_columns = schema_columns[reference_schema]
        
    #     # For each column in reference schema
    #     for ref_col_name, ref_col_info in ref_columns.items():
    #         group = {
    #             'reference_schema': reference_schema,
    #             'reference_column': ref_col_name,
    #             'reference_info': ref_col_info,
    #             'matches': {}
    #         }
            
    #         # Find best match in each other schema
    #         for other_schema in schemas:
    #             if other_schema == reference_schema:
    #                 continue
                
    #             # Get similarity matrix
    #             sim_key = (reference_schema, other_schema)
    #             if sim_key not in similarity_matrices:
    #                 continue
                
    #             sim_matrix = similarity_matrices[sim_key]
    #             other_columns = list(schema_columns[other_schema].keys())
    #             ref_columns_list = list(ref_columns.keys())
                
    #             # Find reference column index
    #             if ref_col_name not in ref_columns_list:
    #                 continue
                
    #             ref_idx = ref_columns_list.index(ref_col_name)
                
    #             # Find best match in other schema
    #             similarities = sim_matrix[ref_idx]
    #             if len(similarities) == 0:
    #                 continue
                
    #             best_match_idx = np.argmax(similarities)
    #             best_similarity = similarities[best_match_idx]
                
    #             # Only keep if similarity meets threshold
    #             if best_similarity >= 0.4:
    #                 best_match_col = other_columns[best_match_idx]
    #                 group['matches'][other_schema] = {
    #                     'column': best_match_col,
    #                     'similarity': float(best_similarity),
    #                     'info': schema_columns[other_schema][best_match_col]
    #                 }
            
    #         # Only keep groups with at least one match
    #         if group['matches']:
    #             column_groups.append(group)
        
    #     return column_groups
    
    def _create_column_mappings(self, reference_schema: str, 
                           schema_columns: Dict[str, Dict],
                           similarity_matrices: Dict) -> List[Dict]:
        """
        Create column mappings starting from reference schema.
        Ensures one-to-one mapping: each source column appears in only one mapping.
        """
        column_groups = []
        schemas = list(schema_columns.keys())
        ref_columns = schema_columns[reference_schema]
        
        # Track which columns have been mapped in each schema
        # to prevent the same column from appearing in multiple mappings
        mapped_columns = {schema: {} for schema in schemas}  # schema -> {column: (group_idx, similarity)}
        
        # First pass: create initial column groups from reference
        for ref_col_name, ref_col_info in ref_columns.items():
            group = {
                'reference_schema': reference_schema,
                'reference_column': ref_col_name,
                'reference_info': ref_col_info,
                'matches': {}
            }
            
            # Find best match in each other schema
            for other_schema in schemas:
                if other_schema == reference_schema:
                    continue
                
                # Get similarity matrix
                sim_key = (reference_schema, other_schema)
                if sim_key not in similarity_matrices:
                    continue
                
                sim_matrix = similarity_matrices[sim_key]
                other_columns = list(schema_columns[other_schema].keys())
                ref_columns_list = list(ref_columns.keys())
                
                # Find reference column index
                if ref_col_name not in ref_columns_list:
                    continue
                
                ref_idx = ref_columns_list.index(ref_col_name)
                
                # Find best match in other schema
                similarities = sim_matrix[ref_idx]
                if len(similarities) == 0:
                    continue
                
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                # Only keep if similarity meets threshold
                if best_similarity >= 0.4:
                    best_match_col = other_columns[best_match_idx]
                    
                    # Check if this column is already mapped in another group
                    if best_match_col in mapped_columns[other_schema]:
                        # Compare with existing mapping
                        existing_group_idx, existing_similarity = mapped_columns[other_schema][best_match_col]
                        
                        # Keep only the mapping with highest similarity
                        if best_similarity > existing_similarity:
                            # Remove from old group
                            old_group = column_groups[existing_group_idx]
                            if other_schema in old_group['matches']:
                                del old_group['matches'][other_schema]
                            
                            # Add to new group
                            group['matches'][other_schema] = {
                                'column': best_match_col,
                                'similarity': float(best_similarity),
                                'info': schema_columns[other_schema][best_match_col]
                            }
                            
                            # Update mapping tracker
                            mapped_columns[other_schema][best_match_col] = (len(column_groups), best_similarity)
                    else:
                        # Column not mapped yet, add to current group
                        group['matches'][other_schema] = {
                            'column': best_match_col,
                            'similarity': float(best_similarity),
                            'info': schema_columns[other_schema][best_match_col]
                        }
                        
                        # Track this mapping
                        mapped_columns[other_schema][best_match_col] = (len(column_groups), best_similarity)
            
            # Only keep groups with at least one match
            if group['matches']:
                column_groups.append(group)
        
        return column_groups
    def _create_golden_schema(self, column_groups: List[Dict], 
                            schema_columns: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create golden schema from column groups.
        """
        golden_columns = []
        column_details = {}
        
        for group_idx, group in enumerate(column_groups, 1):
            ref_col = group['reference_column']
            ref_info = group['reference_info']
            
            # Collect all columns in this group
            all_columns = [(group['reference_schema'], ref_col)]
            all_columns.extend([(schema, match['column']) 
                              for schema, match in group['matches'].items()])
            
            # Choose golden name
            column_names = [col[1] for col in all_columns]
            golden_name = self._choose_golden_name(column_names, ref_col)
            
            # Collect types
            all_types = []
            for schema_name, col_name in all_columns:
                if col_name in schema_columns.get(schema_name, {}):
                    col_type = schema_columns[schema_name][col_name]['type']
                    all_types.append(col_type)
            
            # Determine consensus type
            type_counts = Counter(all_types)
            consensus_type = type_counts.most_common(1)[0][0] if type_counts else 'unknown'
            
            # Calculate confidence (average similarity of matches)
            similarities = [match['similarity'] for match in group['matches'].values()]
            confidence = np.mean(similarities) if similarities else 0.5
            
            # Create golden column entry
            golden_col = {
                'name': golden_name,
                'original_reference': ref_col,
                'consensus_type': consensus_type,
                'confidence': float(confidence),
                'mapped_schemas': len(all_columns),
                'source_mappings': [
                    {
                        'schema': schema_name,
                        'column': col_name,
                        'type': schema_columns[schema_name][col_name]['type']
                        if schema_name in schema_columns and col_name in schema_columns[schema_name]
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
            'column_details': column_details
        }
    
    def _choose_golden_name(self, column_names: List[str], reference_name: str) -> str:
        """
        Choose golden column name from list of column names.
        Simple heuristic: use most common, fall back to reference.
        """
        if not column_names:
            return "unknown_column"
        
        # Count frequencies
        name_counts = Counter(column_names)
        
        # If one name dominates, use it
        most_common_name, most_common_count = name_counts.most_common(1)[0]
        if most_common_count / len(column_names) >= 0.5:
            return most_common_name
        
        # Otherwise, prefer the reference name if it exists in the list
        if reference_name in column_names:
            return reference_name
        
        # Fallback: choose shortest meaningful name
        meaningful_names = [name for name in column_names if len(name) > 2]
        if meaningful_names:
            return min(meaningful_names, key=len)
        
        return column_names[0]
    
    def _create_mapping_matrix(self, golden_schema: Dict[str, Any], 
                              schema_columns: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create mapping matrix showing column correspondences.
        """
        schemas = list(schema_columns.keys())
        
        mapping_matrix = {
            'schemas': schemas,
            'golden_columns': golden_schema['columns'],
            'mappings': []
        }
        
        for golden_col_name in golden_schema['columns']:
            golden_col_details = golden_schema['column_details'][golden_col_name]
            source_mappings = golden_col_details['source_mappings']
            
            mapping = {
                'golden_column': golden_col_name,
                'original_reference': golden_col_details['original_reference'],
                'consensus_type': golden_col_details['consensus_type'],
                'confidence': golden_col_details['confidence'],
                'schema_mappings': {}
            }
            
            for schema_name in schemas:
                # Find if this schema has a mapping for this golden column
                schema_mapping = next(
                    (m for m in source_mappings if m['schema'] == schema_name), 
                    None
                )
                
                if schema_mapping:
                    mapping['schema_mappings'][schema_name] = {
                        'column': schema_mapping['column'],
                        'type': schema_mapping['type'],
                        'mapped': True,
                        'is_reference': schema_mapping['column'] == golden_col_details['original_reference']
                    }
                else:
                    mapping['schema_mappings'][schema_name] = {
                        'column': None,
                        'type': None,
                        'mapped': False,
                        'is_reference': False
                    }
            
            mapping_matrix['mappings'].append(mapping)
        
        return mapping_matrix
    
    def _calculate_quality_metrics(self, column_groups: List[Dict], 
                                 schema_columns: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate quality metrics for the mapping.
        """
        if not column_groups:
            return {}
        
        # Calculate similarity statistics
        all_similarities = []
        for group in column_groups:
            for match in group['matches'].values():
                all_similarities.append(match['similarity'])
        
        # Calculate coverage
        total_columns = sum(len(cols) for cols in schema_columns.values())
        mapped_columns = sum(len(group['matches']) + 1 for group in column_groups)  # +1 for reference
        
        return {
            'average_similarity': float(np.mean(all_similarities)) if all_similarities else 0,
            'median_similarity': float(np.median(all_similarities)) if all_similarities else 0,
            'min_similarity': float(min(all_similarities)) if all_similarities else 0,
            'max_similarity': float(max(all_similarities)) if all_similarities else 0,
            'mapped_columns': mapped_columns,
            'total_columns': total_columns,
            'coverage': mapped_columns / total_columns if total_columns > 0 else 0
        }
    
    def _handle_single_schema(self, cluster_files: List[str], cluster_idx: int) -> Dict[str, Any]:
        """
        Handle case where cluster has only one schema.
        """
        if not cluster_files:
            return {}
        
        filename = cluster_files[0]
        schema_info = self._extract_schema_info(filename)
        
        # For single schema, golden schema is just the schema itself
        golden_columns = list(schema_info.keys())
        column_details = {}
        
        for col_name, col_info in schema_info.items():
            column_details[col_name] = {
                'name': col_name,
                'original_reference': col_name,
                'consensus_type': col_info['type'],
                'confidence': 1.0,
                'mapped_schemas': 1,
                'source_mappings': [{
                    'schema': filename,
                    'column': col_name,
                    'type': col_info['type']
                }]
            }
        
        golden_schema = {
            'name': f"Golden_Schema_Single_{cluster_idx}",
            'total_columns': len(golden_columns),
            'columns': golden_columns,
            'column_details': column_details
        }
        
        mapping_matrix = {
            'schemas': [filename],
            'golden_columns': golden_columns,
            'mappings': []
        }
        
        for col_name in golden_columns:
            mapping_matrix['mappings'].append({
                'golden_column': col_name,
                'original_reference': col_name,
                'consensus_type': schema_info[col_name]['type'],
                'confidence': 1.0,
                'schema_mappings': {
                    filename: {
                        'column': col_name,
                        'type': schema_info[col_name]['type'],
                        'mapped': True,
                        'is_reference': True
                    }
                }
            })
        
        return {
            'cluster_files': cluster_files,
            'reference_schema': filename,
            'golden_schema': golden_schema,
            'mapping_matrix': mapping_matrix,
            'quality_metrics': {
                'average_similarity': 1.0,
                'coverage': 1.0,
                'mapped_columns': len(golden_columns),
                'total_columns': len(golden_columns)
            }
        }
    
    def print_results(self, results: Dict[str, Any]):
        """
        Print mapping results in readable format.
        """
        print("\n" + "=" * 100)
        print("COLUMN MAPPING RESULTS")
        print("=" * 100)
        
        for cluster_name, result in results.items():
            if 'mapping_matrix' not in result:
                continue
            
            cluster_files = result.get('cluster_files', [])
            golden_schema = result.get('golden_schema', {})
            mapping_matrix = result.get('mapping_matrix', {})
            quality_metrics = result.get('quality_metrics', {})
            
            print(f"\n{'━' * 80}")
            print(f"📦 {cluster_name}")
            print(f"{'━' * 80}")
            
            # Print cluster files
            print(f"\n📁 Files in cluster ({len(cluster_files)}):")
            for file in cluster_files:
                print(f"   • {file}")
            
            # Print reference schema
            if 'reference_schema' in result:
                print(f"\n🎯 Reference schema: {result['reference_schema']}")
            
            # Print quality metrics
            if quality_metrics:
                print(f"\n📈 Quality Metrics:")
                print(f"   • Average similarity: {quality_metrics.get('average_similarity', 0):.3f}")
                print(f"   • Coverage: {quality_metrics.get('coverage', 0):.1%}")
                print(f"   • Mapped columns: {quality_metrics.get('mapped_columns', 0)}/"
                      f"{quality_metrics.get('total_columns', 0)}")
            
            # Print golden schema
            print(f"\n🎯 Golden Schema: {golden_schema.get('name', 'Unknown')}")
            print(f"   Total columns: {len(golden_schema.get('columns', []))}")
            
            # Print mapping table
            print(f"\n🔗 Column Mappings:")
            print(f"{'─' * 100}")
            
            schemas = mapping_matrix.get('schemas', [])
            header = f"{'Golden Column':<20} | {'Type':<10} | {'Conf':<5} | "
            header += " | ".join([f"{schema[:15]:<15}" for schema in schemas])
            print(header)
            print(f"{'─' * 100}")
            
            for mapping in mapping_matrix.get('mappings', []):
                golden_col = mapping.get('golden_column', '')
                consensus_type = mapping.get('consensus_type', '')
                confidence = mapping.get('confidence', 0)
                schema_mappings = mapping.get('schema_mappings', {})
                
                row = f"{golden_col:<20} | {consensus_type[:10]:<10} | {confidence:.2f} | "
                
                for schema in schemas:
                    schema_map = schema_mappings.get(schema, {})
                    if schema_map.get('mapped', False):
                        col_name = schema_map.get('column', '')
                        
                        # Add indicator
                        if schema_map.get('is_reference', False):
                            indicator = "⭐ "
                        elif confidence >= 0.8:
                            indicator = "🟢 "
                        elif confidence >= 0.6:
                            indicator = "🟡 "
                        else:
                            indicator = "🔴 "
                        
                        row += f"{indicator}{col_name[:12]:<12} | "
                    else:
                        row += f"{'─':<12} | "
                
                print(row)
            
            print(f"{'━' * 80}")
    
    def print_unmapped_analysis(self, results: Dict[str, Any]):
        """
        Print analysis of unmapped columns.
        """
        print("\n" + "=" * 100)
        print("UNMAPPED COLUMNS ANALYSIS")
        print("=" * 100)
        
        total_unmapped = 0
        
        for cluster_name, result in results.items():
            cluster_files = result.get('cluster_files', [])
            mapping_matrix = result.get('mapping_matrix', {})
            
            # Get all mapped columns in this cluster
            mapped_columns = defaultdict(set)
            for mapping in mapping_matrix.get('mappings', []):
                schema_mappings = mapping.get('schema_mappings', {})
                for schema_name, schema_map in schema_mappings.items():
                    if schema_map.get('mapped', False):
                        mapped_columns[schema_name].add(schema_map['column'])
            
            # Find unmapped columns for each schema in cluster
            cluster_unmapped = 0
            
            for schema_name in cluster_files:
                if schema_name not in self.enhanced_schemas:
                    continue
                
                all_columns = set(self.enhanced_schemas[schema_name].get('columns', []))
                mapped_in_schema = mapped_columns.get(schema_name, set())
                unmapped_in_schema = all_columns - mapped_in_schema
                
                if unmapped_in_schema:
                    cluster_unmapped += len(unmapped_in_schema)
                    print(f"\n📁 {schema_name} ({cluster_name}): {len(unmapped_in_schema)} unmapped")
                    
                    # Show unmapped columns
                    for col in sorted(unmapped_in_schema):
                        col_type = self.enhanced_schemas[schema_name].get('column_types', {}).get(col, 'unknown')
                        print(f"   • {col} ({col_type})")
            
            total_unmapped += cluster_unmapped
        
        print(f"\n{'=' * 80}")
        print(f"📊 Total unmapped columns across all clusters: {total_unmapped}")
        print(f"{'=' * 80}")


def save_results(results: Dict[str, Any], filename: str = 'simplified_mappings.json'):
    """
    Save mapping results to JSON file.
    """
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
    
    serializable_results = convert_for_json(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")


def run_simplified_mapping():#all_features
    """
    Main function to run simplified column mapping.
    """
    print("Starting Simplified Column Mapping...")
    
    try:
        # Load schema data
        print("Loading schema_data.json...")
        with open('schema_data.json', 'r') as f:
            enhanced_schemas = json.load(f)
        print(f"✅ Loaded {len(enhanced_schemas)} schemas")
        # enhanced_schemas = features
        
        # Load clustering results
        print("Loading final_clustering_results.json...")
        with open('final_clustering_results.json', 'r') as f:
            clustering_results = json.load(f)
        
        # Get clusters
        clusters = clustering_results.get('clusters', [])
        print(f"✅ Loaded {len(clusters)} clusters")
        
        # Initialize mapper
        mapper = SimplifiedColumnMapper(enhanced_schemas)
        
        # Run mapping
        results = mapper.map_all_clusters(clusters)
        
        # Print results
        mapper.print_results(results)
        
        # Print unmapped analysis
        mapper.print_unmapped_analysis(results)
        
        # Save results
        save_results(results, 'simplified_column_mappings.json')
        
        # Summary statistics
        print(f"\n📊 FINAL SUMMARY")
        print(f"{'=' * 80}")
        
        total_clusters = len(results)
        total_golden_columns = 0
        total_high_conf = 0
        total_medium_conf = 0
        total_low_conf = 0
        
        for cluster_name, result in results.items():
            golden_schema = result.get('golden_schema', {})
            total_golden_columns += len(golden_schema.get('columns', []))
            
            for mapping in result.get('mapping_matrix', {}).get('mappings', []):
                confidence = mapping.get('confidence', 0)
                if confidence >= 0.8:
                    total_high_conf += 1
                elif confidence >= 0.6:
                    total_medium_conf += 1
                else:
                    total_low_conf += 1
        
        total_mappings = total_high_conf + total_medium_conf + total_low_conf
        
        print(f"• Clusters processed: {total_clusters}")
        print(f"• Golden columns created: {total_golden_columns}")
        print(f"• Total mappings: {total_mappings}")
        
        if total_mappings > 0:
            print(f"• High confidence (≥0.8): {total_high_conf} ({total_high_conf/total_mappings:.1%})")
            print(f"• Medium confidence (0.6-0.8): {total_medium_conf} ({total_medium_conf/total_mappings:.1%})")
            print(f"• Low confidence (<0.6): {total_low_conf} ({total_low_conf/total_mappings:.1%})")
        
        print(f"{'=' * 80}")
        print("✅ Simplified mapping complete!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nRequired files:")
        print("1. schema_data.json (from Phase 1)")
        print("2. final_clustering_results.json (from Phase 3)")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# Main execution
if __name__ == "__main__":
    run_simplified_mapping()