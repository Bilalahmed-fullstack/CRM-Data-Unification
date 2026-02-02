import json
import re
import numpy as np
from collections import defaultdict

class PatternBasedSchemaComparator:
    def __init__(self, schema_data):
        """
        Initialize with schema data containing pattern information.
        
        Args:
            schema_data: Dictionary from enhanced_schemas or loaded from file
        """
        self.schema_data = schema_data
        self.files = list(schema_data.keys())
    
    def load_from_json(self, filepath):
        """Load schema data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.schema_data = json.load(f)
        self.files = list(self.schema_data.keys())
        return self.schema_data
    
    def extract_pattern_metrics(self, file):
        """
        Extract pattern metrics from a file's schema data.
        Returns dictionary with ONLY pattern-based metrics.
        """
        if file not in self.schema_data:
            return None
        
        schema = self.schema_data[file]
        patterns = schema.get('value_patterns', {})
        
        if not patterns:
            return None
        
        # Collect pattern metrics from ALL columns
        pattern_metrics = {
            'file': file,
            'columns': [],
            'column_patterns': {}
        }
        
        for col, col_patterns in patterns.items():
            pattern_metrics['columns'].append(col)
            
            # Extract ONLY pattern metrics (no semantic interpretation)
            col_metrics = {}
            
            # 1. Basic pattern statistics
            sep_patterns = col_patterns.get('separator_patterns', {})
            if sep_patterns:
                col_metrics.update({
                    'pattern_diversity': sep_patterns.get('pattern_diversity', 1.0),
                    'unique_patterns': sep_patterns.get('unique_patterns', 0),
                    'total_values': sep_patterns.get('total_values', 0),
                    'pattern_consistency': 1 - sep_patterns.get('pattern_diversity', 1.0)
                })
                
                # 2. Dominant pattern groups analysis
                dominant_groups = sep_patterns.get('dominant_pattern_groups', [])
                if dominant_groups:
                    # Calculate coverage by dominant patterns
                    coverage = sum(group.get('frequency', 0) for group in dominant_groups)
                    col_metrics['dominant_coverage'] = coverage
                    
                    # Pattern length from example patterns
                    pattern_lengths = [len(group.get('example_pattern', '')) for group in dominant_groups]
                    if pattern_lengths:
                        col_metrics['avg_pattern_length'] = np.mean(pattern_lengths)
                        col_metrics['pattern_length_std'] = np.std(pattern_lengths) if len(pattern_lengths) > 1 else 0
                    
                    # Group frequency statistics
                    frequencies = [group.get('frequency', 0) for group in dominant_groups]
                    col_metrics['max_group_frequency'] = max(frequencies) if frequencies else 0
                    col_metrics['min_group_frequency'] = min(frequencies) if frequencies else 0
            
            # 3. Character distribution metrics
            char_dist = col_patterns.get('char_distribution', {})
            if char_dist:
                col_metrics['char_alpha_ratio'] = char_dist.get('alpha', 0)
                col_metrics['char_digit_ratio'] = char_dist.get('digit', 0)
                col_metrics['char_special_ratio'] = char_dist.get('special', 0)
                col_metrics['char_space_ratio'] = char_dist.get('space', 0)
            
            # 4. Other pattern metrics
            col_metrics['digit_ratio'] = col_patterns.get('digit_ratio', 0)
            col_metrics['unique_ratio'] = col_patterns.get('unique_ratio', 1.0)
            
            # 5. Length statistics
            length_stats = col_patterns.get('length_stats', {})
            if length_stats:
                col_metrics.update({
                    f'length_{k}': v for k, v in length_stats.items()
                })
            
            pattern_metrics['column_patterns'][col] = col_metrics
        
        return pattern_metrics
    
    def compare_column_patterns(self, col_patterns1, col_patterns2):
        """
        Compare pattern metrics between two columns.
        Returns similarity score based ONLY on pattern metrics.
        """
        if not col_patterns1 or not col_patterns2:
            return 0
        
        similarity_scores = []
        weights = []
        
        # Compare each available metric
        metrics_to_compare = [
            ('pattern_diversity', 0.2),
            ('char_alpha_ratio', 0.15),
            ('char_digit_ratio', 0.15),
            ('digit_ratio', 0.1),
            ('unique_ratio', 0.1),
            ('dominant_coverage', 0.1),
            ('length_mean', 0.1),
            ('length_std', 0.05),
            ('avg_pattern_length', 0.05)
        ]
        
        for metric, weight in metrics_to_compare:
            if metric in col_patterns1 and metric in col_patterns2:
                val1 = col_patterns1[metric]
                val2 = col_patterns2[metric]
                
                # Calculate similarity for this metric
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if metric in ['pattern_diversity', 'char_alpha_ratio', 'char_digit_ratio', 
                                 'digit_ratio', 'unique_ratio', 'dominant_coverage']:
                        # For ratios (0-1), use 1 - absolute difference
                        similarity = 1 - abs(val1 - val2)
                    elif metric in ['length_mean', 'length_std', 'avg_pattern_length']:
                        # For lengths, use normalized similarity
                        max_val = max(abs(val1), abs(val2), 1)
                        similarity = 1 - abs(val1 - val2) / max_val
                    else:
                        # Default similarity
                        similarity = 1 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1)
                    
                    similarity_scores.append(similarity * weight)
                    weights.append(weight)
        
        # Also compare pattern consistency indicators
        if ('pattern_diversity' in col_patterns1 and 'pattern_diversity' in col_patterns2):
            # Both highly consistent or both variable?
            div1 = col_patterns1['pattern_diversity']
            div2 = col_patterns2['pattern_diversity']
            
            # Classify consistency level
            consistency1 = 'high' if div1 < 0.1 else 'medium' if div1 < 0.3 else 'low'
            consistency2 = 'high' if div2 < 0.1 else 'medium' if div2 < 0.3 else 'low'
            
            consistency_sim = 1.0 if consistency1 == consistency2 else 0.5 if abs(div1 - div2) < 0.2 else 0.2
            similarity_scores.append(consistency_sim * 0.1)
            weights.append(0.1)
        
        # Calculate weighted average
        if weights:
            total_weight = sum(weights)
            if total_weight > 0:
                return sum(similarity_scores) / total_weight
        
        return 0
    
    def find_best_column_matches(self, pattern_metrics1, pattern_metrics2):
        """
        Find best column matches between two files based ONLY on pattern metrics.
        """
        if not pattern_metrics1 or not pattern_metrics2:
            return []
        
        cols1 = pattern_metrics1.get('columns', [])
        cols2 = pattern_metrics2.get('columns', [])
        patterns1 = pattern_metrics1.get('column_patterns', {})
        patterns2 = pattern_metrics2.get('column_patterns', {})
        
        matches = []
        used_cols2 = set()
        
        # Try to match each column in file1 to best match in file2
        for col1 in cols1:
            if col1 not in patterns1:
                continue
            
            best_match = None
            best_score = 0
            
            for col2 in cols2:
                if col2 in used_cols2 or col2 not in patterns2:
                    continue
                
                # Compare pattern metrics
                score = self.compare_column_patterns(patterns1[col1], patterns2[col2])
                
                if score > best_score:
                    best_score = score
                    best_match = col2
            
            if best_match and best_score > 0.4:  # Pattern similarity threshold
                matches.append({
                    'from_column': col1,
                    'to_column': best_match,
                    'pattern_similarity': best_score,
                    'from_pattern_diversity': patterns1[col1].get('pattern_diversity', 1.0),
                    'to_pattern_diversity': patterns2[best_match].get('pattern_diversity', 1.0)
                })
                used_cols2.add(best_match)
        
        return matches
    
    def calculate_file_similarity(self, file1, file2):
        """
        Calculate overall similarity between two files based ONLY on pattern metrics.
        """
        pattern_metrics1 = self.extract_pattern_metrics(file1)
        pattern_metrics2 = self.extract_pattern_metrics(file2)
        
        if not pattern_metrics1 or not pattern_metrics2:
            return 0
        
        # 1. Find column matches based on patterns
        matches = self.find_best_column_matches(pattern_metrics1, pattern_metrics2)
        
        if not matches:
            return 0
        
        # 2. Average pattern similarity of matches
        avg_match_similarity = np.mean([m['pattern_similarity'] for m in matches])
        
        # 3. Column count similarity (normalized)
        cols1 = pattern_metrics1.get('columns', [])
        cols2 = pattern_metrics2.get('columns', [])
        count_similarity = 1 - abs(len(cols1) - len(cols2)) / max(len(cols1), len(cols2), 1)
        
        # 4. Overall pattern consistency similarity
        # Calculate average pattern diversity for each file
        diversities1 = [patterns.get('pattern_diversity', 1.0) 
                       for patterns in pattern_metrics1['column_patterns'].values()]
        diversities2 = [patterns.get('pattern_diversity', 1.0)
                       for patterns in pattern_metrics2['column_patterns'].values()]
        
        avg_div1 = np.mean(diversities1) if diversities1 else 1.0
        avg_div2 = np.mean(diversities2) if diversities2 else 1.0
        diversity_similarity = 1 - abs(avg_div1 - avg_div2)
        
        # Combined similarity (weighted)
        file_similarity = (
            0.5 * avg_match_similarity + 
            0.3 * count_similarity + 
            0.2 * diversity_similarity
        )
        
        return min(file_similarity, 1.0)
    
    def create_similarity_matrix(self):
        """
        Create pattern-based similarity matrix for all file pairs.
        """
        n = len(self.files)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self.calculate_file_similarity(self.files[i], self.files[j])
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
        
        return similarity_matrix
    
    def analyze_pattern_clusters(self, similarity_matrix, threshold=0.9):
        """
        Cluster files based on pattern similarity matrix.
        Returns groups of files with similar pattern characteristics.
        """
        # Simple clustering: group files with similarity above threshold
        n = len(self.files)
        clusters = []
        assigned = set()
        
        for i in range(n):
            if self.files[i] in assigned:
                continue
            
            # Start new cluster with this file
            cluster = [self.files[i]]
            assigned.add(self.files[i])
            
            # Find similar files
            for j in range(n):
                if i != j and self.files[j] not in assigned:
                    if similarity_matrix[i, j] >= threshold:
                        cluster.append(self.files[j])
                        assigned.add(self.files[j])
            
            clusters.append(cluster)
        
        return clusters
    
    def generate_pattern_summary(self, cluster):
        """
        Generate pattern-based summary for a cluster of files.
        Uses ONLY pattern metrics from the data.
        """
        if not cluster:
            return {}
        
        # Collect pattern statistics from all files in cluster
        all_pattern_diversities = []
        all_digit_ratios = []
        all_char_alpha_ratios = []
        
        for file in cluster:
            pattern_metrics = self.extract_pattern_metrics(file)
            if not pattern_metrics:
                continue
            
            for col_metrics in pattern_metrics['column_patterns'].values():
                all_pattern_diversities.append(col_metrics.get('pattern_diversity', 1.0))
                all_digit_ratios.append(col_metrics.get('digit_ratio', 0))
                all_char_alpha_ratios.append(col_metrics.get('char_alpha_ratio', 0))
        
        if not all_pattern_diversities:
            return {}
        
        # Calculate cluster pattern signature
        signature = {
            'avg_pattern_diversity': np.mean(all_pattern_diversities),
            'std_pattern_diversity': np.std(all_pattern_diversities) if len(all_pattern_diversities) > 1 else 0,
            'avg_digit_ratio': np.mean(all_digit_ratios),
            'avg_char_alpha_ratio': np.mean(all_char_alpha_ratios),
            'file_count': len(cluster),
            'total_columns': sum(len(self.extract_pattern_metrics(f).get('columns', [])) for f in cluster)
        }
        
        # Describe pattern characteristics (not entity types)
        characteristics = []
        
        # Pattern consistency
        if signature['avg_pattern_diversity'] < 0.1:
            characteristics.append("high_pattern_consistency")
        elif signature['avg_pattern_diversity'] < 0.3:
            characteristics.append("medium_pattern_consistency")
        else:
            characteristics.append("low_pattern_consistency")
        
        # Content type based on character distribution
        if signature['avg_digit_ratio'] > 0.7:
            characteristics.append("digit_dominant")
        elif signature['avg_digit_ratio'] > 0.3:
            characteristics.append("mixed_digit_text")
        else:
            characteristics.append("text_dominant")
        
        # Alphabetical content
        if signature['avg_char_alpha_ratio'] > 0.7:
            characteristics.append("alphabetic")
        
        signature['pattern_characteristics'] = characteristics
        
        return signature
    
    def run_pattern_based_analysis(self, similarity_threshold=0.5):
        """
        Complete pattern-based analysis using ONLY data from schema_details.
        """
        print("=" * 80)
        print("PATTERN-BASED SCHEMA ANALYSIS")
        print("=" * 80)
        print("Using ONLY pattern metrics from enhanced schema data\n")
        
        # Step 1: Calculate pattern similarities
        print("1. Calculating pattern-based similarities...")
        similarity_matrix = self.create_similarity_matrix()
        
        # Step 2: Cluster based on patterns
        print("2. Clustering files by pattern similarity...")
        clusters = self.analyze_pattern_clusters(similarity_matrix, similarity_threshold)
        
        # Step 3: Analyze each cluster's pattern characteristics
        print("3. Analyzing pattern characteristics...")
        results = {
            'clusters': [],
            'pattern_signatures': {},
            'similarity_matrix': similarity_matrix.tolist() if similarity_matrix is not None else []
        }
        
        for i, cluster in enumerate(clusters):
            cluster_id = i + 1
            pattern_summary = self.generate_pattern_summary(cluster)
            
            # Extract column pattern matches for the cluster
            if len(cluster) > 1:
                # Use first file as reference for pattern matching
                ref_file = cluster[0]
                pattern_matches = {}
                
                for other_file in cluster[1:]:
                    pattern_metrics1 = self.extract_pattern_metrics(ref_file)
                    pattern_metrics2 = self.extract_pattern_metrics(other_file)
                    
                    if pattern_metrics1 and pattern_metrics2:
                        matches = self.find_best_column_matches(pattern_metrics1, pattern_metrics2)
                        if matches:
                            pattern_matches[other_file] = matches
            
            cluster_info = {
                'cluster_id': cluster_id,
                'files': cluster,
                'pattern_summary': pattern_summary,
                'pattern_matches': pattern_matches if 'pattern_matches' in locals() else {}
            }
            
            results['clusters'].append(cluster_info)
            results['pattern_signatures'][f'cluster_{cluster_id}'] = pattern_summary
        
        # Step 4: Generate report
        print("4. Generating pattern-based report...")
        self.print_pattern_analysis_report(results)
        
        return results
    
    def print_pattern_analysis_report(self, results):
        """Print pattern-based analysis report."""
        print("\n" + "=" * 80)
        print("PATTERN ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Total pattern-based clusters: {len(results['clusters'])}")
        
        for cluster_info in results['clusters']:
            cluster_id = cluster_info['cluster_id']
            files = cluster_info['files']
            pattern_summary = cluster_info['pattern_summary']
            
            print(f"\n{'='*60}")
            print(f"PATTERN CLUSTER {cluster_id}")
            print(f"{'='*60}")
            print(f"Files ({len(files)}): {files}")
            
            if pattern_summary:
                print(f"\n📈 Pattern Signature:")
                print(f"  • Average Pattern Diversity: {pattern_summary.get('avg_pattern_diversity', 0):.3f}")
                print(f"  • Average Digit Ratio: {pattern_summary.get('avg_digit_ratio', 0):.3f}")
                print(f"  • Average Alpha Ratio: {pattern_summary.get('avg_char_alpha_ratio', 0):.3f}")
                
                characteristics = pattern_summary.get('pattern_characteristics', [])
                if characteristics:
                    print(f"  • Pattern Characteristics: {', '.join(characteristics)}")
            
            # Show pattern-based column matches
            pattern_matches = cluster_info.get('pattern_matches', {})
            if pattern_matches:
                print(f"\n🔗 Pattern-based Column Matches (to {files[0]}):")
                for other_file, matches in pattern_matches.items():
                    if matches:
                        print(f"  {other_file}:")
                        for match in matches[:3]:  # Show top 3 matches
                            print(f"    {match['from_column']} ↔ {match['to_column']} "
                                  f"(similarity: {match['pattern_similarity']:.2f})")
                        if len(matches) > 3:
                            print(f"    ... and {len(matches) - 3} more matches")
        
        # Overall statistics
        print(f"\n{'='*60}")
        print("OVERALL PATTERN STATISTICS")
        print(f"{'='*60}")
        
        total_files = sum(len(cluster['files']) for cluster in results['clusters'])
        print(f"Total files analyzed: {total_files}")
        
        # Calculate average pattern diversity across all files
        all_diversities = []
        for cluster_info in results['clusters']:
            summary = cluster_info.get('pattern_summary', {})
            if 'avg_pattern_diversity' in summary:
                all_diversities.append(summary['avg_pattern_diversity'])
        
        if all_diversities:
            print(f"Overall average pattern diversity: {np.mean(all_diversities):.3f}")
            print(f"Pattern diversity range: [{min(all_diversities):.3f}, {max(all_diversities):.3f}]")
        
        print("\n" + "=" * 80)


































import re
import json
import ast
from typing import Dict, Any

class TextSchemaParser:
    def __init__(self):
        self.schema_data = {}
    
    def parse_text_file(self, filepath: str) -> Dict[str, Any]:
        """
        Parse the pretty-printed text file back into structured data.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by file sections
        file_sections = re.split(r'\n={80}\nFILE:', content)
        
        for section in file_sections:
            if not section.strip():
                continue
            
            # Extract filename
            if section.startswith('='):
                # First section
                lines = section.split('\n', 2)
                if len(lines) >= 3:
                    filename_line = lines[1]
                    data_start = lines[2] if len(lines) > 2 else ""
                else:
                    continue
            else:
                # Subsequent sections
                lines = section.split('\n', 1)
                if len(lines) >= 2:
                    filename_line = lines[0]
                    data_start = lines[1]
                else:
                    continue
            
            # Extract filename
            filename = filename_line.strip()
            
            # Find the data section
            if "COMPLETE SCHEMA DATA:" in data_start:
                data_text = data_start.split("COMPLETE SCHEMA DATA:", 1)[1]
            else:
                data_text = data_start
            
            # Parse the pretty-printed dictionary
            schema = self._parse_pprint_dict(data_text)
            if schema:
                self.schema_data[filename] = schema
        
        return self.schema_data
    
    def _parse_pprint_dict(self, text: str) -> Dict[str, Any]:
        """
        Parse pretty-printed dictionary from text.
        """
        try:
            # Try to parse as Python literal using ast
            return ast.literal_eval(text.strip())
        except:
            # Fallback: manual parsing
            return self._manual_parse_dict(text)
    
    def _manual_parse_dict(self, text: str) -> Dict[str, Any]:
        """
        Manual parsing of pretty-printed dictionary.
        """
        result = {}
        lines = text.strip().split('\n')
        current_key = None
        current_value = []
        indent_level = 0
        
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            
            # Count leading spaces for indentation
            leading_spaces = len(line) - len(line.lstrip())
            
            # Check for key-value pair
            if ': ' in line and leading_spaces == 2:  # Top-level keys
                # Save previous key-value pair
                if current_key is not None:
                    result[current_key] = self._parse_value('\n'.join(current_value))
                
                # Start new key-value pair
                key, value_start = line.split(': ', 1)
                current_key = key.strip("'\"")
                current_value = [value_start]
                indent_level = leading_spaces
            elif current_key is not None and leading_spaces > indent_level:
                # Continuation of current value
                current_value.append(line[indent_level:])
            else:
                # Start of new top-level item
                if current_key is not None:
                    result[current_key] = self._parse_value('\n'.join(current_value))
                    current_key = None
                    current_value = []
        
        # Add last key-value pair
        if current_key is not None:
            result[current_key] = self._parse_value('\n'.join(current_value))
        
        return result
    
    def _parse_value(self, text: str) -> Any:
        """
        Parse a value from text.
        """
        text = text.strip()
        
        # Try to parse as Python literal
        try:
            return ast.literal_eval(text)
        except:
            # Check for lists
            if text.startswith('[') and text.endswith(']'):
                items = text[1:-1].split(',')
                return [item.strip().strip("'\"") for item in items if item.strip()]
            
            # Check for dictionaries
            if text.startswith('{') and text.endswith('}'):
                # Simple dictionary parsing
                pairs = text[1:-1].split(',')
                result = {}
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        result[key.strip().strip("'\"")] = value.strip().strip("'\"")
                return result
            
            # Return as string
            return text

# Simplified version: Save and load as JSON instead
def save_as_json(enhanced_schemas: Dict[str, Any], filename: str = 'schema_data.json'):
    """
    Save enhanced schemas as JSON for easier parsing.
    """
    # Convert numpy types to Python native types
    import numpy as np
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_data = convert_to_serializable(enhanced_schemas)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 JSON saved to: {filename}")

def load_from_json(filepath: str) -> Dict[str, Any]:
    """Load schema data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)



def main_map(enhanced_schemas):
    # Save as JSON for easier processing
    save_as_json(enhanced_schemas, 'schema_data.json')
    
   
    schema_data = load_from_json('schema_data.json')
    # comparator = PatternBasedSchemaComparator(schema_data)
    # results = comparator.run_pattern_based_analysis()
    # # 4. Save results
    # with open('schema_data_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    return schema_data





# ====================== MAIN EXECUTION ======================

if __name__ == "__main__":





    # Assuming you have enhanced_schemas from your extraction
    from metadata import EnhancedDataExtractor
    extractor = EnhancedDataExtractor(sample_size=20)  # 20 samples for display, but ALL for patterns
    
    # Extract from directory
    directory = '../../Amazon Sales Dataset/raw data'
    enhanced_schemas = extractor.extract_from_directory(directory)
    # Save as JSON for easier processing
    save_as_json(enhanced_schemas, 'schema_data.json')
    
    # Then use with PatternBasedSchemaComparator
    schema_data = load_from_json('schema_data.json')
    comparator = PatternBasedSchemaComparator(schema_data)
    results = comparator.run_pattern_based_analysis()
    # 4. Save results
    with open('schema_data_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    
    print("Recommended workflow:")
    print("1. Save your enhanced_schemas as JSON using save_as_json()")
    print("2. Load it with load_from_json()")
    print("3. Use with PatternBasedSchemaComparator")



