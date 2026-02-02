import os
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

class EnhancedDataExtractor:
    def __init__(self, sample_size: int = 100):
        """
        Initialize the enhanced data extractor.
        
        Args:
            sample_size: Number of rows to sample for analysis
        """
        self.sample_size = sample_size
        self.schema_dict = {}
    
    def infer_data_type(self, values: List) -> str:
        """
        Infer data type from a list of values.
        Dynamic type inference without pre-assumed patterns.
        """
        if not values:
            return 'unknown'
        
        # Convert to string and strip
        str_values = [str(v).strip() if v is not None else '' for v in values]
        
        # Check for boolean patterns
        bool_patterns = {'true', 'false', 'yes', 'no', '0', '1', 't', 'f', 'y', 'n'}
        if all(v.lower() in bool_patterns for v in str_values if v):
            return 'boolean'
        
        # Check for integer
        int_count = 0
        for v in str_values:
            if v:
                try:
                    int(v)
                    int_count += 1
                except:
                    pass
        
        if int_count / max(len([v for v in str_values if v]), 1) > 0.9:
            return 'integer'
        
        # Check for float
        float_count = 0
        for v in str_values:
            if v:
                try:
                    float(v)
                    float_count += 1
                except:
                    pass
        
        if float_count / max(len([v for v in str_values if v]), 1) > 0.9:
            return 'float'
        
        # Check for datetime patterns (dynamic discovery)
        date_patterns = self._discover_date_patterns(str_values)
        if date_patterns:
            return 'datetime'
        
        # Default to string
        return 'string'
    
    def _discover_date_patterns(self, values: List[str]) -> List[str]:
        """
        Dynamically discover date patterns in values.
        No pre-assumed date formats.
        """
        patterns = []
        
        # Common date separators
        separators = ['-', '/', '.', ' ']
        
        for val in values:
            if not val:
                continue
            
            # Check if value has date-like structure
            # Has numbers and separators in a structured way
            for sep in separators:
                if sep in val:
                    parts = val.split(sep)
                    if len(parts) == 3:
                        # Check if parts could be day/month/year
                        if all(p.isdigit() for p in parts):
                            patterns.append(f"digit{sep}digit{sep}digit")
                    
                    # Check for ISO-like format
                    if 'T' in val and ':' in val:
                        patterns.append("iso_timestamp")
        
        return list(set(patterns))
    
    def extract_value_patterns(self, values: List) -> Dict[str, Any]:
        """
        Extract patterns from values dynamically.
        No pre-assumed patterns (email, phone, etc.)
        """
        if not values:
            return {}
        
        str_values = [str(v).strip() if v is not None else '' for v in values]
        non_empty = [v for v in str_values if v]
        
        if not non_empty:
            return {}
        
        patterns = {
            'char_distribution': self._analyze_char_distribution(non_empty),
            'length_stats': self._analyze_length_stats(non_empty),
            'separator_patterns': self._discover_separator_patterns(non_empty),
            'digit_ratio': self._calculate_digit_ratio(non_empty),
            'unique_ratio': len(set(non_empty)) / len(non_empty)
        }
        
        return patterns
    
    def _analyze_char_distribution(self, values: List[str]) -> Dict[str, float]:
        """Analyze character type distribution in values."""
        total_chars = sum(len(v) for v in values)
        if total_chars == 0:
            return {}
        
        char_types = {
            'alpha': 0,
            'digit': 0,
            'special': 0,
            'space': 0
        }
        
        for val in values:
            for char in val:
                if char.isalpha():
                    char_types['alpha'] += 1
                elif char.isdigit():
                    char_types['digit'] += 1
                elif char.isspace():
                    char_types['space'] += 1
                else:
                    char_types['special'] += 1
        
        # Convert to ratios
        return {k: v/total_chars for k, v in char_types.items()}
    
    def _analyze_length_stats(self, values: List[str]) -> Dict[str, float]:
        """Calculate length statistics for values."""
        lengths = [len(v) for v in values]
        if not lengths:
            return {}
        
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths)
        }
    
    def _discover_separator_patterns(self, values: List[str]) -> List[str]:
        """Discover common separator patterns in values."""
        separators = ['-', '_', '.', ' ', '@', '+', '(', ')', '/']
        patterns = []
        
        for val in values:
            pattern = []
            for char in val:
                if char in separators:
                    pattern.append(char)
                elif char.isdigit():
                    pattern.append('D')
                elif char.isalpha():
                    pattern.append('L')
                else:
                    pattern.append('S')
            
            # Create pattern string
            pattern_str = ''.join(pattern)
            if pattern_str not in patterns:
                patterns.append(pattern_str)
        
        return patterns[:10]  # Return top 10 patterns
    
    def _calculate_digit_ratio(self, values: List[str]) -> float:
        """Calculate the ratio of digits to total characters."""
        total_chars = sum(len(v) for v in values)
        if total_chars == 0:
            return 0
        
        digit_chars = sum(1 for v in values for char in v if char.isdigit())
        return digit_chars / total_chars
    
    def extract_file_schema(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Extract enhanced schema information from a single file.
        """
        schema_info = {
            'filename': filename,
            'file_path': file_path,
            'columns': [],
            'column_count': 0,
            'value_samples': {},
            'column_types': {},
            'value_patterns': {}
        }
        
        try:
            if filename.endswith('.csv'):
                # Read CSV with limited rows for efficiency
                df = pd.read_csv(file_path, nrows=self.sample_size)
            elif filename.endswith('.json'):
                # Read JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data[:self.sample_size])
                elif isinstance(data, dict):
                    # Handle single object or nested structure
                    # Flatten if needed
                    df = pd.json_normalize(data)
                    df = df.head(self.sample_size)
                else:
                    raise ValueError(f"Unsupported JSON structure in {filename}")
            else:
                return schema_info
            
            # Store basic column info
            schema_info['columns'] = list(df.columns)
            schema_info['column_count'] = len(df.columns)
            
            # Extract enhanced information for each column
            for col in df.columns:
                # Get value samples
                samples = df[col].dropna().tolist() #samples = df[col].dropna().head(20).tolist()
                schema_info['value_samples'][col] = samples
                
                # Infer data type
                col_type = self.infer_data_type(samples)
                schema_info['column_types'][col] = col_type
                
                # Extract value patterns
                patterns = self.extract_value_patterns(samples)
                schema_info['value_patterns'][col] = patterns
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            # Return basic info if full extraction fails
            schema_info['error'] = str(e)
        
        return schema_info
    
    def extract_from_directory(self, directory: str) -> Dict[str, Any]:
        """
        Extract enhanced schema information from all CSV/JSON files in directory.
        """
        all_schemas = {}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.csv', '.json')):
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file}")
                    
                    schema_info = self.extract_file_schema(file_path, file)
                    all_schemas[file] = schema_info
        
        return all_schemas
    
    def print_full_schema_dict(self, schema_dict: Dict[str, Any]):
        """Print the complete schema dictionary with all details."""
        print("\n" + "=" * 80)
        print("COMPLETE SCHEMA DICTIONARY")
        print("=" * 80)
        
        import pprint
        pp = pprint.PrettyPrinter(indent=2, width=100)
        
        for filename, schema in schema_dict.items():
            print(f"\n{'='*60}")
            print(f"FILE: {filename}")
            print(f"{'='*60}")
            
            # Print all schema information
            for key, value in schema.items():
                print(f"\n{key.upper()}:")
                if key in ['value_samples', 'value_patterns']:
                    # Handle nested dictionaries specially
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            print(f"  {subkey}:")
                            if key == 'value_samples':
                                # Print samples
                                print(f"    {subvalue[:5]}")  # Show first 5 samples
                                if len(subvalue) > 5:
                                    print(f"    ... and {len(subvalue) - 5} more")
                            elif key == 'value_patterns':
                                # Print patterns
                                if isinstance(subvalue, dict):
                                    for pattern_key, pattern_val in subvalue.items():
                                        print(f"    {pattern_key}: {pattern_val}")
                                else:
                                    print(f"    {subvalue}")
                elif key == 'column_types':
                    print(f"  {value}")
                elif isinstance(value, list):
                    print(f"  {value}")
                else:
                    print(f"  {value}")
    
# Example usage
if __name__ == "__main__":

    directory = '../../Amazon Sales Dataset/raw data'
    # directory = '../../Amazon-GoogleProducts'
    from metadata import main_meta_data_extractor
    print("Phase: 1 Data Extraction----------------------------------------------")
    enhanced_schemas = main_meta_data_extractor(directory)
    from json_map import main_map
    schema_data = main_map(enhanced_schemas)
    from feature_engineering import feature_engineering
    print("Phase: 2 Feature Engineering----------------------------------------------")
    all_features  = feature_engineering(schema_data)
    from clustering import clustering
    print("Phase: 3 Schema Discovery----------------------------------------------")
    clustering(all_features)
    from column_mapping import run_simplified_mapping
    run_simplified_mapping()
    from unify.analysis import unify
    unify(directory)
    from unify.deduplication import deduplicate
    deduplicate()







    # # Initialize extractor
    # extractor = EnhancedDataExtractor(sample_size=100)
    
    # # Extract from directory
    # directory = '../../Amazon Sales Dataset/raw data'
    # enhanced_schemas = extractor.extract_from_directory(directory)
    
    # # Print summary
    # extractor.print_full_schema_dict(enhanced_schemas)
    
    # # Save to file for later use
    # import pickle
    # with open('enhanced_schemas.pkl', 'wb') as f:
    #     pickle.dump(enhanced_schemas, f)
    
    # print("\n✅ Enhanced schema extraction complete!")
    # print(f"📊 Total files: {len(enhanced_schemas)}")
    # print(f"💾 Saved to: enhanced_schemas.pkl")