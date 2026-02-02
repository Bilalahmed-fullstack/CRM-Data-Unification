import os
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from llm_judge import describe_column

class EnhancedDataExtractor:
    def __init__(self, sample_size: int = None):
        """
        Initialize the enhanced data extractor.
        
        Args:
            sample_size: Number of rows to sample for analysis. None means use all data.
        """
        self.sample_size = sample_size
    
    # def infer_data_type(self, values: List) -> str:
    #     """
    #     Infer data type from a list of values.
    #     Dynamic type inference without pre-assumed patterns.
    #     """
    #     if not values:
    #         return 'unknown'
        
    #     # Convert to string and strip
    #     str_values = [str(v).strip() if v is not None else '' for v in values]
        
    #     # Check for boolean patterns
    #     bool_patterns = {'true', 'false', 'yes', 'no', '0', '1', 't', 'f', 'y', 'n'}
    #     if all(v.lower() in bool_patterns for v in str_values if v):
    #         return 'boolean'
        
    #     # Check for integer
    #     int_count = 0
    #     for v in str_values:
    #         if v:
    #             try:
    #                 int(v)
    #                 int_count += 1
    #             except:
    #                 pass
        
    #     if int_count / max(len([v for v in str_values if v]), 1) > 0.9:
    #         return 'integer'
        
    #     # Check for float
    #     float_count = 0
    #     for v in str_values:
    #         if v:
    #             try:
    #                 float(v)
    #                 float_count += 1
    #             except:
    #                 pass
        
    #     if float_count / max(len([v for v in str_values if v]), 1) > 0.9:
    #         return 'float'
        
    #     # Check for datetime patterns (dynamic discovery)
    #     date_patterns = self._discover_date_patterns(str_values)
    #     if date_patterns:
    #         return 'datetime'
        
    #     # Default to string
    #     return 'string'
    
    # def _discover_date_patterns(self, values: List[str]) -> List[str]:
    #     """
    #     Dynamically discover date patterns in values.
    #     No pre-assumed date formats.
    #     """
    #     patterns = []
        
    #     # Common date separators
    #     separators = ['-', '/', '.', ' ']
        
    #     for val in values:
    #         if not val:
    #             continue
            
    #         # Check if value has date-like structure
    #         # Has numbers and separators in a structured way
    #         for sep in separators:
    #             if sep in val:
    #                 parts = val.split(sep)
    #                 if len(parts) == 3:
    #                     # Check if parts could be day/month/year
    #                     if all(p.isdigit() for p in parts):
    #                         patterns.append(f"digit{sep}digit{sep}digit")
                    
    #                 # Check for ISO-like format
    #                 if 'T' in val and ':' in val:
    #                     patterns.append("iso_timestamp")
        
    #     return list(set(patterns))
    def _discover_date_patterns(self, values: List[str]) -> List[str]:
        """
        Dynamically discover date patterns in values.
        Enhanced to distinguish dates from phone numbers and other patterns.
        """
        patterns = []
        
        # Common date separators
        separators = ['-', '/', '.', ' ']
        
        for val in values:
            if not val:
                continue
            
            val_str = str(val)
            
            # Skip if looks like phone number
            if self._looks_like_phone_number(val_str):
                continue
            
            # Skip if looks like ID/code
            if self._looks_like_id_or_code(val_str):
                continue
            
            # Check if value has date-like structure
            for sep in separators:
                if sep in val_str:
                    parts = val_str.split(sep)
                    if len(parts) == 3:
                        # Check if parts could be day/month/year
                        if all(p.isdigit() for p in parts):
                            # Validate as possible date
                            if self._could_be_date(parts):
                                patterns.append(f"digit{sep}digit{sep}digit")
                    
                    # Check for ISO-like format
                    if 'T' in val_str and ':' in val_str and '-' in val_str:
                        patterns.append("iso_timestamp")
        
        return list(set(patterns))

    def _looks_like_phone_number(self, value: str) -> bool:
        """
        Check if value looks like a phone number.
        """
        # Remove common phone separators
        cleaned = value.replace('-', '').replace('(', '').replace(')', '').replace(' ', '')
        
        # Phone numbers are usually 10 digits (US) or 7-15 digits internationally
        if cleaned.isdigit():
            length = len(cleaned)
            return 7 <= length <= 15
        
        return False

    def _looks_like_id_or_code(self, value: str) -> bool:
        """
        Check if value looks like an ID or code (not a date).
        """
        # IDs/codes often have consistent length and high digit ratio
        if not value:
            return False
        
        # Check length consistency (dates vary, IDs are often fixed length)
        if 5 <= len(value) <= 20:
            # Check if mostly alphanumeric with consistent pattern
            if value.isalnum() or all(c.isdigit() or c.isalpha() or c in '-_' for c in value):
                # Check digit ratio
                digit_count = sum(1 for c in value if c.isdigit())
                digit_ratio = digit_count / len(value)
                
                # IDs often have high digit ratio but not mixed with date separators
                if digit_ratio > 0.7:
                    # Check if has common ID patterns
                    common_id_patterns = ['ID', 'NO', 'NUM', 'REF', 'CODE']
                    for pattern in common_id_patterns:
                        if pattern.lower() in value.lower():
                            return True
                
                # Product codes, SKUs often have alphanumeric patterns
                if any(c.isalpha() for c in value) and any(c.isdigit() for c in value):
                    return True
        
        return False

    def _could_be_date(self, parts: List[str]) -> bool:
        """
        Check if numeric parts could represent a valid date.
        """
        if len(parts) != 3:
            return False
        
        try:
            # Try different date interpretations
            day_month_year_combinations = [
                (int(parts[0]), int(parts[1]), int(parts[2])),  # DD-MM-YYYY
                (int(parts[1]), int(parts[0]), int(parts[2])),  # MM-DD-YYYY
                (int(parts[2]), int(parts[1]), int(parts[0])),  # YYYY-MM-DD
            ]
            
            for month, day, year in day_month_year_combinations:
                # Basic date validation
                if 1 <= month <= 12 and 1 <= day <= 31 and year > 0:
                    # More sophisticated: check day valid for month
                    if month in [1, 3, 5, 7, 8, 10, 12] and day <= 31:
                        return True
                    elif month in [4, 6, 9, 11] and day <= 30:
                        return True
                    elif month == 2:
                        # February
                        if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) and day <= 29:
                            return True
                        elif day <= 28:
                            return True
            
            return False
        except:
            return False

    def infer_data_type(self, values: List) -> str:
        """
        Enhanced data type inference.
        """
        if not values:
            return 'unknown'
        
        # Convert to string and strip
        str_values = [str(v).strip() if v is not None else '' for v in values]
        
        # Filter out empty values
        non_empty_values = [v for v in str_values if v]
        if not non_empty_values:
            return 'string'
        
        # Check for boolean patterns
        bool_patterns = {'true', 'false', 'yes', 'no', '0', '1', 't', 'f', 'y', 'n'}
        if all(v.lower() in bool_patterns for v in non_empty_values):
            return 'boolean'
        
        # First check: could be phone numbers?
        phone_count = sum(1 for v in non_empty_values if self._looks_like_phone_number(v))
        if phone_count / len(non_empty_values) > 0.8:
            return 'string'  # Phone numbers are strings
        
        # Check for integer
        int_count = 0
        for v in non_empty_values:
            try:
                # Check if it's a valid integer
                int_val = int(v)
                # Additional check: not a year (to avoid confusion with dates)
                if 1000 <= int_val <= 9999:
                    # Could be a year, so be cautious
                    pass
                else:
                    int_count += 1
            except:
                pass
        
        if int_count / len(non_empty_values) > 0.9:
            return 'integer'
        
        # Check for float
        float_count = 0
        for v in non_empty_values:
            try:
                float(v)
                float_count += 1
            except:
                pass
        
        if float_count / len(non_empty_values) > 0.9:
            return 'float'
        
        # Enhanced date detection with phone number filtering
        date_candidates = []
        for v in non_empty_values:
            if not self._looks_like_phone_number(v) and not self._looks_like_id_or_code(v):
                date_candidates.append(v)
        
        if date_candidates:
            date_patterns = self._discover_date_patterns(date_candidates)
            if date_patterns and len(date_candidates) / len(non_empty_values) > 0.7:
                return 'datetime' 
        
        # Default to string
        return 'string'
    
    def _analyze_pattern_lengths(self, patterns: List[str]) -> Dict[str, float]:
        """Analyze statistical features of pattern lengths."""
        if not patterns:
            return {}
        
        lengths = [len(pattern) for pattern in patterns]
        
        return {
            'mean_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths)),
            'min_length': float(min(lengths)),
            'max_length': float(max(lengths)),
            'median_length': float(np.median(lengths))
        }
    
    def _create_length_histogram(self, lengths: List[int]) -> Dict[int, float]:
        """Create a histogram of pattern lengths."""
        if not lengths:
            return {}
        
        hist = {}
        total = len(lengths)
        for length in lengths:
            hist[length] = hist.get(length, 0) + 1
        
        # Convert to percentages
        return {length: count/total for length, count in hist.items()}
    
    def _pattern_matches(self, value: str, pattern_template: str) -> bool:
        """Check if a value matches a pattern template."""
        separators = ['-', '_', '.', ' ', '@', '+', '(', ')', '/', ':', ';', ',', '|', '#', '$', '%', '&', '=', '?']
        
        generated_pattern = []
        for char in str(value):
            if char in separators:
                generated_pattern.append(char)
            elif char.isdigit():
                generated_pattern.append('D')
            elif char.isalpha():
                generated_pattern.append('L')
            else:
                generated_pattern.append('S')
        
        return ''.join(generated_pattern) == pattern_template
    
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
        
        # Use ALL values for pattern analysis
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
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': float(min(lengths)),
            'max': float(max(lengths))
        }
    
    def _discover_separator_patterns(self, values: List[str]) -> Dict[str, Any]:
        """Cluster similar patterns and analyze distribution using ALL data."""
        separators = ['-', '_', '.', ' ', '@', '+', '(', ')', '/', ':', ';', ',', '|', '#', '$', '%', '&', '=', '?']
        all_patterns = []
        
        for val in values:
            if not val or not isinstance(val, str):
                continue
                
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
            
            all_patterns.append(''.join(pattern))
        
        if not all_patterns:
            return {}
        
        # Analyze pattern lengths
        length_stats = self._analyze_pattern_lengths(all_patterns)
        
        # Group patterns by key features
        pattern_groups = defaultdict(list)
        for pattern in all_patterns:
            # Create grouping key based on pattern characteristics
            key_parts = [
                f"len{len(pattern)}",
                f"D{pattern.count('D')}",
                f"L{pattern.count('L')}",
                f"S{pattern.count('S')}",
                f"sep{sum(1 for c in pattern if c in separators)}"
            ]
            key = '_'.join(key_parts)
            pattern_groups[key].append(pattern)
        
        # Analyze group distribution
        result = {
            'total_values': len(all_patterns),
            'unique_patterns': len(set(all_patterns)),
            'pattern_diversity': len(set(all_patterns)) / len(all_patterns),
            'length_stats': length_stats,
            'dominant_pattern_groups': []
        }
        
        # Get dominant pattern groups (covering significant portion of data)
        sorted_groups = sorted(pattern_groups.items(), 
                              key=lambda x: len(x[1]), 
                              reverse=True)
        
        cumulative_coverage = 0
        for group_key, patterns in sorted_groups:
            group_freq = len(patterns) / len(all_patterns)
            if group_freq >= 0.01:  # Only include groups with at least 1% frequency
                # Find an example value that matches this pattern
                example_value = ""
                for val in values:
                    if self._pattern_matches(val, patterns[0]):
                        example_value = val
                        break
                
                result['dominant_pattern_groups'].append({
                    'group_key': group_key,
                    'frequency': group_freq,
                    'count': len(patterns),
                    'example_pattern': patterns[0] if patterns else "",
                    'example_value': example_value
                })
                cumulative_coverage += group_freq
            
            # Stop when we've covered 90% of data or have top 5 groups
            if cumulative_coverage >= 0.9 or len(result['dominant_pattern_groups']) >= 5:
                break
        
        return result

    
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
        Uses ALL data for pattern analysis.
        """
        schema_info = {
            'filename': filename,
            'file_path': file_path,
            'columns': [],
            'column_count': 0,
            'value_samples': {},
            'column_types': {},
            'description': {},
            'value_patterns': {}
        }
        
        try:
            if filename.endswith('.csv'):
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, use 'utf-8' with error handling
                    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')

                # # Read CSV - use all data for pattern analysis
                # df = pd.read_csv(file_path)
            elif filename.endswith('.json'):
                # Read JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Handle single object or nested structure
                    # Flatten if needed
                    df = pd.json_normalize(data)
                else:
                    raise ValueError(f"Unsupported JSON structure in {filename}")
            else:
                return schema_info
            
            self.sample_size = None
            # If sample_size is specified, use it for value samples only
            if self.sample_size is not None:
                print("*"*100)
                df_sample = df.head(self.sample_size)
            else:
                print("-"*100)
                df_sample = df
            
            # Store basic column info
            schema_info['columns'] = list(df.columns)
            schema_info['column_count'] = len(df.columns)
            
            # Extract enhanced information for each column
            for col in df.columns:
                # Get value samples (limited for display)
                samples = df_sample[col].dropna().head(20).tolist()
                schema_info['value_samples'][col] = samples
                
                # For pattern analysis, use ALL data from the column
                all_values = df[col].dropna().tolist()
                
                # Infer data type from all values
                col_type = self.infer_data_type(all_values)
                schema_info['column_types'][col] = col_type

                #generate descriptions of columns(headers)
                description = describe_column(col)
                schema_info['description'][col] = description
                
                # Extract value patterns from ALL data
                patterns = self.extract_value_patterns(all_values)
                schema_info['value_patterns'][col] = patterns
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            # Return basic info if full extraction fails
            schema_info['error'] = str(e)
        
        return schema_info
    
    def extract_from_directory(self, directory: str) -> Dict[str, Any]:
        """
        Extract enhanced schema information from all CSV/JSON files in directory.
        Uses ALL data for pattern analysis.
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
                                        if pattern_key == 'dominant_pattern_groups':
                                            print(f"        {pattern_key}:")
                                            for group in pattern_val:
                                                print(f"          Group: {group['group_key']}")
                                                print(f"            Frequency: {group['frequency']:.2%}")
                                                print(f"            Example pattern: {group['example_pattern']}")
                                                print(f"            Example value: {group['example_value']}")
                                        elif pattern_key == 'char_distribution':
                                            print(f"        {pattern_key}:")
                                            for char_type, ratio in pattern_val.items():
                                                print(f"          {char_type}: {ratio:.2%}")
                                        elif pattern_key == 'length_stats':
                                            print(f"        {pattern_key}:")
                                            for stat_name, stat_value in pattern_val.items():
                                                print(f"          {stat_name}: {stat_value:.2f}")
                                        elif pattern_key == 'separator_patterns':
                                            print(f"        {pattern_key}:")
                                            for subpattern_key, subpattern_val in pattern_val.items():
                                                if subpattern_key not in ['dominant_pattern_groups', 'char_distribution', 'length_stats']:
                                                    print(f"          {subpattern_key}: {subpattern_val}")
                                        else:
                                            print(f"          {pattern_key}: {pattern_val}")
                                else:
                                    print(f"    {subvalue}")
                elif key == 'column_types':
                    print(f"  {value}")
                elif isinstance(value, list):
                    print(f"  {value}")
                else:
                    print(f"  {value}")
    def save_all_details_to_txt(self,enhanced_schemas: Dict[str, Any], filename: str = 'all_schema_details.txt'):
        """
        Save ALL details from enhanced_schemas to a text file.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("COMPLETE ENHANCED SCHEMAS DATA\n")
            f.write("=" * 100 + "\n\n")
            
            import pprint
            pp = pprint.PrettyPrinter(indent=2, width=120, stream=f)
            
            for file_name, schema in enhanced_schemas.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"FILE: {file_name}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write("COMPLETE SCHEMA DATA:\n")
                pp.pprint(schema)
                f.write("\n")
        
        print(f"📄 ALL details saved to: {filename}")
    
def main_meta_data_extractor(directory):
     # Initialize extractor - None means use ALL data for pattern analysis
    extractor = EnhancedDataExtractor(sample_size=20)  # 20 samples for display, but ALL for patterns
    # Extract from directory
    enhanced_schemas = extractor.extract_from_directory(directory)
    # extractor.save_all_details_to_txt(enhanced_schemas)
    return enhanced_schemas
    
    # # Usage:
    # extractor.save_all_details_to_txt(enhanced_schemas, 'all_schema_details.txt')
    # # Print full schema dictionary
    # extractor.print_full_schema_dict(enhanced_schemas)
    
    # # Save to file for later use
    # import pickle
    # with open('enhanced_schemas.pkl', 'wb') as f:
    #     pickle.dump(enhanced_schemas, f)
    
    # print("\n✅ Enhanced schema extraction complete!")
    # print(f"📊 Total files: {len(enhanced_schemas)}")
    # print(f"💾 Saved to: enhanced_schemas.pkl")



# # Example usage
# if __name__ == "__main__":
#     # Initialize extractor - None means use ALL data for pattern analysis
#     extractor = EnhancedDataExtractor(sample_size=20)  # 20 samples for display, but ALL for patterns
    
#     # Extract from directory
#     directory = '../../Amazon Sales Dataset/raw data'
#     enhanced_schemas = extractor.extract_from_directory(directory)
    
    
#     # Usage:
#     extractor.save_all_details_to_txt(enhanced_schemas, 'all_schema_details.txt')
#     # Print full schema dictionary
#     extractor.print_full_schema_dict(enhanced_schemas)
    
#     # Save to file for later use
#     import pickle
#     with open('enhanced_schemas.pkl', 'wb') as f:
#         pickle.dump(enhanced_schemas, f)
    
#     print("\n✅ Enhanced schema extraction complete!")
#     print(f"📊 Total files: {len(enhanced_schemas)}")
#     print(f"💾 Saved to: enhanced_schemas.pkl")