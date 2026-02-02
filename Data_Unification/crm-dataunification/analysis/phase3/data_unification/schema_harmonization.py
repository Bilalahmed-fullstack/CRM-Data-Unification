import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
import os
from datetime import datetime
from dataclasses import dataclass
from colorama import init, Fore, Style
import warnings
warnings.filterwarnings('ignore')

# Initialize colorama
init(autoreset=True)

@dataclass
class MergeStrategy:
    """Data class for merge strategy decision"""
    cluster_id: int
    strategy_type: str  # 'horizontal', 'vertical', 'hybrid'
    merge_key: str = None
    temporal_field: str = None
    priority_order: List[str] = None
    confidence_level: float = 0.0
    reasoning: List[str] = None
    
    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = []


class DataUnificationPhase4:
    def __init__(self, 
                 clustering_file: str = '../final_clustering_results.json',
                 mapping_file: str = '../simplified_column_mappings.json',
                 data_dir: str = '../../../Amazon Sales Dataset/raw data'):
        """
        Initialize Phase 4: Data Unification
        
        Args:
            clustering_file: Path to clustering results
            mapping_file: Path to column mapping results
            data_dir: Directory containing original data files
        """
        print(Fore.CYAN + "\n" + "="*100)
        print("PHASE 4: DATA UNIFICATION & MERGING")
        print("="*100 + Style.RESET_ALL)
        
        # Load configuration files
        print(Fore.YELLOW + "\n📂 Loading configuration files..." + Style.RESET_ALL)
        self.clustering_data = self._load_json(clustering_file)
        self.mapping_data = self._load_json(mapping_file)
        
        if not self.clustering_data or not self.mapping_data:
            raise ValueError("Failed to load required configuration files")
        
        self.data_dir = data_dir
        self.clusters = self.clustering_data.get('clusters', [])
        self.file_names = self.clustering_data.get('file_names', [])
        
        # Store assessment results
        self.assessments = {}
        self.merge_strategies = {}
        self.harmonized_schemas = {}
        
        print(Fore.GREEN + f"✅ Loaded {len(self.clusters)} clusters" + Style.RESET_ALL)
    
    def _load_json(self, filepath: str) -> Dict:
        """Load JSON file with error handling"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(Fore.RED + f"❌ Error: {filepath} not found" + Style.RESET_ALL)
            return {}
        except json.JSONDecodeError as e:
            print(Fore.RED + f"❌ Error parsing {filepath}: {e}" + Style.RESET_ALL)
            return {}
    
    # ============================================
    # STEP 1: DATA ASSESSMENT & STRATEGY SELECTION
    # ============================================
    
    def assess_data_and_select_strategy(self):
        """
        STEP 1: Perform data assessment and select merge strategy for each cluster
        """
        print(Fore.CYAN + "\n" + "━"*50)
        print("STEP 1: DATA ASSESSMENT & STRATEGY SELECTION")
        print("━"*50 + Style.RESET_ALL)
        
        for cluster_idx, cluster_files in enumerate(self.clusters, 1):
            print(f"\n{Fore.YELLOW}📊 Assessing Cluster {cluster_idx}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{'─'*40}{Style.RESET_ALL}")
            
            # Perform assessment
            assessment = self._assess_cluster(cluster_idx, cluster_files)
            self.assessments[cluster_idx] = assessment
            
            # Select merge strategy
            strategy = self._select_merge_strategy(cluster_idx, assessment)
            self.merge_strategies[cluster_idx] = strategy
            
            # Print results
            self._print_assessment_results(cluster_idx, assessment, strategy)
        
        print(Fore.GREEN + "\n✅ STEP 1 COMPLETE: All clusters assessed and strategies selected" + Style.RESET_ALL)
        return self.merge_strategies
    def get_path(self,filename):
        for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file == filename:
                        return os.path.join(root, file)
        return ""
                    
    def _assess_cluster(self, cluster_id: int, cluster_files: List[str]) -> Dict[str, Any]:
        """
        Assess data characteristics for a cluster
        """
        assessment = {
            'cluster_id': cluster_id,
            'files': cluster_files,
            'file_assessments': {},
            'summary': {},
            'warnings': [],
            'recommendations': []
        }
        
        total_rows = 0
        total_columns = 0
        all_column_names = set()
        file_sizes = []
        temporal_fields = []
        
        # Assess each file in the cluster
        for filename in cluster_files:
            # file_path = os.path.join(self.data_dir, filename)
            file_path = self.get_path(filename)
            file_assessment = self._assess_file(file_path)
            assessment['file_assessments'][filename] = file_assessment
            
            # Aggregate statistics
            total_rows += file_assessment.get('row_count', 0)
            total_columns += file_assessment.get('column_count', 0)
            all_column_names.update(file_assessment.get('columns', []))
            file_sizes.append(file_assessment.get('file_size_mb', 0))
            
            # Check for temporal fields
            if file_assessment.get('temporal_fields'):
                temporal_fields.extend(file_assessment['temporal_fields'])
        
        # Calculate summary statistics
        assessment['summary'] = {
            'total_files': len(cluster_files),
            'total_rows': total_rows,
            'avg_rows_per_file': total_rows / len(cluster_files) if cluster_files else 0,
            'unique_columns_across_files': len(all_column_names),
            'total_file_size_mb': sum(file_sizes),
            'avg_file_size_mb': np.mean(file_sizes) if file_sizes else 0,
            'has_temporal_data': len(temporal_fields) > 0,
            'temporal_fields': list(set(temporal_fields)),
            'column_overlap_ratio': self._calculate_column_overlap(assessment['file_assessments']),
            'estimated_overlap_rows': self._estimate_row_overlap(assessment['file_assessments'])
        }
        
        # Generate warnings and recommendations
        self._generate_assessment_insights(assessment)
        
        return assessment
    
    def _assess_file(self, file_path: str) -> Dict[str, Any]:
        """
        Assess a single data file
        """
        assessment = {
            'file_path': file_path,
            'exists': False,
            'error': None,
            'row_count': 0,
            'column_count': 0,
            'columns': [],
            'data_types': {},
            'null_percentages': {},
            'unique_counts': {},
            'file_size_mb': 0,
            'sample_data': {},
            'temporal_fields': [],
            'candidate_keys': []
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            assessment['error'] = f"File not found: {file_path}"
            return assessment
        
        assessment['exists'] = True
        
        try:
            # Get file size
            assessment['file_size_mb'] = os.path.getsize(file_path) / (1024 * 1024)
            
            # Load data (first 1000 rows for assessment)
            df = self._load_data_file(file_path, sample_rows=1000)
            
            if df is not None:
                assessment['row_count'] = len(df)
                assessment['column_count'] = len(df.columns)
                assessment['columns'] = list(df.columns)
                
                # Data type assessment
                for col in df.columns:
                    # Data type
                    dtype = str(df[col].dtype)
                    assessment['data_types'][col] = dtype
                    
                    # Null percentage
                    null_pct = df[col].isnull().sum() / len(df) * 100
                    assessment['null_percentages'][col] = null_pct
                    
                    # Unique count
                    unique_count = df[col].nunique()
                    assessment['unique_counts'][col] = unique_count
                    
                    # Check for temporal fields
                    if self._is_temporal_field(col, df[col]):
                        assessment['temporal_fields'].append(col)
                    
                    # Sample data (first non-null value)
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample_val is not None:
                        assessment['sample_data'][col] = str(sample_val)[:50]  # Truncate
                
                # Identify candidate keys (columns that might be unique identifiers)
                assessment['candidate_keys'] = self._identify_candidate_keys(df)
                
        except Exception as e:
            assessment['error'] = str(e)
        
        return assessment
    
    def _load_data_file(self, file_path: str, sample_rows: int = None) -> pd.DataFrame:
        """Load data file with appropriate engine"""
        try:
            # Determine file type
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                if sample_rows:
                    return pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
                else:
                    return pd.read_csv(file_path, low_memory=False)
            elif ext in ['.xlsx', '.xls']:
                if sample_rows:
                    return pd.read_excel(file_path, nrows=sample_rows)
                else:
                    return pd.read_excel(file_path)
            elif ext == '.parquet':
                if sample_rows:
                    return pd.read_parquet(file_path).head(sample_rows)
                else:
                    return pd.read_parquet(file_path)
            elif ext == '.json':
                if sample_rows:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        # Handle single object or nested structure
                        # Flatten if needed
                        df = pd.json_normalize(data)
                    return df    
                    # return pd.read_json(file_path, lines=True).head(sample_rows)
                else:
                    return pd.read_json(file_path, lines=True)
            else:
                # Try CSV as default
                if sample_rows:
                    return pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
                else:
                    return pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(Fore.RED + f"⚠️  Error loading {file_path}: {e}" + Style.RESET_ALL)
            return None
    
    def _is_temporal_field(self, col_name: str, series: pd.Series) -> bool:
        """Check if a field contains temporal data"""
        col_lower = col_name.lower()
        temporal_keywords = ['date', 'time', 'timestamp', 'year', 'month', 'day', 'hour']
        
        # Check column name
        if any(keyword in col_lower for keyword in temporal_keywords):
            return True
        
        # Check data type
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Check sample values
        try:
            sample = series.dropna().iloc[0] if not series.dropna().empty else None
            if sample:
                # Try to parse as date
                pd.to_datetime(str(sample), errors='raise')
                return True
        except:
            pass
        
        return False
    
    def _identify_candidate_keys(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that could serve as unique keys"""
        candidate_keys = []
        
        for col in df.columns:
            # Check uniqueness
            unique_ratio = df[col].nunique() / len(df)
            
            # Good candidate if:
            # 1. High uniqueness (> 95% unique values)
            # 2. Not too many nulls (< 5%)
            # 3. Reasonable data type (not free text)
            null_pct = df[col].isnull().sum() / len(df) * 100
            
            if (unique_ratio > 0.95 and 
                null_pct < 5 and 
                not pd.api.types.is_object_dtype(df[col])):
                candidate_keys.append(col)
        
        return candidate_keys
    
    def _calculate_column_overlap(self, file_assessments: Dict) -> float:
        """Calculate how much columns overlap between files"""
        if not file_assessments:
            return 0.0
        
        all_columns = []
        for assessment in file_assessments.values():
            if assessment.get('columns'):
                all_columns.append(set(assessment['columns']))
        
        if len(all_columns) < 2:
            return 1.0  # Single file has perfect overlap with itself
        
        # Calculate Jaccard similarity between all pairs
        similarities = []
        for i in range(len(all_columns)):
            for j in range(i+1, len(all_columns)):
                intersection = len(all_columns[i] & all_columns[j])
                union = len(all_columns[i] | all_columns[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _estimate_row_overlap(self, file_assessments: Dict) -> float:
        """Estimate potential row overlap between files"""
        # Simple heuristic based on column overlap and data characteristics
        column_overlap = self._calculate_column_overlap(file_assessments)
        
        # If columns are very similar, likely overlapping entities
        if column_overlap > 0.8:
            return 0.7  # High estimated overlap
        elif column_overlap > 0.5:
            return 0.4  # Medium estimated overlap
        else:
            return 0.1  # Low estimated overlap
    
    def _generate_assessment_insights(self, assessment: Dict):
        """Generate warnings and recommendations based on assessment"""
        summary = assessment['summary']
        
        # Warnings
        if summary['total_file_size_mb'] > 100:
            assessment['warnings'].append("Large total file size (>100MB) - consider chunked processing")
        
        if summary['avg_rows_per_file'] > 100000:
            assessment['warnings'].append("Very large files (>100K rows) - memory considerations needed")
        
        if any(f['error'] for f in assessment['file_assessments'].values()):
            assessment['warnings'].append("Some files had errors during assessment")
        
        # Recommendations
        if summary['column_overlap_ratio'] > 0.7:
            assessment['recommendations'].append("High column overlap suggests horizontal merge")
        elif summary['column_overlap_ratio'] < 0.3:
            assessment['recommendations'].append("Low column overlap suggests vertical merge")
        else:
            assessment['recommendations'].append("Moderate column overlap suggests hybrid merge")
        
        if summary['has_temporal_data']:
            assessment['recommendations'].append(f"Use temporal field '{summary['temporal_fields'][0]}' for conflict resolution")
    
    def _select_merge_strategy(self, cluster_id: int, assessment: Dict) -> MergeStrategy:
        """
        Select appropriate merge strategy for the cluster
        """
        summary = assessment['summary']
        file_assessments = assessment['file_assessments']
        
        # Get mapping data for this cluster
        cluster_key = f"Cluster_{cluster_id}"
        mapping_info = self.mapping_data.get(cluster_key, {})
        golden_schema = mapping_info.get('golden_schema', {})
        quality_metrics = mapping_info.get('quality_metrics', {})
        
        # Decision logic
        reasoning = []
        
        # 1. Determine merge type
        column_overlap = summary['column_overlap_ratio']
        
        if column_overlap > 0.7:
            strategy_type = 'horizontal'
            reasoning.append(f"High column overlap ({column_overlap:.2f}) - files represent same entity")
        elif column_overlap < 0.3:
            strategy_type = 'vertical'
            reasoning.append(f"Low column overlap ({column_overlap:.2f}) - files represent complementary attributes")
        else:
            strategy_type = 'hybrid'
            reasoning.append(f"Moderate column overlap ({column_overlap:.2f}) - mixed entity representation")
        
        # 2. Determine merge key (for vertical/hybrid)
        merge_key = None
        if strategy_type in ['vertical', 'hybrid']:
            # Look for common candidate keys across files
            all_candidate_keys = []
            for fname, f_assess in file_assessments.items():
                all_candidate_keys.extend(f_assess.get('candidate_keys', []))
            
            key_counts = Counter(all_candidate_keys)
            if key_counts:
                # Choose most common candidate key
                merge_key, count = key_counts.most_common(1)[0]
                if count > 1:
                    reasoning.append(f"Using '{merge_key}' as merge key (appears in {count} files)")
                else:
                    reasoning.append(f"Potential merge key '{merge_key}' but appears in only 1 file")
                    merge_key = None  # Not reliable
        
        # 3. Determine temporal field
        temporal_field = None
        if summary['has_temporal_data'] and summary['temporal_fields']:
            temporal_field = summary['temporal_fields'][0]
            reasoning.append(f"Temporal field '{temporal_field}' available for conflict resolution")
        
        # 4. Determine priority order
        priority_order = list(file_assessments.keys())
        if temporal_field:
            # Try to order by temporal recency if we can infer
            reasoning.append("Priority order based on file names (temporal inference if possible)")
        
        # 5. Calculate confidence level
        confidence = quality_metrics.get('average_similarity', 0.5)
        reasoning.append(f"Mapping confidence: {confidence:.2f}")
        
        return MergeStrategy(
            cluster_id=cluster_id,
            strategy_type=strategy_type,
            merge_key=merge_key,
            temporal_field=temporal_field,
            priority_order=priority_order,
            confidence_level=confidence,
            reasoning=reasoning
        )
    
    def _print_assessment_results(self, cluster_id: int, assessment: Dict, strategy: MergeStrategy):
        """Print assessment results in readable format"""
        summary = assessment['summary']
        
        print(f"{Fore.CYAN}📈 Assessment Summary:{Style.RESET_ALL}")
        print(f"  • Files: {summary['total_files']}")
        print(f"  • Total rows: {summary['total_rows']:,}")
        print(f"  • Unique columns: {summary['unique_columns_across_files']}")
        print(f"  • Column overlap: {summary['column_overlap_ratio']:.2f}")
        print(f"  • Has temporal data: {summary['has_temporal_data']}")
        if summary['has_temporal_data']:
            print(f"  • Temporal fields: {', '.join(summary['temporal_fields'][:3])}")
        
        print(f"\n{Fore.CYAN}🎯 Selected Strategy:{Style.RESET_ALL}")
        print(f"  • Type: {Fore.YELLOW}{strategy.strategy_type.upper()}{Style.RESET_ALL}")
        if strategy.merge_key:
            print(f"  • Merge key: {strategy.merge_key}")
        if strategy.temporal_field:
            print(f"  • Temporal field: {strategy.temporal_field}")
        print(f"  • Confidence: {strategy.confidence_level:.2f}")
        
        if assessment['warnings']:
            print(f"\n{Fore.RED}⚠️  Warnings:{Style.RESET_ALL}")
            for warning in assessment['warnings'][:3]:  # Show first 3
                print(f"  • {warning}")
        
        print(f"\n{Fore.GREEN}💡 Reasoning:{Style.RESET_ALL}")
        for reason in strategy.reasoning:
            print(f"  • {reason}")
    
    # ============================================
    # STEP 2: SCHEMA HARMONIZATION
    # ============================================
    
    def harmonize_schemas(self):
        """
        STEP 2: Apply golden schema to all files in each cluster
        """
        print(Fore.CYAN + "\n" + "━"*50)
        print("STEP 2: SCHEMA HARMONIZATION")
        print("━"*50 + Style.RESET_ALL)
        
        for cluster_id in range(1, len(self.clusters) + 1):
            cluster_key = f"Cluster_{cluster_id}"
            
            print(f"\n{Fore.YELLOW}🔄 Harmonizing {cluster_key}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{'─'*40}{Style.RESET_ALL}")
            
            # Get cluster information
            cluster_files = self.clusters[cluster_id - 1]
            mapping_info = self.mapping_data.get(cluster_key, {})
            
            if not mapping_info:
                print(Fore.RED + f"⚠️  No mapping data for {cluster_key}" + Style.RESET_ALL)
                continue
            
            golden_schema = mapping_info.get('golden_schema', {})
            mapping_matrix = mapping_info.get('mapping_matrix', {})
            
            if not golden_schema or not mapping_matrix:
                print(Fore.RED + f"⚠️  Incomplete mapping data for {cluster_key}" + Style.RESET_ALL)
                continue
            
            # Create harmonization plan
            harmonization_plan = self._create_harmonization_plan(
                cluster_id, cluster_files, golden_schema, mapping_matrix
            )
            
            # Store harmonization plan
            self.harmonized_schemas[cluster_id] = harmonization_plan
            
            # Print harmonization details
            self._print_harmonization_details(cluster_id, harmonization_plan)
        
        print(Fore.GREEN + "\n✅ STEP 2 COMPLETE: All schemas harmonization plans created" + Style.RESET_ALL)
        return self.harmonized_schemas
    
    def _create_harmonization_plan(self, cluster_id: int, cluster_files: List[str],
                                 golden_schema: Dict, mapping_matrix: Dict) -> Dict[str, Any]:
        """
        Create detailed plan for harmonizing schemas to golden schema
        """
        plan = {
            'cluster_id': cluster_id,
            'golden_schema_name': golden_schema.get('name', f'Golden_Schema_{cluster_id}'),
            'golden_columns': golden_schema.get('columns', []),
            'column_details': golden_schema.get('column_details', {}),
            'file_plans': {},
            'transformations_needed': [],
            'data_type_resolutions': {},
            'warnings': []
        }
        
        # Get mapping information
        mappings = mapping_matrix.get('mappings', [])
        schemas_in_cluster = mapping_matrix.get('schemas', [])
        
        # For each golden column, determine data type consensus
        for golden_col in plan['golden_columns']:
            col_details = plan['column_details'].get(golden_col, {})
            source_mappings = col_details.get('source_mappings', [])
            
            # Collect all data types for this column
            all_types = []
            for mapping in source_mappings:
                all_types.append(mapping.get('type', 'unknown'))
            
            # Determine consensus type
            if all_types:
                type_counts = Counter(all_types)
                consensus_type, count = type_counts.most_common(1)[0]
                confidence = count / len(all_types)
                
                plan['data_type_resolutions'][golden_col] = {
                    'consensus_type': consensus_type,
                    'type_distribution': dict(type_counts),
                    'confidence': confidence,
                    'requires_conversion': len(set(all_types)) > 1
                }
        
        # Create harmonization plan for each file
        for filename in cluster_files:
            if filename not in schemas_in_cluster:
                plan['warnings'].append(f"File {filename} not in mapping matrix")
                continue
            
            file_plan = self._create_file_harmonization_plan(
                filename, golden_schema, mapping_matrix
            )
            plan['file_plans'][filename] = file_plan
            
            # Collect transformations needed
            if file_plan['transformations_needed']:
                plan['transformations_needed'].extend(file_plan['transformations_needed'])
        
        # Remove duplicate transformations
        plan['transformations_needed'] = list(set(plan['transformations_needed']))
        
        return plan
    
    def _create_file_harmonization_plan(self, filename: str, golden_schema: Dict, 
                                      mapping_matrix: Dict) -> Dict[str, Any]:
        """
        Create harmonization plan for a specific file
        """
        file_plan = {
            'filename': filename,
            'golden_columns_mapped': [],
            'column_mappings': {},
            'columns_to_add': [],
            'columns_to_rename': [],
            'columns_to_drop': [],
            'type_conversions': [],
            'transformations_needed': []
        }
        
        # Get assessment data for this file
        cluster_id = self._find_cluster_for_file(filename)
        if cluster_id:
            assessment = self.assessments.get(cluster_id, {}).get('file_assessments', {}).get(filename, {})
            original_columns = assessment.get('columns', [])
            original_types = assessment.get('data_types', {})
        else:
            original_columns = []
            original_types = {}
        
        # Find mappings for this file
        for mapping in mapping_matrix.get('mappings', []):
            golden_col = mapping.get('golden_column', '')
            schema_mappings = mapping.get('schema_mappings', {})
            file_mapping = schema_mappings.get(filename, {})
            
            if file_mapping.get('mapped', False):
                original_col = file_mapping.get('column', '')
                file_plan['golden_columns_mapped'].append(golden_col)
                file_plan['column_mappings'][golden_col] = original_col
                
                # Check if rename needed
                if golden_col != original_col:
                    file_plan['columns_to_rename'].append({
                        'from': original_col,
                        'to': golden_col
                    })
                    file_plan['transformations_needed'].append('rename')
                
                # Check if type conversion needed
                golden_type = mapping.get('consensus_type', 'unknown')
                original_type = original_types.get(original_col, 'unknown')
                
                if (original_type and golden_type and 
                    original_type != golden_type and 
                    self._is_type_conversion_needed(original_type, golden_type)):
                    
                    file_plan['type_conversions'].append({
                        'column': golden_col,
                        'from_type': original_type,
                        'to_type': golden_type,
                        'original_column': original_col
                    })
                    file_plan['transformations_needed'].append('type_conversion')
        
        # Identify columns to add (golden columns not in this file)
        all_golden_cols = golden_schema.get('columns', [])
        file_plan['columns_to_add'] = [col for col in all_golden_cols 
                                      if col not in file_plan['golden_columns_mapped']]
        
        if file_plan['columns_to_add']:
            file_plan['transformations_needed'].append('add_columns')
        
        # Identify columns to drop (original columns not mapped to any golden column)
        mapped_original_cols = list(file_plan['column_mappings'].values())
        file_plan['columns_to_drop'] = [col for col in original_columns 
                                       if col not in mapped_original_cols]
        
        if file_plan['columns_to_drop']:
            file_plan['transformations_needed'].append('drop_columns')
        
        return file_plan
    
    def _find_cluster_for_file(self, filename: str) -> int:
        """Find which cluster a file belongs to"""
        for cluster_idx, cluster_files in enumerate(self.clusters, 1):
            if filename in cluster_files:
                return cluster_idx
        return None
    
    def _is_type_conversion_needed(self, from_type: str, to_type: str) -> bool:
        """Determine if type conversion is actually needed"""
        # Some type pairs are compatible
        compatible_pairs = [
            ('int64', 'float64'),
            ('float64', 'int64'),  # May lose precision
            ('object', 'string'),
            ('string', 'object'),
            ('bool', 'int64'),
            ('int64', 'bool')
        ]
        
        return (from_type, to_type) not in compatible_pairs
    
    def _print_harmonization_details(self, cluster_id: int, plan: Dict):
        """Print harmonization plan details"""
        golden_cols = plan['golden_columns']
        column_details = plan['column_details']
        
        print(f"{Fore.CYAN}🎯 Golden Schema: {plan['golden_schema_name']}{Style.RESET_ALL}")
        print(f"  • Unified columns: {len(golden_cols)}")
        
        # Show data type resolutions
        print(f"\n{Fore.CYAN}📊 Data Type Resolutions:{Style.RESET_ALL}")
        for golden_col, resolution in list(plan['data_type_resolutions'].items())[:5]:  # Show first 5
            if resolution['requires_conversion']:
                status = f"{Fore.YELLOW}Needs conversion{Style.RESET_ALL}"
            else:
                status = f"{Fore.GREEN}Consistent{Style.RESET_ALL}"
            
            print(f"  • {golden_col}: {resolution['consensus_type']} ({status})")
            if resolution['requires_conversion']:
                print(f"    Distribution: {resolution['type_distribution']}")
        
        if len(plan['data_type_resolutions']) > 5:
            print(f"  ... and {len(plan['data_type_resolutions']) - 5} more columns")
        
        # Show file-level transformations
        print(f"\n{Fore.CYAN}🔄 File Transformations:{Style.RESET_ALL}")
        for filename, file_plan in list(plan['file_plans'].items())[:3]:  # Show first 3 files
            print(f"  📁 {filename}:")
            print(f"    • Mapped columns: {len(file_plan['golden_columns_mapped'])}/{len(golden_cols)}")
            
            transformations = file_plan['transformations_needed']
            if transformations:
                print(f"    • Transformations: {', '.join(transformations)}")
                
                # Show examples
                if file_plan['columns_to_rename']:
                    example = file_plan['columns_to_rename'][0]
                    print(f"    • Example rename: {example['from']} → {example['to']}")
                
                if file_plan['type_conversions']:
                    example = file_plan['type_conversions'][0]
                    print(f"    • Example conversion: {example['from_type']} → {example['to_type']}")
            else:
                print(f"    • {Fore.GREEN}No transformations needed{Style.RESET_ALL}")
        
        # Summary
        total_files = len(plan['file_plans'])
        files_with_transforms = sum(1 for fp in plan['file_plans'].values() 
                                   if fp['transformations_needed'])
        
        print(f"\n{Fore.CYAN}📈 Summary:{Style.RESET_ALL}")
        print(f"  • Files to process: {total_files}")
        print(f"  • Files needing transformations: {files_with_transforms}")
        print(f"  • Unique transformations needed: {len(plan['transformations_needed'])}")
    
    # ============================================
    # EXPORT & SUMMARY FUNCTIONS
    # ============================================
    
    def export_assessment_results(self, output_dir: str = 'phase4_results'):
        """Export assessment and harmonization results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export assessments
        assessments_export = {
            'timestamp': datetime.now().isoformat(),
            'total_clusters': len(self.assessments),
            'clusters': self.assessments
        }
        
        with open(f'{output_dir}/data_assessments.json', 'w') as f:
            json.dump(assessments_export, f, indent=2, default=str)
        
        # Export merge strategies
        strategies_export = []
        for cluster_id, strategy in self.merge_strategies.items():
            strategies_export.append({
                'cluster_id': cluster_id,
                'strategy_type': strategy.strategy_type,
                'merge_key': strategy.merge_key,
                'temporal_field': strategy.temporal_field,
                'priority_order': strategy.priority_order,
                'confidence_level': strategy.confidence_level,
                'reasoning': strategy.reasoning
            })
        
        with open(f'{output_dir}/merge_strategies.json', 'w') as f:
            json.dump(strategies_export, f, indent=2)
        
        # Export harmonization plans
        with open(f'{output_dir}/harmonization_plans.json', 'w') as f:
            json.dump(self.harmonized_schemas, f, indent=2, default=str)
        
        print(Fore.GREEN + f"\n💾 Results exported to {output_dir}/" + Style.RESET_ALL)
        
        # Generate summary report
        self._generate_summary_report(output_dir)
    
    def _generate_summary_report(self, output_dir: str):
        """Generate a summary report"""
        report_lines = [
            "=" * 80,
            "DATA UNIFICATION - PHASE 4 SUMMARY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total Clusters: {len(self.clusters)}",
            ""
        ]
        
        # Cluster summaries
        for cluster_id in range(1, len(self.clusters) + 1):
            assessment = self.assessments.get(cluster_id, {})
            strategy = self.merge_strategies.get(cluster_id)
            harmonization = self.harmonized_schemas.get(cluster_id, {})
            
            if assessment and strategy:
                summary = assessment.get('summary', {})
                
                report_lines.extend([
                    f"Cluster {cluster_id}:",
                    f"  Files: {summary.get('total_files', 0)}",
                    f"  Total Rows: {summary.get('total_rows', 0):,}",
                    f"  Strategy: {strategy.strategy_type.upper()}",
                    f"  Merge Key: {strategy.merge_key or 'N/A'}",
                    f"  Golden Columns: {len(harmonization.get('golden_columns', []))}",
                    ""
                ])
        
        # Overall statistics
        total_rows = sum(a.get('summary', {}).get('total_rows', 0) 
                        for a in self.assessments.values())
        total_files = sum(len(c) for c in self.clusters)
        
        strategy_counts = Counter(s.strategy_type for s in self.merge_strategies.values())
        
        report_lines.extend([
            "OVERALL STATISTICS:",
            f"Total Files Analyzed: {total_files}",
            f"Total Rows: {total_rows:,}",
            f"Merge Strategy Distribution:",
        ])
        
        for strategy_type, count in strategy_counts.items():
            report_lines.append(f"  {strategy_type.upper()}: {count} clusters")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        # Write report
        with open(f'{output_dir}/summary_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(Fore.GREEN + f"📋 Summary report generated: {output_dir}/summary_report.txt" + Style.RESET_ALL)


def run_phase4():
    """Main function to run Phase 4"""
    print(Fore.CYAN + """
    ╔══════════════════════════════════════════════════════════════╗
    ║            PHASE 4: DATA UNIFICATION PREPARATION            ║
    ╚══════════════════════════════════════════════════════════════╝
    """ + Style.RESET_ALL)
    
    print("This phase prepares data for unification by:")
    print("1. 📊 Assessing data characteristics")
    print("2. 🎯 Selecting optimal merge strategies")
    print("3. 🔄 Creating schema harmonization plans")
    print()
    
    # # Ask for data directory
    # data_dir = input("Enter path to data directory (default: 'data'): ").strip()
    # if not data_dir:
    #     data_dir = 'data'
    
    # if not os.path.exists(data_dir):
    #     print(Fore.RED + f"❌ Data directory '{data_dir}' not found!" + Style.RESET_ALL)
    #     print("Please create it and add your data files, or specify the correct path.")
    #     return
    
    try:
        # Initialize Phase 4
        phase4 = DataUnificationPhase4()
        
        # Step 1: Data Assessment & Strategy Selection
        print(Fore.YELLOW + "\n▶️ Starting STEP 1: Data Assessment & Strategy Selection" + Style.RESET_ALL)
        strategies = phase4.assess_data_and_select_strategy()
        
        # Step 2: Schema Harmonization
        print(Fore.YELLOW + "\n▶️ Starting STEP 2: Schema Harmonization" + Style.RESET_ALL)
        harmonization_plans = phase4.harmonize_schemas()
        
        # Export results
        export = input("\n📤 Export results? (y/n): ").lower()
        if export == 'y':
            phase4.export_assessment_results()
        
        print(Fore.CYAN + "\n" + "="*100)
        print("✅ PHASE 4 COMPLETE!")
        print("="*100 + Style.RESET_ALL)
        
        print("\n🎯 Next steps:")
        print("1. Review the assessment results in phase4_results/")
        print("2. Based on merge strategies, proceed with actual data merging")
        print("3. Implement conflict resolution based on selected strategies")
        
    except Exception as e:
        print(Fore.RED + f"\n❌ Error: {e}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import pandas as pd
        import numpy as np
        from colorama import init, Fore, Style
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "colorama"])
        print("Packages installed. Please run the script again.")
        sys.exit(0)
    
    run_phase4()