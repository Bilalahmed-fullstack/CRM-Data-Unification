import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style
import warnings
warnings.filterwarnings('ignore')

# Initialize colorama
init(autoreset=True)


class SafeDataUnifier:
    def __init__(self, 
                 data_dir: str = '../../../Amazon Sales Dataset/raw data',
                 output_dir: str = 'unified_data',
                 backup_dir: str = 'backups'):
        """
        SAFE data unifier - never modifies original files
        
        Args:
            data_dir: Root directory containing all data files
            output_dir: Directory to save unified files
            backup_dir: Directory for backup logs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.backup_dir = Path(backup_dir)
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        print(Fore.GREEN + f"✅ Output will be saved to: {self.output_dir}/" + Style.RESET_ALL)
        print(Fore.YELLOW + f"📁 Original files in {self.data_dir}/ will NOT be modified" + Style.RESET_ALL)
        
        # Load previous phase results
        self._load_previous_phase_results()
    
    def _load_previous_phase_results(self):
        """Load results from Phase 4 Steps 1-2"""
        print(Fore.YELLOW + "\n📂 Loading previous phase results..." + Style.RESET_ALL)
        
        try:
            # Load merge strategies
            with open('phase4_results/merge_strategies.json', 'r') as f:
                self.strategies_data = json.load(f)
            print(Fore.GREEN + "✅ Loaded merge_strategies.json" + Style.RESET_ALL)
            
            # Load harmonization plans
            with open('phase4_results/harmonization_plans.json', 'r') as f:
                self.harmonization_data = json.load(f)
            print(Fore.GREEN + "✅ Loaded harmonization_plans.json" + Style.RESET_ALL)
            
            # Load clustering data
            with open('../final_clustering_results.json', 'r') as f:
                self.clustering_data = json.load(f)
            print(Fore.GREEN + "✅ Loaded final_clustering_results.json" + Style.RESET_ALL)
            
            # Load assessments (optional, for reference)
            with open('phase4_results/data_assessments.json', 'r') as f:
                self.assessments_data = json.load(f)
            print(Fore.GREEN + "✅ Loaded data_assessments.json" + Style.RESET_ALL)
            
        except FileNotFoundError as e:
            print(Fore.RED + f"❌ Error loading files: {e}" + Style.RESET_ALL)
            raise
    
    def get_path(self, filename: str) -> Optional[str]:
        """
        Find file path by walking through data directory.
        Uses the function you provided.
        """
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file == filename:
                    return os.path.join(root, file)
        return None
    
    # ============================================
    # STEP 3: DATA LOADING & TRANSFORMATION
    # ============================================
    
    def load_and_transform_data(self, cluster_id: int) -> List[pd.DataFrame]:
        """
        STEP 3: Load and transform data for a cluster
        
        Args:
            cluster_id: ID of the cluster to process
            
        Returns:
            List of transformed DataFrames
        """
        print(Fore.CYAN + f"\n{'━'*50}")
        print(f"STEP 3: LOADING & TRANSFORMING DATA - Cluster {cluster_id}")
        print(f"{'━'*50}" + Style.RESET_ALL)
        
        # Get cluster information
        cluster_files = self.clustering_data['clusters'][cluster_id - 1]
        harmonization_plan = self.harmonization_data.get(str(cluster_id), {})
        
        if not harmonization_plan:
            print(Fore.RED + f"❌ No harmonization plan for Cluster {cluster_id}" + Style.RESET_ALL)
            return []
        
        transformed_dataframes = []
        
        for filename in cluster_files:
            print(f"\n📁 Processing: {filename}")
            
            # Step 3.1: Load data (Extract)
            df = self._load_file_safely(filename)
            if df is None:
                continue
            
            # Step 3.2: Apply transformations (Transform)
            transformed_df = self._apply_harmonization(df, filename, harmonization_plan)
            
            # Add source file info as a column
            transformed_df['_source_file'] = filename
            transformed_df['_cluster_id'] = cluster_id
            
            transformed_dataframes.append(transformed_df)
            
            print(Fore.GREEN + f"✅ Transformed: {len(df)} rows, {len(df.columns)} → {len(transformed_df.columns)} columns" + Style.RESET_ALL)
        
        return transformed_dataframes
    
    def _load_file_safely(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Safely load a data file (read-only)
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame or None if file not found/can't be loaded
        """
        # Find file path
        file_path = self.get_path(filename)
        if not file_path:
            print(Fore.RED + f"  ❌ File not found: {filename}" + Style.RESET_ALL)
            return None
        
        print(f"  📍 Found at: {file_path}")
        
        try:
            # Determine file type by extension
            ext = Path(file_path).suffix.lower()
            
            if ext == '.csv':
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                        print(f"  📊 Loaded with {encoding}: {len(df)} rows, {len(df.columns)} columns")
                        return df
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail, try without specifying encoding
                df = pd.read_csv(file_path, low_memory=False, encoding_errors='ignore')
                print(f"  ⚠️  Loaded with encoding errors ignored: {len(df)} rows")
                return df
                
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                print(f"  📊 Loaded Excel: {len(df)} rows, {len(df.columns)} columns")
                return df
                
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
                print(f"  📊 Loaded Parquet: {len(df)} rows, {len(df.columns)} columns")
                return df
                
            elif ext == '.json':
                try:
                    # df = pd.read_json(file_path)

                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        elif isinstance(data, dict):
                            # Handle single object or nested structure
                            # Flatten if needed
                            df = pd.json_normalize(data)
                        return df    
                    


                except:
                    # Try line-delimited JSON
                    df = pd.read_json(file_path, lines=True)
                print(f"  📊 Loaded JSON: {len(df)} rows, {len(df.columns)} columns")
                return df
                
            else:
                # Try as CSV for unknown extensions
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    print(f"  📊 Loaded as CSV: {len(df)} rows, {len(df.columns)} columns")
                    return df
                except:
                    print(Fore.RED + f"  ❌ Could not load file: {file_path}" + Style.RESET_ALL)
                    return None
                    
        except Exception as e:
            print(Fore.RED + f"  ❌ Error loading {filename}: {e}" + Style.RESET_ALL)
            return None
    
    def _apply_harmonization(self, df: pd.DataFrame, filename: str, 
                           harmonization_plan: Dict) -> pd.DataFrame:
        """
        Apply harmonization transformations to a DataFrame
        
        Args:
            df: Original DataFrame
            filename: Name of the source file
            harmonization_plan: Harmonization plan from previous phase
            
        Returns:
            Transformed DataFrame (new copy)
        """
        # Create a copy to avoid modifying original
        transformed_df = df.copy()
        
        file_plan = harmonization_plan.get('file_plans', {}).get(filename, {})
        golden_columns = harmonization_plan.get('golden_columns', [])
        
        if not file_plan:
            print(f"  ⚠️  No harmonization plan for {filename}, using basic transformation")
            return self._basic_harmonization(transformed_df, golden_columns)
        
        print(f"  🔄 Applying {len(file_plan.get('transformations_needed', []))} transformations")
        
        # 1. Rename columns
        rename_count = 0
        for rename in file_plan.get('columns_to_rename', []):
            old_col = rename['from']
            new_col = rename['to']
            
            if old_col in transformed_df.columns:
                transformed_df.rename(columns={old_col: new_col}, inplace=True)
                rename_count += 1
                # print(f"    • Renamed: {old_col} → {new_col}")
        
        if rename_count > 0:
            print(f"    📝 Renamed {rename_count} columns")
        
        # 2. Add missing golden columns (with nulls)
        add_count = 0
        for golden_col in golden_columns:
            if golden_col not in transformed_df.columns:
                transformed_df[golden_col] = np.nan
                add_count += 1
        
        if add_count > 0:
            print(f"    ➕ Added {add_count} missing columns")
        
        # 3. Apply type conversions
        type_conv_count = 0
        for conversion in file_plan.get('type_conversions', []):
            column = conversion['column']
            from_type = conversion['from_type']
            to_type = conversion['to_type']
            
            if column in transformed_df.columns:
                try:
                    converted = self._convert_column_type(
                        transformed_df[column], from_type, to_type
                    )
                    transformed_df[column] = converted
                    type_conv_count += 1
                    # print(f"    • Type: {column}: {from_type} → {to_type}")
                except Exception as e:
                    print(f"    ⚠️  Failed to convert {column}: {e}")
        
        if type_conv_count > 0:
            print(f"    🔧 Converted {type_conv_count} column types")
        
        # 4. Reorder columns to match golden schema
        # Keep unmapped columns at the end
        current_cols = list(transformed_df.columns)
        unmapped_cols = [
            col for col in current_cols 
            if col not in golden_columns and 
            col not in ['_source_file', '_cluster_id']
        ]
        
        # Order: golden columns first, then special columns, then unmapped
        desired_order = (
            golden_columns + 
            ['_source_file', '_cluster_id'] + 
            unmapped_cols
        )
        
        # Filter to only columns that exist
        existing_cols = [col for col in desired_order if col in transformed_df.columns]
        transformed_df = transformed_df[existing_cols]
        
        return transformed_df
    
    def _basic_harmonization(self, df: pd.DataFrame, golden_columns: List[str]) -> pd.DataFrame:
        """
        Basic harmonization when no specific plan is available
        """
        transformed_df = df.copy()
        
        # Add missing golden columns
        for col in golden_columns:
            if col not in transformed_df.columns:
                transformed_df[col] = np.nan
        
        # Add source tracking columns
        transformed_df['_source_file'] = 'unknown'
        transformed_df['_cluster_id'] = 0
        
        return transformed_df
    
    def _convert_column_type(self, series: pd.Series, from_type: str, to_type: str) -> pd.Series:
        """
        Convert a column from one type to another
        
        Args:
            series: Pandas Series to convert
            from_type: Original data type
            to_type: Target data type
            
        Returns:
            Converted Series
        """
        # Handle common type conversions
        to_type_lower = to_type.lower()
        
        if to_type_lower in ['int', 'integer', 'int64']:
            return pd.to_numeric(series, errors='coerce').astype('Int64')  # nullable integer
        
        elif to_type_lower in ['float', 'float64', 'double']:
            return pd.to_numeric(series, errors='coerce').astype('float64')
        
        elif to_type_lower in ['str', 'string', 'object', 'text']:
            return series.astype(str)
        
        elif to_type_lower in ['date', 'datetime', 'timestamp']:
            return pd.to_datetime(series, errors='coerce')
        
        elif to_type_lower in ['bool', 'boolean']:
            # Map common boolean representations
            if series.dtype == 'object':
                true_values = ['true', 'yes', 'y', '1', 't']
                false_values = ['false', 'no', 'n', '0', 'f']
                return series.str.lower().isin(true_values)
            return series.astype(bool)
        
        else:
            # Unknown type, keep as is
            return series
    
    # ============================================
    # STEP 6: RECORD MERGING
    # ============================================
    
    def merge_records(self, cluster_id: int, transformed_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        STEP 6: Merge records based on strategy
        
        Args:
            cluster_id: ID of the cluster
            transformed_dfs: List of transformed DataFrames
            
        Returns:
            Merged DataFrame
        """
        print(Fore.CYAN + f"\n{'━'*50}")
        print(f"STEP 6: RECORD MERGING - Cluster {cluster_id}")
        print(f"{'━'*50}" + Style.RESET_ALL)
        
        if not transformed_dfs:
            print(Fore.RED + "❌ No transformed data to merge" + Style.RESET_ALL)
            return pd.DataFrame()
        
        # Get merge strategy for this cluster
        strategy_info = next(
            (s for s in self.strategies_data if s['cluster_id'] == cluster_id), 
            None
        )
        
        if not strategy_info:
            print(Fore.YELLOW + f"⚠️  No strategy for Cluster {cluster_id}, using horizontal merge" + Style.RESET_ALL)
            strategy_type = 'horizontal'
        else:
            strategy_type = strategy_info['strategy_type']
            print(f"🎯 Using {strategy_type.upper()} merge strategy")
        
        # Perform merge based on strategy
        if strategy_type == 'horizontal':
            merged_df = self._horizontal_merge(transformed_dfs)
        elif strategy_type == 'vertical':
            merge_key = strategy_info.get('merge_key')
            merged_df = self._vertical_merge(transformed_dfs, merge_key)
        elif strategy_type == 'hybrid':
            merge_key = strategy_info.get('merge_key')
            merged_df = self._hybrid_merge(transformed_dfs, merge_key)
        else:
            print(Fore.RED + f"❌ Unknown strategy: {strategy_type}" + Style.RESET_ALL)
            merged_df = self._horizontal_merge(transformed_dfs)
        
        print(Fore.GREEN + f"✅ Merged {len(transformed_dfs)} files → {len(merged_df)} records" + Style.RESET_ALL)
        return merged_df
    
    def _horizontal_merge(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Horizontal merge (concatenation) - for same entity, different instances
        
        Args:
            dataframes: List of DataFrames to merge
            
        Returns:
            Concatenated DataFrame
        """
        print("  📊 Performing horizontal merge (concatenation)")
        
        # Ensure all DataFrames have the same columns
        all_columns = set()
        for df in dataframes:
            all_columns.update(df.columns)
        
        # Add missing columns to each DataFrame
        aligned_dfs = []
        for df in dataframes:
            df_aligned = df.copy()
            for col in all_columns:
                if col not in df_aligned.columns:
                    df_aligned[col] = np.nan
            aligned_dfs.append(df_aligned)
        
        # Concatenate
        merged = pd.concat(aligned_dfs, ignore_index=True, sort=False)
        
        # Reorder columns consistently
        column_order = sorted(list(all_columns))
        merged = merged[column_order]
        
        return merged
    
    def _vertical_merge(self, dataframes: List[pd.DataFrame], merge_key: str = None) -> pd.DataFrame:
        """
        Vertical merge (joining) - for complementary attributes
        
        Args:
            dataframes: List of DataFrames to merge
            merge_key: Column to join on
            
        Returns:
            Joined DataFrame
        """
        print(f"  📊 Performing vertical merge (joining)")
        
        if not merge_key:
            # Try to find a common key
            common_columns = set.intersection(*[set(df.columns) for df in dataframes])
            common_columns -= {'_source_file', '_cluster_id'}
            
            if common_columns:
                merge_key = next(iter(common_columns))
                print(f"  🔑 Using inferred merge key: {merge_key}")
            else:
                print(Fore.RED + "  ❌ No merge key found, falling back to horizontal merge" + Style.RESET_ALL)
                return self._horizontal_merge(dataframes)
        else:
            print(f"  🔑 Using specified merge key: {merge_key}")
        
        # Start with first DataFrame
        merged = dataframes[0].copy()
        
        # Merge with subsequent DataFrames
        for i, df in enumerate(dataframes[1:], 1):
            try:
                # Perform outer join to keep all records
                merged = pd.merge(
                    merged, 
                    df, 
                    on=merge_key, 
                    how='outer',
                    suffixes=(f'_{i-1}', f'_{i}')
                )
            except Exception as e:
                print(Fore.RED + f"  ❌ Error merging DataFrame {i}: {e}" + Style.RESET_ALL)
                # Fall back to concatenation
                merged = pd.concat([merged, df], ignore_index=True)
        
        # Handle duplicate columns from merges
        merged = self._resolve_duplicate_columns(merged)
        
        return merged
    
    def _hybrid_merge(self, dataframes: List[pd.DataFrame], merge_key: str = None) -> pd.DataFrame:
        """
        Hybrid merge - combination of horizontal and vertical
        
        Args:
            dataframes: List of DataFrames to merge
            merge_key: Column to join on when needed
            
        Returns:
            Merged DataFrame
        """
        print("  📊 Performing hybrid merge")
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Group DataFrames by similarity (simplified: check column overlap)
        # For now, use horizontal merge for all, then deduplicate
        horizontally_merged = self._horizontal_merge(dataframes)
        
        # Apply deduplication if merge_key is available
        if merge_key and merge_key in horizontally_merged.columns:
            print(f"  🔍 Applying deduplication on key: {merge_key}")
            
            # Mark duplicates
            duplicates = horizontally_merged.duplicated(subset=[merge_key], keep='first')
            duplicate_count = duplicates.sum()
            
            if duplicate_count > 0:
                print(f"  🗑️  Found {duplicate_count} duplicate records by key")
                # Keep first occurrence of each duplicate
                deduped = horizontally_merged.drop_duplicates(subset=[merge_key], keep='first')
                print(f"  📉 Removed {duplicate_count} duplicates → {len(deduped)} unique records")
                return deduped
        
        return horizontally_merged
    
    def _resolve_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve duplicate column names after merge
        
        Args:
            df: DataFrame with potential duplicate columns
            
        Returns:
            DataFrame with resolved columns
        """
        # Identify duplicate column names (excluding suffix numbers)
        column_counts = {}
        for col in df.columns:
            base_col = col.split('_')[0]  # Simple heuristic
            column_counts[base_col] = column_counts.get(base_col, 0) + 1
        
        # For columns that appear multiple times, consolidate
        for base_col, count in column_counts.items():
            if count > 1:
                # Find all columns starting with this base
                duplicate_cols = [c for c in df.columns if c.startswith(base_col + '_') or c == base_col]
                
                if len(duplicate_cols) > 1:
                    print(f"    🔧 Consolidating {len(duplicate_cols)} columns for '{base_col}'")
                    
                    # Create consolidated column
                    consolidated = pd.Series(np.nan, index=df.index)
                    
                    for col in duplicate_cols:
                        # Take non-null values from each column
                        mask = df[col].notna() & consolidated.isna()
                        consolidated[mask] = df.loc[mask, col]
                    
                    # Replace duplicate columns with consolidated one
                    df[base_col] = consolidated
                    
                    # Drop the old duplicate columns
                    cols_to_drop = [c for c in duplicate_cols if c != base_col]
                    df = df.drop(columns=cols_to_drop)
        
        return df
    
    # ============================================
    # MAIN UNIFICATION PIPELINE
    # ============================================
    
    def unify_all_clusters(self) -> Dict[int, str]:
        """
        Complete unification pipeline for all clusters
        
        Returns:
            Dictionary mapping cluster_id to output file path
        """
        print(Fore.CYAN + "\n" + "="*100)
        print("COMPLETE DATA UNIFICATION PIPELINE")
        print("="*100 + Style.RESET_ALL)
        
        clusters = self.clustering_data.get('clusters', [])
        unified_files = {}
        
        for cluster_id in range(1, len(clusters) + 1):
            print(Fore.YELLOW + f"\n▶️ PROCESSING CLUSTER {cluster_id}" + Style.RESET_ALL)
            print(Fore.WHITE + f"{'─'*40}" + Style.RESET_ALL)
            
            try:
                # STEP 3: Load and Transform
                transformed_dfs = self.load_and_transform_data(cluster_id)
                
                if not transformed_dfs:
                    print(Fore.RED + f"❌ No data loaded for Cluster {cluster_id}" + Style.RESET_ALL)
                    continue
                
                # STEP 6: Merge Records
                merged_df = self.merge_records(cluster_id, transformed_dfs)
                
                if merged_df.empty:
                    print(Fore.RED + f"❌ Merge failed for Cluster {cluster_id}" + Style.RESET_ALL)
                    continue
                
                # Save unified data
                output_path = self._save_unified_data(cluster_id, merged_df)
                
                # Create transformation log
                self._create_transformation_log(cluster_id, transformed_dfs, output_path)
                
                unified_files[cluster_id] = output_path
                
                print(Fore.GREEN + f"\n✅ Cluster {cluster_id} unified successfully!" + Style.RESET_ALL)
                
            except Exception as e:
                print(Fore.RED + f"\n❌ Error processing Cluster {cluster_id}: {e}" + Style.RESET_ALL)
                import traceback
                traceback.print_exc()
        
        return unified_files
    
    def _save_unified_data(self, cluster_id: int, merged_df: pd.DataFrame) -> str:
        """
        Save unified data to file
        
        Args:
            cluster_id: ID of the cluster
            merged_df: Merged DataFrame
            
        Returns:
            Path to saved file
        """
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"unified_cluster_{cluster_id}_{timestamp}.csv"
        output_path = self.output_dir / output_filename
        
        # Save to CSV
        merged_df.to_csv(output_path, index=False)
        
        print(f"💾 Saved unified data: {output_path}")
        print(f"   • Records: {len(merged_df):,}")
        print(f"   • Columns: {len(merged_df.columns)}")
        print(f"   • Size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return str(output_path)
    
    def _create_transformation_log(self, cluster_id: int, 
                                 transformed_dfs: List[pd.DataFrame], 
                                 output_path: str):
        """
        Create audit log of transformations
        """
        log = {
            'timestamp': datetime.now().isoformat(),
            'cluster_id': cluster_id,
            'input_files': [df['_source_file'].iloc[0] for df in transformed_dfs 
                          if '_source_file' in df.columns and len(df) > 0],
            'output_file': output_path,
            'transformation_summary': {
                'files_merged': len(transformed_dfs),
                'total_input_records': sum(len(df) for df in transformed_dfs),
                'output_records': len(transformed_dfs[0]) if transformed_dfs else 0,
                'output_columns': len(transformed_dfs[0].columns) if transformed_dfs else 0,
                'backup_available': True,
                'original_files_preserved': True
            },
            'column_statistics': self._get_column_stats(transformed_dfs)
        }
        
        log_path = self.backup_dir / f"transformation_log_cluster_{cluster_id}.json"
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2, default=str)
    
    def _get_column_stats(self, dataframes: List[pd.DataFrame]) -> Dict:
        """Get column statistics for log"""
        if not dataframes:
            return {}
        
        # Get golden columns from first DataFrame
        golden_cols = [
            col for col in dataframes[0].columns 
            if not col.startswith('_') and col != '_source_file' and col != '_cluster_id'
        ]
        
        stats = {}
        for col in golden_cols[:10]:  # Limit to first 10 columns for brevity
            # Collect data from all dataframes
            all_values = []
            for df in dataframes:
                if col in df.columns:
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        all_values.extend(non_null.head(10).tolist())  # Sample
            
            if all_values:
                stats[col] = {
                    'sample_values': all_values[:5],  # First 5 samples
                    'total_non_null': sum(1 for v in all_values if pd.notna(v)),
                    'value_types': list(set(type(v).__name__ for v in all_values[:10]))
                }
        
        return stats
    
    def generate_summary_report(self, unified_files: Dict[int, str]):
        """
        Generate summary report of unification process
        """
        print(Fore.CYAN + "\n" + "="*80)
        print("UNIFICATION SUMMARY REPORT")
        print("="*80 + Style.RESET_ALL)
        
        total_clusters = len(self.clustering_data.get('clusters', []))
        successful_clusters = len(unified_files)
        
        print(f"📊 Processing Summary:")
        print(f"   • Total clusters: {total_clusters}")
        print(f"   • Successfully unified: {successful_clusters}")
        print(f"   • Success rate: {successful_clusters/total_clusters*100:.1f}%")
        
        print(f"\n📁 Output Files:")
        for cluster_id, filepath in unified_files.items():
            filename = Path(filepath).name
            print(f"   • Cluster {cluster_id}: {filename}")
        
        print(f"\n📋 Directory Structure:")
        print(f"   • Original data: {self.data_dir}/ (untouched)")
        print(f"   • Unified data: {self.output_dir}/")
        print(f"   • Backup logs: {self.backup_dir}/")
        
        print(Fore.GREEN + "\n✅ DATA UNIFICATION COMPLETE!" + Style.RESET_ALL)
        print("   Original files are preserved.")
        print("   Unified data is ready for analysis.")


def run_safe_unification():
    """Main function to run safe data unification"""
    print(Fore.CYAN + """
    ╔══════════════════════════════════════════════════════════════╗
    ║              SAFE DATA UNIFICATION - STEPS 3 & 6            ║
    ╚══════════════════════════════════════════════════════════════╝
    """ + Style.RESET_ALL)
    
    print("This phase performs:")
    print("1. 📥 STEP 3: Data Loading & Transformation")
    print("2. 🔗 STEP 6: Record Merging (Horizontal/Vertical/Hybrid)")
    print("\n🔒 SAFETY GUARANTEE: Original files will NOT be modified")
    
    # # Ask for data directory
    # data_dir = input("\nEnter root data directory (default: 'data'): ").strip()
    # if not data_dir:
    #     data_dir = 'data'
    
    # if not os.path.exists(data_dir):
    #     print(Fore.RED + f"❌ Data directory '{data_dir}' not found!" + Style.RESET_ALL)
    #     return
    
    try:
        data_dir = "../../../Amazon Sales Dataset/raw data"
        # Initialize safe unifier
        unifier = SafeDataUnifier(
            data_dir=data_dir,
            output_dir='unified_output',
            backup_dir='unification_backups'
        )
        
        # Run complete unification pipeline
        unified_files = unifier.unify_all_clusters()
        
        # Generate summary
        unifier.generate_summary_report(unified_files)
        
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
    
    run_safe_unification()