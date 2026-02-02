import json, os
from typing import Dict, List, Any
import pandas as pd

def get_cluster_files(
    clustering_file: str = 'final_clustering_results.json'
) -> Dict[str, List[str]]:
    """
    Returns dictionary with cluster names as keys and file lists as values.
    
    Args:
        clustering_file: Path to clustering results JSON file
        
    Returns:
        Dictionary: {cluster_name: [file1, file2, ...]}
    """
    try:
        with open(clustering_file, 'r') as f:
            clustering_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {clustering_file} not found")
        return {}
    
    clusters = clustering_data.get('clusters', [])
    cluster_dict = {}
    
    for i, file_list in enumerate(clusters, 1):
        cluster_name = f"Cluster_{i}"
        cluster_dict[cluster_name] = file_list
    
    return cluster_dict


def get_golden_schemas(
    mapping_file: str = 'simplified_column_mappings.json'
) -> Dict[str, Dict[str, Any]]:
    """
    Returns golden schemas with column mapping information.
    
    Args:
        mapping_file: Path to column mappings JSON file
        
    Returns:
        Dictionary: {
            'Cluster_1': {
                'golden_schema': {
                    'name': 'schema_name',
                    'columns': ['col1', 'col2', ...]
                },
                'column_mappings': {
                    'golden_column_name': {
                        'mapped_columns': ['source_col1', 'source_col2', ...],
                        'consensus_type': 'string',
                        'confidence': 0.95
                    }
                }
            }
        }
    """
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {mapping_file} not found")
        return {}
    
    golden_schemas = {}
    
    for cluster_key, cluster_data in mapping_data.items():
        golden_schema = {
            'golden_schema': cluster_data.get('golden_schema', {}),
            'column_mappings': {}
        }
        
        # Extract column mappings from mapping matrix
        mapping_matrix = cluster_data.get('mapping_matrix', {})
        if mapping_matrix and 'mappings' in mapping_matrix:
            for mapping in mapping_matrix['mappings']:
                golden_column = mapping.get('golden_column', '')
                if golden_column:
                    # Get all mapped columns from schemas
                    mapped_columns = []
                    schema_mappings = mapping.get('schema_mappings', {})
                    
                    for schema_name, schema_data in schema_mappings.items():
                        if schema_data.get('mapped', False):
                            mapped_columns.append({
                                'schema': schema_name,
                                'column': schema_data.get('column', ''),
                                'type': schema_data.get('type', '')
                            })
                    
                    golden_schema['column_mappings'][golden_column] = {
                        'mapped_columns': mapped_columns,
                        'consensus_type': mapping.get('consensus_type', ''),
                        'confidence': mapping.get('confidence', 0)
                    }
        
        golden_schemas[cluster_key] = golden_schema
    
    return golden_schemas


def print_analysis_results(
    cluster_dict: Dict[str, List[str]],
    golden_schemas: Dict[str, Dict[str, Any]]
) -> None:
    """
    Prints the contents of both dictionaries in a readable format.
    
    Args:
        cluster_dict: Dictionary from get_cluster_files()
        golden_schemas: Dictionary from get_golden_schemas()
    """
    print("=" * 80)
    print("SCHEMA CLUSTERING AND GOLDEN SCHEMA ANALYSIS")
    print("=" * 80)
    
    # 1. Print cluster information
    print("\n📁 CLUSTER INFORMATION")
    print("-" * 40)
    
    for cluster_name, files in cluster_dict.items():
        print(f"\n{cluster_name} ({len(files)} files):")
        for i, file_name in enumerate(files, 1):
            print(f"  {i}. {file_name}")
    
    # 2. Print golden schema information
    print("\n⭐ GOLDEN SCHEMAS")
    print("-" * 40)
    
    for cluster_name, schema_info in golden_schemas.items():
        if cluster_name not in cluster_dict:
            continue
            
        print(f"\n{cluster_name}:")
        
        # Print golden schema basic info
        golden_schema = schema_info.get('golden_schema', {})
        print(f"  Golden Schema Name: {golden_schema.get('name', 'N/A')}")
        print(f"  Number of columns: {len(golden_schema.get('columns', []))}")
        
        # Print column details with mappings
        column_mappings = schema_info.get('column_mappings', {})
        if column_mappings:
            print(f"  Column Mappings:")
            for golden_col, mapping_info in column_mappings.items():
                mapped_count = len(mapping_info.get('mapped_columns', []))
                confidence = mapping_info.get('confidence', 0)
                col_type = mapping_info.get('consensus_type', 'unknown')
                
                print(f"    - {golden_col} ({col_type})")
                print(f"      Confidence: {confidence:.2f}")
                print(f"      Mapped in: {mapped_count} schemas")
                
                # Show first few mapped columns
                mapped_columns = mapping_info.get('mapped_columns', [])[:3]
                for mapped in mapped_columns:
                    print(f"        • {mapped['schema']}: {mapped['column']} ({mapped['type']})")
                
                total_mapped = len(mapping_info.get('mapped_columns', []))
                if total_mapped > 3:
                    print(f"        ... and {total_mapped - 3} more")
                print()


def run_analysis():
    """
    Main function to run the analysis.
    """
    print("Analyzing schema clustering and golden schemas...\n")
    
    # Get cluster information
    print("📂 Loading clustering data...")
    clusters = get_cluster_files()
    
    if not clusters:
        print("No clustering data found or error loading file.")
        return
    
    print(f"✅ Found {len(clusters)} clusters")
    
    # Get golden schema information
    print("\n⭐ Loading golden schema data...")
    schemas = get_golden_schemas()
    
    if not schemas:
        print("No golden schema data found or error loading file.")
        return
    
    print(f"✅ Found {len(schemas)} golden schemas")
    
    # Print results
    print_analysis_results(clusters, schemas)
    
    # Optional: Return data for further processing
    return clusters, schemas
def get_path(filename,data_dir):
        for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file == filename:
                        return os.path.join(root, file)
        return ""
def generate_cluster_csvs(
    clusters: Dict[str, List[str]],
    golden_schemas: Dict[str, Dict[str, Any]],
    data_directory: str = "../../../Amazon Sales Dataset/raw data",
    output_directory: str = "unified_data"
) -> Dict[str, str]:
    """
    Generate CSV files for each cluster with golden schema headers,
    populated with values from mapped columns.
    
    Args:
        clusters: Dictionary from get_cluster_files()
        golden_schemas: Dictionary from get_golden_schemas()
        data_directory: Directory containing the original data files
        output_directory: Directory to save the unified CSV files
        
    Returns:
        Dictionary mapping cluster names to their output file paths
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Dictionary to store output file paths
    output_files = {}
    
    print(f"\n📊 GENERATING UNIFIED CSV FILES")
    print("=" * 60)
    
    for cluster_name in clusters:
        if cluster_name not in golden_schemas:
            print(f"⚠️  Skipping {cluster_name}: No golden schema found")
            continue
        
        print(f"\nProcessing {cluster_name}...")
        
        # Get golden schema for this cluster
        schema_info = golden_schemas[cluster_name]
        golden_schema = schema_info.get('golden_schema', {})
        column_mappings = schema_info.get('column_mappings', {})
        
        if not golden_schema or not column_mappings:
            print(f"  ⚠️  No column mappings found for {cluster_name}")
            continue
        
        # Get golden column names (headers for CSV)
        golden_columns = golden_schema.get('columns', [])
        if not golden_columns:
            golden_columns = list(column_mappings.keys())
        
        # Create output filename
        output_filename = f"{cluster_name.lower()}_unified.csv"
        output_path = os.path.join(output_directory, output_filename)
        output_files[cluster_name] = output_path
        
        # Prepare to collect data
        all_rows = []
        
        # Process each file in the cluster
        for file_name in clusters[cluster_name]:
            print(f"  📁 Reading: {file_name}")
            
            # Load the data from the file
            # file_path = os.path.join(data_directory, file_name)
            file_path = get_path(file_name,data_directory)
            
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_name.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        # Handle single object or nested structure
                        # Flatten if needed
                        df = pd.json_normalize(data)  
                    # df = pd.read_json(file_path)
                else:
                    print(f"    ⚠️  Unsupported file format: {file_name}")
                    continue
            except Exception as e:
                print(f"    ❌ Error reading {file_name}: {str(e)}")
                continue
            
            # For each row in the dataframe
            for _, row in df.iterrows():
                unified_row = {}
                
                # For each golden column, find the mapped source column value
                for golden_col in golden_columns:
                    if golden_col not in column_mappings:
                        unified_row[golden_col] = None
                        continue
                    
                    mapping_info = column_mappings[golden_col]
                    mapped_columns = mapping_info.get('mapped_columns', [])
                    
                    # Find the source column for this specific file
                    source_value = None
                    for mapped_col in mapped_columns:
                        if mapped_col['schema'] == file_name:
                            source_col_name = mapped_col['column']
                            if source_col_name in df.columns:
                                source_value = row[source_col_name]
                            break
                    
                    unified_row[golden_col] = source_value
                
                all_rows.append(unified_row)
        
        # Write to CSV
        if all_rows:
            # Create DataFrame with golden columns as headers
            unified_df = pd.DataFrame(all_rows, columns=golden_columns)
            
            # Write to CSV
            unified_df.to_csv(output_path, index=False)
            
            # Print statistics
            print(f"  ✅ Created: {output_filename}")
            print(f"     Rows: {len(unified_df)}")
            print(f"     Columns: {len(golden_columns)}")
            print(f"     File: {output_path}")
        else:
            print(f"  ⚠️  No data found for {cluster_name}")
    
    print(f"\n✅ All files generated in '{output_directory}' directory")
    
    return output_files


def unify(directory):
    clusters, golden_schemas = run_analysis()
    output_files = generate_cluster_csvs(
    clusters=clusters,
    golden_schemas=golden_schemas,
    # data_directory="../../Amazon Sales Dataset/raw data",  # Your data folder
    data_directory = directory,
    output_directory="unified_data"      # Output folder
    )

    # Check output
    for cluster, filepath in output_files.items():
        print(f"{cluster}: {filepath}")
if __name__ == "__main__":
    # Simple usage
    clusters, golden_schemas = run_analysis()
    
    # Example of accessing the data programmatically:
    # print("\nExample: Accessing Cluster_1 files:")
    # print(clusters.get("Cluster_1", []))
    
    # print("\nExample: Accessing golden schema for Cluster_1:")
    # print(golden_schemas.get("Cluster_1", {}))

    output_files = generate_cluster_csvs(
    clusters=clusters,
    golden_schemas=golden_schemas,
    data_directory="../../../Amazon Sales Dataset/raw data",  # Your data folder
    output_directory="unified_data"      # Output folder
    )

    # Check output
    for cluster, filepath in output_files.items():
        print(f"{cluster}: {filepath}")