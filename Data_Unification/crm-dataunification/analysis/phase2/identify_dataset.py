from collections import defaultdict
import math

def calculate_column_similarity(columns1, columns2):
    """
    Calculate similarity between two sets of column names.
    Returns a score between 0 and 1.
    """
    if not columns1 or not columns2:
        return 0
    
    # Convert to lowercase for comparison
    cols1_set = {str(col).lower().strip() for col in columns1}
    cols2_set = {str(col).lower().strip() for col in columns2}
    
    # Calculate Jaccard similarity
    intersection = cols1_set.intersection(cols2_set)
    union = cols1_set.union(cols2_set)
    
    if not union:
        return 0
    
    return len(intersection) / len(union)

def calculate_column_naming_consistency(columns_list):
    """
    Calculate how consistent column names are across multiple files.
    """
    if len(columns_list) <= 1:
        return 1.0
    
    # Calculate all pairwise similarities
    similarities = []
    for i in range(len(columns_list)):
        for j in range(i + 1, len(columns_list)):
            sim = calculate_column_similarity(columns_list[i], columns_list[j])
            similarities.append(sim)
    
    if not similarities:
        return 0
    
    return sum(similarities) / len(similarities)

def analyze_group_structure(grouping_result, file_columns_dict):
    """
    Analyze which grouping (prefix or suffix) better represents dataset domains.
    
    Args:
        grouping_result: Dictionary from group_files_by_prefix_suffix()
        file_columns_dict: Original dictionary with file->columns mapping
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'prefix_groups_analysis': [],
        'suffix_groups_analysis': [],
        'summary': {}
    }
    
    # Analyze prefix-based groups
    for i, group in enumerate(grouping_result['by_prefix']):
        # Get column sets for all files in this group
        column_sets = [file_columns_dict[filename] for filename in group]
        
        # Calculate metrics
        consistency_score = calculate_column_naming_consistency(column_sets)
        
        # Calculate column overlap (common columns across all files)
        column_sets_lower = [{str(col).lower().strip() for col in cols} for cols in column_sets]
        common_columns = set.intersection(*column_sets_lower) if column_sets_lower else set()
        
        avg_columns = sum(len(cols) for cols in column_sets) / len(column_sets)
        
        group_info = {
            'group_number': i + 1,
            'files': group,
            'column_consistency': consistency_score,
            'common_columns': list(common_columns),
            'common_columns_count': len(common_columns),
            'avg_column_count': avg_columns,
            'group_type': 'prefix',
            'pattern': 'Prefix: ' + group[0].split('_')[0] if '_' in group[0] else group[0]
        }
        analysis['prefix_groups_analysis'].append(group_info)
    
    # Analyze suffix-based groups
    for i, group in enumerate(grouping_result['by_suffix']):
        # Get column sets for all files in this group
        column_sets = [file_columns_dict[filename] for filename in group]
        
        # Calculate metrics
        consistency_score = calculate_column_naming_consistency(column_sets)
        
        # Calculate column overlap
        column_sets_lower = [{str(col).lower().strip() for col in cols} for cols in column_sets]
        common_columns = set.intersection(*column_sets_lower) if column_sets_lower else set()
        
        avg_columns = sum(len(cols) for cols in column_sets) / len(column_sets)
        
        # Extract suffix pattern
        suffix = None
        for file in group:
            parts = file.replace('.csv', '').replace('.json', '').split('_')
            if len(parts) > 1:
                suffix = parts[-1]
                break
        
        group_info = {
            'group_number': i + 1,
            'files': group,
            'column_consistency': consistency_score,
            'common_columns': list(common_columns),
            'common_columns_count': len(common_columns),
            'avg_column_count': avg_columns,
            'group_type': 'suffix',
            'pattern': 'Suffix: ' + suffix if suffix else 'Unknown'
        }
        analysis['suffix_groups_analysis'].append(group_info)
    
    # Calculate summary statistics
    prefix_avg_consistency = sum(g['column_consistency'] for g in analysis['prefix_groups_analysis']) / max(len(analysis['prefix_groups_analysis']), 1)
    suffix_avg_consistency = sum(g['column_consistency'] for g in analysis['suffix_groups_analysis']) / max(len(analysis['suffix_groups_analysis']), 1)
    
    prefix_avg_common = sum(g['common_columns_count'] for g in analysis['prefix_groups_analysis']) / max(len(analysis['prefix_groups_analysis']), 1)
    suffix_avg_common = sum(g['common_columns_count'] for g in analysis['suffix_groups_analysis']) / max(len(analysis['suffix_groups_analysis']), 1)
    
    analysis['summary'] = {
        'prefix_avg_consistency': prefix_avg_consistency,
        'suffix_avg_consistency': suffix_avg_consistency,
        'prefix_avg_common_columns': prefix_avg_common,
        'suffix_avg_common_columns': suffix_avg_common,
        'recommended_grouping': 'prefix' if prefix_avg_consistency > suffix_avg_consistency else 'suffix',
        'recommendation_reason': 'Higher column consistency' if prefix_avg_consistency > suffix_avg_consistency else 'Higher common columns'
    }
    
    return analysis

def print_analysis_results(analysis):
    """Print the analysis results in a readable format."""
    
    print("\n" + "=" * 80)
    print("DATASET DOMAIN ANALYSIS BASED ON COLUMN SIMILARITY")
    print("=" * 80)
    
    print("\n📊 PREFIX-BASED GROUPS ANALYSIS:")
    print("-" * 80)
    for group in analysis['prefix_groups_analysis']:
        print(f"\nGroup {group['group_number']} ({group['pattern']}):")
        print(f"  Files: {group['files']}")
        print(f"  Column Consistency Score: {group['column_consistency']:.2%}")
        print(f"  Common Columns: {group['common_columns_count']} columns")
        if group['common_columns']:
            print(f"  Common Columns List: {', '.join(group['common_columns'][:5])}")
            if len(group['common_columns']) > 5:
                print(f"    ... and {len(group['common_columns']) - 5} more")
    
    print("\n📊 SUFFIX-BASED GROUPS ANALYSIS:")
    print("-" * 80)
    for group in analysis['suffix_groups_analysis']:
        print(f"\nGroup {group['group_number']} ({group['pattern']}):")
        print(f"  Files: {group['files']}")
        print(f"  Column Consistency Score: {group['column_consistency']:.2%}")
        print(f"  Common Columns: {group['common_columns_count']} columns")
        if group['common_columns']:
            print(f"  Common Columns List: {', '.join(group['common_columns'][:5])}")
            if len(group['common_columns']) > 5:
                print(f"    ... and {len(group['common_columns']) - 5} more")
    
    print("\n" + "=" * 80)
    print("📈 SUMMARY & RECOMMENDATION:")
    print("-" * 80)
    summary = analysis['summary']
    print(f"\nPrefix-based grouping:")
    print(f"  Average Column Consistency: {summary['prefix_avg_consistency']:.2%}")
    print(f"  Average Common Columns: {summary['prefix_avg_common_columns']:.1f}")
    
    print(f"\nSuffix-based grouping:")
    print(f"  Average Column Consistency: {summary['suffix_avg_consistency']:.2%}")
    print(f"  Average Common Columns: {summary['suffix_avg_common_columns']:.1f}")
    
    print(f"\n✅ RECOMMENDATION: Use {summary['recommended_grouping'].upper()}-based grouping")
    print(f"   Reason: {summary['recommendation_reason']}")
    
    # Additional insight
    if summary['prefix_avg_consistency'] > 0.7 and summary['suffix_avg_consistency'] > 0.7:
        print("\n💡 INSIGHT: Both groupings show good column consistency.")
        print("   This suggests files are well-structured across both dimensions.")
    elif summary['prefix_avg_consistency'] > summary['suffix_avg_consistency']:
        print("\n💡 INSIGHT: Prefix grouping represents different SYSTEMS/SOURCES")
        print("   (e.g., modern, legacy, third_system implementations)")
    else:
        print("\n💡 INSIGHT: Suffix grouping represents different DOMAINS/ENTITIES")
        print("   (e.g., customers, products, sales datasets)")
    
    print("=" * 80)

# Example usage:
# groupings = group_files_by_prefix_suffix(file_columns_dict)
# analysis = analyze_group_structure(groupings, file_columns_dict)
# print_analysis_results(analysis)