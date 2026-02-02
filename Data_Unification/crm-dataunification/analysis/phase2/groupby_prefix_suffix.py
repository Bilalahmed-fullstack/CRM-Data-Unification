
from collections import defaultdict

def extract_prefix_suffix(filename):
    """
    Extract prefix and suffix from a filename.
    Prefix: first part before separator
    Suffix: last part before extension
    """
    # Remove file extension
    if '.' in filename:
        name_without_ext = filename.rsplit('.', 1)[0]
    else:
        name_without_ext = filename
    
    # Split by common separators
    parts = name_without_ext.replace('_', ' ').replace('-', ' ').replace('.', ' ').split()
    
    prefix = parts[0] if parts else None
    suffix = parts[-1] if len(parts) > 1 else None
    
    return prefix, suffix

def group_by_prefix(file_names):
    """Group files by common prefix."""
    prefix_groups = defaultdict(list)
    
    for filename in file_names:
        prefix, _ = extract_prefix_suffix(filename)
        if prefix:
            prefix_groups[prefix].append(filename)
    
    # Return only groups with at least 2 files
    return [sorted(files) for prefix, files in prefix_groups.items() if len(files) >= 2]

def group_by_suffix(file_names):
    """Group files by common suffix."""
    suffix_groups = defaultdict(list)
    
    for filename in file_names:
        _, suffix = extract_prefix_suffix(filename)
        if suffix:
            suffix_groups[suffix].append(filename)
    
    # Return only groups with at least 2 files
    return [sorted(files) for suffix, files in suffix_groups.items() if len(files) >= 2]

def group_files_by_prefix_suffix(file_columns_dict):
    """
    Create two separate groupings: one by prefix, one by suffix.
    Returns a dictionary with both groupings.
    """
    file_names = list(file_columns_dict.keys())
    
    prefix_groups = group_by_prefix(file_names)
    suffix_groups = group_by_suffix(file_names)
    
    return {
        'by_prefix': prefix_groups,
        'by_suffix': suffix_groups
    }
def print_grouping(groupings):
    # # Print the results
    print("Prefix-based Grouping:")
    print("=" * 60)
    for i, group in enumerate(groupings['by_prefix']):
        print(f"Group {i+1} (Prefix: {extract_prefix_suffix(group[0])[0]}): {group}")

    print("\nSuffix-based Grouping:")
    print("=" * 60)
    for i, group in enumerate(groupings['by_suffix']):
        print(f"Group {i+1} (Suffix: {extract_prefix_suffix(group[0])[1]}): {group}")
    print("=" * 60)