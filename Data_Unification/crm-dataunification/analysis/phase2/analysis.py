from get_directory_structure import get_file_columns, print_structure

directory = '../../Amazon Sales Dataset/raw data'
# Example usage
file_columns_dict = get_file_columns(directory)
print_structure(file_columns_dict)

from groupby_prefix_suffix import group_files_by_prefix_suffix, extract_prefix_suffix, print_grouping
# Example usage with file_columns_dict from previous steps
groupings = group_files_by_prefix_suffix(file_columns_dict)
print_grouping(groupings)

from identify_dataset import analyze_group_structure, print_analysis_results
analysis = analyze_group_structure(groupings, file_columns_dict)
if analysis["summary"]["recommended_grouping"] == "prefix":
    group = groupings['by_prefix']
else:
    groupings['by_suffix']
print_analysis_results(analysis)