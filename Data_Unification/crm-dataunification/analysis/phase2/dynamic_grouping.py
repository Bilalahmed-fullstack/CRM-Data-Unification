import numpy as np
from collections import defaultdict

def calculate_column_similarity(columns1, columns2):
    """Calculate similarity between two sets of column names."""
    if not columns1 or not columns2:
        return 0
    
    cols1_set = {str(col).lower().strip() for col in columns1}
    cols2_set = {str(col).lower().strip() for col in columns2}
    
    intersection = cols1_set.intersection(cols2_set)
    union = cols1_set.union(cols2_set)
    
    if not union:
        return 0
    
    return len(intersection) / len(union)

def create_similarity_matrix(file_columns_dict):
    """Create a similarity matrix for all files based on column names."""
    file_names = list(file_columns_dict.keys())
    n = len(file_names)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                sim = calculate_column_similarity(
                    file_columns_dict[file_names[i]],
                    file_columns_dict[file_names[j]]
                )
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
    
    return file_names, similarity_matrix

def hierarchical_cluster_files(similarity_matrix, threshold=0.3):
    """Cluster files using hierarchical clustering based on similarity matrix."""
    n = similarity_matrix.shape[0]
    clusters = [[i] for i in range(n)]  # Start with each file as its own cluster
    distances = similarity_matrix.copy()
    
    # Perform hierarchical clustering
    while len(clusters) > 1:
        # Find closest clusters
        max_sim = -1
        merge_i = merge_j = -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Calculate average similarity between clusters
                sim_sum = 0
                count = 0
                for idx_i in clusters[i]:
                    for idx_j in clusters[j]:
                        sim_sum += similarity_matrix[idx_i][idx_j]
                        count += 1
                avg_sim = sim_sum / count if count > 0 else 0
                
                if avg_sim > max_sim:
                    max_sim = avg_sim
                    merge_i, merge_j = i, j
        
        # Stop if no clusters are similar enough
        if max_sim < threshold:
            break
        
        # Merge clusters
        clusters[merge_i].extend(clusters[merge_j])
        clusters.pop(merge_j)
    
    return clusters

def cluster_based_on_columns(file_columns_dict, similarity_threshold=0.3):
    """
    Cluster files based on actual column similarity.
    This is the primary grouping method.
    """
    file_names, similarity_matrix = create_similarity_matrix(file_columns_dict)
    clusters = hierarchical_cluster_files(similarity_matrix, similarity_threshold)
    
    # Convert cluster indices to file names
    column_clusters = []
    for cluster_indices in clusters:
        cluster_files = [file_names[idx] for idx in cluster_indices]
        column_clusters.append(sorted(cluster_files))
    
    return column_clusters, similarity_matrix, file_names

def extract_prefix_suffix(filename):
    """Extract prefix and suffix from filename."""
    if '.' in filename:
        name_without_ext = filename.rsplit('.', 1)[0]
    else:
        name_without_ext = filename
    
    parts = name_without_ext.replace('_', ' ').replace('-', ' ').replace('.', ' ').split()
    
    prefix = parts[0] if parts else None
    suffix = parts[-1] if len(parts) > 1 else None
    
    return prefix, suffix

def compare_with_name_based_groupings(column_clusters, file_columns_dict):
    """
    Compare column-based clusters with prefix/suffix based groupings.
    """
    # Create prefix and suffix groupings for comparison
    file_names = list(file_columns_dict.keys())
    
    # Prefix grouping
    prefix_groups = defaultdict(list)
    for filename in file_names:
        prefix, _ = extract_prefix_suffix(filename)
        if prefix:
            prefix_groups[prefix].append(filename)
    prefix_clusters = [sorted(files) for files in prefix_groups.values() if len(files) >= 2]
    
    # Suffix grouping
    suffix_groups = defaultdict(list)
    for filename in file_names:
        _, suffix = extract_prefix_suffix(filename)
        if suffix:
            suffix_groups[suffix].append(filename)
    suffix_clusters = [sorted(files) for files in suffix_groups.values() if len(files) >= 2]
    
    # Calculate alignment scores
    def calculate_alignment(column_clusters, name_clusters):
        """Calculate how well column clusters align with name clusters."""
        if not column_clusters or not name_clusters:
            return 0
        
        # Convert to sets for easier comparison
        column_sets = [set(cluster) for cluster in column_clusters]
        name_sets = [set(cluster) for cluster in name_clusters]
        
        # Calculate alignment score
        total_alignment = 0
        for col_set in column_sets:
            best_alignment = 0
            for name_set in name_sets:
                # Jaccard similarity between clusters
                intersection = len(col_set.intersection(name_set))
                union = len(col_set.union(name_set))
                if union > 0:
                    alignment = intersection / union
                    best_alignment = max(best_alignment, alignment)
            total_alignment += best_alignment
        
        return total_alignment / len(column_clusters)
    
    prefix_alignment = calculate_alignment(column_clusters, prefix_clusters)
    suffix_alignment = calculate_alignment(column_clusters, suffix_clusters)
    
    return {
        'column_clusters': column_clusters,
        'prefix_clusters': prefix_clusters,
        'suffix_clusters': suffix_clusters,
        'prefix_alignment': prefix_alignment,
        'suffix_alignment': suffix_alignment,
        'recommended_based_on_alignment': 'prefix' if prefix_alignment > suffix_alignment else 'suffix'
    }

def analyze_and_adjust_groupings(file_columns_dict, similarity_threshold=0.3):
    """
    Main function: First cluster by columns, then compare with name patterns.
    Returns final adjusted groupings.
    """
    # Step 1: Cluster files based on column similarity
    column_clusters, similarity_matrix, file_names = cluster_based_on_columns(
        file_columns_dict, similarity_threshold
    )
    
    # Step 2: Compare with name-based groupings
    comparison = compare_with_name_based_groupings(column_clusters, file_columns_dict)
    
    # Step 3: Create final groupings based on evidence
    final_groupings = []
    
    if comparison['prefix_alignment'] > 0.7 or comparison['suffix_alignment'] > 0.7:
        # High alignment with name patterns, use column clusters as is
        final_groupings = comparison['column_clusters']
        grouping_type = 'column_based_with_high_name_alignment'
    else:
        # Poor alignment, need to create adjusted groupings
        # Merge small clusters that share name patterns
        final_groupings = merge_clusters_by_name_patterns(
            comparison['column_clusters'], file_columns_dict
        )
        grouping_type = 'adjusted_based_on_mixed_evidence'
    
    # Step 4: Analyze cluster quality
    cluster_quality = analyze_cluster_quality(final_groupings, file_columns_dict)
    
    return {
        'final_groupings': final_groupings,
        'column_clusters': comparison['column_clusters'],
        'prefix_clusters': comparison['prefix_clusters'],
        'suffix_clusters': comparison['suffix_clusters'],
        'alignment_scores': {
            'prefix': comparison['prefix_alignment'],
            'suffix': comparison['suffix_alignment']
        },
        'grouping_type': grouping_type,
        'cluster_quality': cluster_quality,
        'similarity_matrix': similarity_matrix,
        'file_names': file_names
    }

def merge_clusters_by_name_patterns(column_clusters, file_columns_dict):
    """Merge clusters that share strong name patterns."""
    # Convert to list of sets
    cluster_sets = [set(cluster) for cluster in column_clusters]
    merged = True
    
    while merged:
        merged = False
        new_clusters = []
        used = [False] * len(cluster_sets)
        
        for i in range(len(cluster_sets)):
            if used[i]:
                continue
            
            current = cluster_sets[i].copy()
            used[i] = True
            
            # Check if should merge with others
            for j in range(i + 1, len(cluster_sets)):
                if used[j]:
                    continue
                
                # Check name pattern similarity between clusters
                if should_merge_clusters(current, cluster_sets[j], file_columns_dict):
                    current.update(cluster_sets[j])
                    used[j] = True
                    merged = True
            
            new_clusters.append(sorted(list(current)))
        
        cluster_sets = [set(cluster) for cluster in new_clusters]
    
    return new_clusters

def should_merge_clusters(cluster1, cluster2, file_columns_dict):
    """Determine if two clusters should be merged based on name patterns."""
    # Check if clusters share common name patterns
    prefixes1 = {extract_prefix_suffix(f)[0] for f in cluster1}
    prefixes2 = {extract_prefix_suffix(f)[0] for f in cluster2}
    
    suffixes1 = {extract_prefix_suffix(f)[1] for f in cluster1}
    suffixes2 = {extract_prefix_suffix(f)[1] for f in cluster2}
    
    # Check for common prefixes or suffixes
    common_prefixes = prefixes1.intersection(prefixes2)
    common_suffixes = suffixes1.intersection(suffixes2)
    
    # Also check column similarity between clusters
    avg_similarity = 0
    count = 0
    for file1 in cluster1:
        for file2 in cluster2:
            avg_similarity += calculate_column_similarity(
                file_columns_dict[file1],
                file_columns_dict[file2]
            )
            count += 1
    
    if count > 0:
        avg_similarity /= count
    
    # Merge if they share name patterns OR have decent column similarity
    return len(common_prefixes) > 0 or len(common_suffixes) > 0 or avg_similarity > 0.2

def analyze_cluster_quality(groupings, file_columns_dict):
    """Analyze the quality of the final groupings."""
    quality_scores = []
    
    for group in groupings:
        if len(group) <= 1:
            continue
        
        # Calculate intra-cluster similarity
        column_sets = [file_columns_dict[filename] for filename in group]
        similarities = []
        
        for i in range(len(column_sets)):
            for j in range(i + 1, len(column_sets)):
                sim = calculate_column_similarity(column_sets[i], column_sets[j])
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        quality_scores.append({
            'group': group,
            'avg_similarity': avg_similarity,
            'size': len(group),
            'quality': 'High' if avg_similarity > 0.6 else 'Medium' if avg_similarity > 0.3 else 'Low'
        })
    
    overall_avg = sum(s['avg_similarity'] for s in quality_scores) / len(quality_scores) if quality_scores else 0
    
    return {
        'per_group_scores': quality_scores,
        'overall_avg_similarity': overall_avg,
        'total_groups': len(groupings),
        'total_files': sum(len(g) for g in groupings)
    }

def print_final_analysis(results):
    """Print comprehensive analysis results."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE DATASET GROUPING ANALYSIS")
    print("=" * 100)
    
    print("\n📊 COLUMN-BASED CLUSTERS (Primary Analysis):")
    print("-" * 100)
    for i, cluster in enumerate(results['column_clusters']):
        print(f"\nCluster {i+1} ({len(cluster)} files):")
        print(f"  {cluster}")
    
    print("\n📊 NAME-BASED GROUPINGS FOR COMPARISON:")
    print("-" * 100)
    print("\nPrefix-based groups:")
    for i, cluster in enumerate(results['prefix_clusters']):
        print(f"  Group {i+1}: {cluster}")
    
    print("\nSuffix-based groups:")
    for i, cluster in enumerate(results['suffix_clusters']):
        print(f"  Group {i+1}: {cluster}")
    
    print("\n📈 ALIGNMENT ANALYSIS:")
    print("-" * 100)
    print(f"\nColumn clusters vs Prefix groups: {results['alignment_scores']['prefix']:.2%}")
    print(f"Column clusters vs Suffix groups: {results['alignment_scores']['suffix']:.2%}")
    
    if results['alignment_scores']['prefix'] > 0.7 or results['alignment_scores']['suffix'] > 0.7:
        print("\n✅ Good alignment with name patterns detected!")
        if results['alignment_scores']['prefix'] > results['alignment_scores']['suffix']:
            print("   Files are primarily grouped by SYSTEM/SOURCE prefixes")
        else:
            print("   Files are primarily grouped by DOMAIN/ENTITY suffixes")
    else:
        print("\n⚠️  Poor alignment with name patterns.")
        print("   Files may have unusual naming or mixed schema patterns")
    
    print("\n✅ FINAL ADJUSTED GROUPINGS:")
    print("-" * 100)
    quality = results['cluster_quality']
    for i, group in enumerate(results['final_groupings']):
        group_score = next((s for s in quality['per_group_scores'] if s['group'] == group), None)
        if group_score:
            print(f"\nGroup {i+1} ({group_score['quality']} quality, {group_score['avg_similarity']:.2%} similarity):")
        else:
            print(f"\nGroup {i+1}:")
        print(f"  {sorted(group)}")
    
    print("\n📊 CLUSTER QUALITY SUMMARY:")
    print("-" * 100)
    print(f"Overall average similarity: {quality['overall_avg_similarity']:.2%}")
    print(f"Total groups: {quality['total_groups']}")
    print(f"Total files: {quality['total_files']}")
    
    print(f"\n📋 GROUPING TYPE: {results['grouping_type'].upper()}")
    print("=" * 100)

# Example usage:
# results = analyze_and_adjust_groupings(file_columns_dict, similarity_threshold=0.3)
# print_final_analysis(results)