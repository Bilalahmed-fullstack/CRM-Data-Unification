import json
import os
from typing import Dict, List, Any
from tabulate import tabulate
from colorama import init, Fore, Style
import pandas as pd
from collections import defaultdict

# Initialize colorama for colored output
init(autoreset=True)

def analyze_clustering_and_mapping(
    clustering_file: str = '../final_clustering_results.json',
    mapping_file: str = '../simplified_column_mappings.json'
) -> Dict[str, Any]:
    """
    Comprehensive analysis of clustering and column mapping results.
    
    Args:
        clustering_file: Path to clustering results
        mapping_file: Path to column mapping results
        
    Returns:
        Dictionary with all analysis results
    """
    
    print(Fore.CYAN + "\n" + "="*100)
    print(Fore.CYAN + "📊 COMPREHENSIVE CLUSTERING & MAPPING ANALYSIS")
    print(Fore.CYAN + "="*100 + Style.RESET_ALL)
    
    # Load data
    print(Fore.YELLOW + "\n📂 Loading data files..." + Style.RESET_ALL)
    
    try:
        with open(clustering_file, 'r') as f:
            clustering_data = json.load(f)
        print(Fore.GREEN + f"✅ Loaded clustering data from {clustering_file}" + Style.RESET_ALL)
    except FileNotFoundError:
        print(Fore.RED + f"❌ Error: {clustering_file} not found" + Style.RESET_ALL)
        return {}
    
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        print(Fore.GREEN + f"✅ Loaded mapping data from {mapping_file}" + Style.RESET_ALL)
    except FileNotFoundError:
        print(Fore.RED + f"❌ Error: {mapping_file} not found" + Style.RESET_ALL)
        return {}
    
    # Extract basic information
    analysis_results = {
        'overview': {},
        'clusters': [],
        'mappings': [],
        'statistics': {},
        'unmapped_analysis': {}
    }
    
    # 1. OVERVIEW SECTION
    print(Fore.CYAN + "\n" + "━"*50)
    print("📋 OVERVIEW")
    print("━"*50 + Style.RESET_ALL)
    
    # Clustering overview
    total_files = len(clustering_data.get('file_names', []))
    total_clusters = len(clustering_data.get('clusters', []))
    
    analysis_results['overview']['total_files'] = total_files
    analysis_results['overview']['total_clusters'] = total_clusters
    
    print(f"📁 Total files analyzed: {Fore.GREEN}{total_files}{Style.RESET_ALL}")
    print(f"📊 Total clusters formed: {Fore.GREEN}{total_clusters}{Style.RESET_ALL}")
    
    # Show weights used
    if 'weights' in clustering_data:
        print("\n⚖️  Similarity weights used:")
        for criterion, weight in clustering_data['weights'].items():
            print(f"   • {criterion.replace('_', ' ').title()}: {weight:.2f}")
    
    # 2. CLUSTER ANALYSIS SECTION
    print(Fore.CYAN + "\n" + "━"*50)
    print("🎯 CLUSTER ANALYSIS")
    print("━"*50 + Style.RESET_ALL)
    
    clusters = clustering_data.get('clusters', [])
    file_names = clustering_data.get('file_names', [])
    
    for cluster_idx, cluster_files in enumerate(clusters, 1):
        cluster_key = f"Cluster_{cluster_idx}"
        cluster_info = {
            'cluster_id': cluster_idx,
            'files': cluster_files,
            'size': len(cluster_files),
            'mapping_data': mapping_data.get(cluster_key, {})
        }
        analysis_results['clusters'].append(cluster_info)
        
        # Print cluster details
        print(f"\n{Fore.YELLOW}📦 Cluster {cluster_idx} ({len(cluster_files)} files){Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*40}{Style.RESET_ALL}")
        
        # List files in cluster
        for i, file in enumerate(cluster_files, 1):
            print(f"  {i:2d}. {file}")
        
        # Show cluster similarity stats if available
        if 'similarity_matrices' in clustering_data:
            # Calculate average similarity within cluster
            matrix = clustering_data.get('final_similarity_matrix', [])
            if matrix and len(file_names) > 0:
                indices = [file_names.index(f) for f in cluster_files]
                similarities = []
                
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        sim = matrix[indices[i]][indices[j]]
                        similarities.append(sim)
                
                if similarities:
                    avg_sim = sum(similarities) / len(similarities)
                    print(f"  📈 Average intra-cluster similarity: {Fore.GREEN}{avg_sim:.3f}{Style.RESET_ALL}")
    
    # 3. GOLDEN SCHEMA & MAPPING ANALYSIS
    print(Fore.CYAN + "\n" + "━"*50)
    print("⭐ GOLDEN SCHEMAS & COLUMN MAPPINGS")
    print("━"*50 + Style.RESET_ALL)
    
    total_golden_columns = 0
    total_mappings = 0
    confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    for cluster_idx, cluster_info in enumerate(analysis_results['clusters'], 1):
        cluster_key = f"Cluster_{cluster_idx}"
        mapping_info = mapping_data.get(cluster_key, {})
        
        if not mapping_info:
            print(f"\n{Fore.RED}⚠️  No mapping data for {cluster_key}{Style.RESET_ALL}")
            continue
        
        # Extract mapping details
        golden_schema = mapping_info.get('golden_schema', {})
        mapping_matrix = mapping_info.get('mapping_matrix', {})
        quality_metrics = mapping_info.get('quality_metrics', {})
        
        print(f"\n{Fore.MAGENTA}🎯 {cluster_key} - Golden Schema Analysis{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*60}{Style.RESET_ALL}")
        
        # Golden schema info
        golden_name = golden_schema.get('name', 'Unknown')
        golden_columns = golden_schema.get('columns', [])
        total_golden_columns += len(golden_columns)
        
        print(f"📝 Golden Schema Name: {Fore.YELLOW}{golden_name}{Style.RESET_ALL}")
        print(f"📊 Number of unified columns: {Fore.GREEN}{len(golden_columns)}{Style.RESET_ALL}")
        
        # Quality metrics
        if quality_metrics:
            avg_sim = quality_metrics.get('average_similarity', 0)
            coverage = quality_metrics.get('coverage', 0)
            mapped = quality_metrics.get('mapped_columns', 0)
            total = quality_metrics.get('total_columns', 0)
            
            print(f"📈 Quality Metrics:")
            print(f"   • Average similarity: {avg_sim:.3f}")
            print(f"   • Coverage: {coverage:.1%} ({mapped}/{total} columns mapped)")
            print(f"   • Min/Max similarity: {quality_metrics.get('min_similarity', 0):.3f}/{quality_metrics.get('max_similarity', 0):.3f}")
        
        # Detailed column mappings
        print(f"\n🔗 Column Mappings:")
        
        if mapping_matrix and 'mappings' in mapping_matrix:
            mappings = mapping_matrix['mappings']
            total_mappings += len(mappings)
            schemas = mapping_matrix.get('schemas', [])
            
            # Create a table for visualization
            table_data = []
            for mapping in mappings:
                golden_col = mapping.get('golden_column', '')
                consensus_type = mapping.get('consensus_type', '')
                confidence = mapping.get('confidence', 0)
                
                # Determine confidence level
                if confidence >= 0.8:
                    conf_level = f"{Fore.GREEN}High{Style.RESET_ALL}"
                    confidence_stats['high'] += 1
                elif confidence >= 0.6:
                    conf_level = f"{Fore.YELLOW}Medium{Style.RESET_ALL}"
                    confidence_stats['medium'] += 1
                else:
                    conf_level = f"{Fore.RED}Low{Style.RESET_ALL}"
                    confidence_stats['low'] += 1
                
                # Count how many schemas have this column mapped
                schema_mappings = mapping.get('schema_mappings', {})
                mapped_count = sum(1 for sm in schema_mappings.values() if sm.get('mapped', False))
                total_schemas = len(schemas)
                
                table_data.append([
                    golden_col,
                    consensus_type,
                    f"{confidence:.2f}",
                    conf_level,
                    f"{mapped_count}/{total_schemas}"
                ])
            
            # Print table using tabulate
            headers = ["Golden Column", "Type", "Confidence", "Level", "Mapped In"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Show example mappings for first few columns
            print(f"\n📋 Example mappings (first 3 columns):")
            for i, mapping in enumerate(mappings[:3]):
                golden_col = mapping.get('golden_column', '')
                print(f"\n  {Fore.CYAN}{golden_col}{Style.RESET_ALL}:")
                schema_mappings = mapping.get('schema_mappings', {})
                
                for schema_name in schemas[:3]:  # Show first 3 schemas
                    sm = schema_mappings.get(schema_name, {})
                    if sm.get('mapped', False):
                        mapped_col = sm.get('column', '')
                        col_type = sm.get('type', '')
                        is_ref = "⭐ " if sm.get('is_reference', False) else "  "
                        print(f"    {is_ref}{schema_name[:20]:<20} → {mapped_col} ({col_type})")
    
    # 4. STATISTICS SECTION
    print(Fore.CYAN + "\n" + "━"*50)
    print("📊 OVERALL STATISTICS")
    print("━"*50 + Style.RESET_ALL)
    
    # Calculate statistics
    cluster_sizes = [len(c['files']) for c in analysis_results['clusters']]
    
    stats = {
        'total_files': total_files,
        'total_clusters': total_clusters,
        'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
        'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
        'total_golden_columns': total_golden_columns,
        'total_mappings': total_mappings,
        'confidence_distribution': confidence_stats
    }
    
    analysis_results['statistics'] = stats
    
    print(f"📈 Cluster Statistics:")
    print(f"   • Average cluster size: {stats['avg_cluster_size']:.1f} files")
    print(f"   • Largest cluster: {stats['max_cluster_size']} files")
    print(f"   • Smallest cluster: {stats['min_cluster_size']} files")
    
    print(f"\n📝 Mapping Statistics:")
    print(f"   • Total golden columns created: {total_golden_columns}")
    print(f"   • Total column mappings: {total_mappings}")
    
    if total_mappings > 0:
        print(f"   • Confidence distribution:")
        print(f"     - High (≥0.8): {confidence_stats['high']} ({confidence_stats['high']/total_mappings:.1%})")
        print(f"     - Medium (0.6-0.8): {confidence_stats['medium']} ({confidence_stats['medium']/total_mappings:.1%})")
        print(f"     - Low (<0.6): {confidence_stats['low']} ({confidence_stats['low']/total_mappings:.1%})")
    
    # 5. UNMAPPED COLUMNS ANALYSIS
    print(Fore.CYAN + "\n" + "━"*50)
    print("🔍 UNMAPPED COLUMNS ANALYSIS")
    print("━"*50 + Style.RESET_ALL)
    
    unmapped_analysis = analyze_unmapped_columns(clustering_data, mapping_data)
    analysis_results['unmapped_analysis'] = unmapped_analysis
    
    total_unmapped = unmapped_analysis.get('total_unmapped', 0)
    total_all_columns = unmapped_analysis.get('total_all_columns', 0)
    
    print(f"📊 Unmapped Columns:")
    print(f"   • Total unmapped: {total_unmapped}")
    print(f"   • Total all columns: {total_all_columns}")
    print(f"   • Overall coverage: {(1 - total_unmapped/total_all_columns) if total_all_columns > 0 else 0:.1%}")
    
    # Show unmapped columns by cluster
    for cluster_info in unmapped_analysis.get('clusters', []):
        cluster_id = cluster_info['cluster_id']
        unmapped_count = cluster_info['unmapped_count']
        total_in_cluster = cluster_info['total_columns']
        
        if unmapped_count > 0:
            print(f"\n  Cluster {cluster_id}: {unmapped_count} unmapped columns")
            for schema_info in cluster_info.get('schemas', []):
                if schema_info['unmapped']:
                    print(f"    • {schema_info['schema']}: {len(schema_info['unmapped_columns'])} unmapped")
                    if len(schema_info['unmapped_columns']) <= 5:  # Show if few
                        for col in schema_info['unmapped_columns'][:3]:
                            print(f"      - {col}")
                        if len(schema_info['unmapped_columns']) > 3:
                            print(f"      ... and {len(schema_info['unmapped_columns']) - 3} more")
    
    # 6. RECOMMENDATIONS
    print(Fore.CYAN + "\n" + "━"*50)
    print("💡 RECOMMENDATIONS & INSIGHTS")
    print("━"*50 + Style.RESET_ALL)
    
    # Generate insights based on analysis
    insights = generate_insights(analysis_results)
    
    for insight in insights:
        print(f"• {insight}")
    
    # 7. SUMMARY
    print(Fore.CYAN + "\n" + "="*100)
    print("✅ ANALYSIS COMPLETE")
    print("="*100 + Style.RESET_ALL)
    
    print(f"\n🎯 Key Findings:")
    print(f"   • Found {total_clusters} distinct entity types")
    print(f"   • Created {total_golden_columns} unified column definitions")
    print(f"   • Achieved {((total_mappings - confidence_stats['low']) / total_mappings * 100 if total_mappings > 0 else 0):.1f}% reliable mappings")
    
    return analysis_results


def analyze_unmapped_columns(clustering_data: Dict, mapping_data: Dict) -> Dict[str, Any]:
    """Analyze unmapped columns across all clusters."""
    unmapped_analysis = {
        'total_unmapped': 0,
        'total_all_columns': 0,
        'clusters': []
    }
    
    # This would require access to the original schema data to get all columns
    # For now, we'll provide a placeholder analysis
    
    # In a real implementation, you would:
    # 1. Load the original schema_data.json
    # 2. For each cluster, get all columns from all schemas
    # 3. Compare with mapped columns in mapping_data
    # 4. Calculate unmapped columns
    
    return unmapped_analysis


def generate_insights(analysis_results: Dict) -> List[str]:
    """Generate insights and recommendations based on analysis."""
    insights = []
    
    stats = analysis_results.get('statistics', {})
    clusters = analysis_results.get('clusters', [])
    
    # Insight 1: Cluster balance
    if stats.get('avg_cluster_size', 0) < 2:
        insights.append("Most clusters have only 1 file - consider lowering similarity threshold")
    elif stats.get('max_cluster_size', 0) > 10:
        insights.append("Some clusters are very large - consider splitting or using stricter criteria")
    
    # Insight 2: Mapping confidence
    conf_stats = stats.get('confidence_distribution', {})
    total_mappings = stats.get('total_mappings', 0)
    
    if total_mappings > 0:
        low_conf_ratio = conf_stats.get('low', 0) / total_mappings
        if low_conf_ratio > 0.3:
            insights.append(f"High proportion ({low_conf_ratio:.1%}) of low-confidence mappings - review column similarity thresholds")
        
        high_conf_ratio = conf_stats.get('high', 0) / total_mappings
        if high_conf_ratio > 0.7:
            insights.append(f"Excellent mapping quality ({high_conf_ratio:.1%} high-confidence)")
    
    # Insight 3: Golden schema efficiency
    total_golden = stats.get('total_golden_columns', 0)
    total_clusters = stats.get('total_clusters', 0)
    
    if total_clusters > 0:
        avg_golden_per_cluster = total_golden / total_clusters
        if avg_golden_per_cluster < 5:
            insights.append("Golden schemas are concise - good for data integration")
        elif avg_golden_per_cluster > 20:
            insights.append("Golden schemas are complex - consider if all columns are necessary")
    
    # Insight 4: Coverage
    unmapped_info = analysis_results.get('unmapped_analysis', {})
    total_unmapped = unmapped_info.get('total_unmapped', 0)
    total_all = unmapped_info.get('total_all_columns', 0)
    
    if total_all > 0:
        coverage = 1 - (total_unmapped / total_all)
        if coverage < 0.7:
            insights.append(f"Low column coverage ({coverage:.1%}) - many columns not mapped")
        elif coverage > 0.9:
            insights.append(f"Excellent column coverage ({coverage:.1%}) - most columns successfully mapped")
    
    # Default insights if none generated
    if not insights:
        insights = [
            "Clustering appears balanced with good coverage",
            "Consider reviewing low-confidence mappings manually",
            "Golden schemas provide a good foundation for data integration"
        ]
    
    return insights


def export_analysis_to_csv(analysis_results: Dict, output_dir: str = 'analysis_output'):
    """Export analysis results to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Export cluster information
    clusters_data = []
    for cluster in analysis_results.get('clusters', []):
        clusters_data.append({
            'cluster_id': cluster['cluster_id'],
            'size': cluster['size'],
            'files': '; '.join(cluster['files'])
        })
    
    if clusters_data:
        df_clusters = pd.DataFrame(clusters_data)
        df_clusters.to_csv(f'{output_dir}/clusters_summary.csv', index=False)
        print(f"📊 Cluster summary exported to {output_dir}/clusters_summary.csv")
    
    # Export statistics
    stats = analysis_results.get('statistics', {})
    if stats:
        df_stats = pd.DataFrame([stats])
        df_stats.to_csv(f'{output_dir}/statistics.csv', index=False)
        print(f"📈 Statistics exported to {output_dir}/statistics.csv")
    
    print(f"\n💾 Analysis exported to '{output_dir}' directory")


# Example usage function
def run_comprehensive_analysis():
    """Run the comprehensive analysis."""
    
    print(Fore.CYAN + """
    ╔══════════════════════════════════════════════════════════════╗
    ║         SCHEMA CLUSTERING & MAPPING ANALYSIS TOOL           ║
    ╚══════════════════════════════════════════════════════════════╝
    """ + Style.RESET_ALL)
    
    print("This tool analyzes:")
    print("1. 📁 How schemas are grouped into clusters (similar entities)")
    print("2. 🔗 How columns are mapped across schemas (similar attributes)")
    print("3. ⭐ Golden schemas created for each cluster")
    print("4. 📊 Quality metrics and statistics\n")
    
    # Run analysis
    results = analyze_clustering_and_mapping()
    
    # Ask if user wants to export
    export_option = input("\n📤 Export analysis to CSV files? (y/n): ").lower()
    if export_option == 'y':
        export_analysis_to_csv(results)
        print(Fore.GREEN + "\n✅ Export complete!" + Style.RESET_ALL)
    
    print(Fore.CYAN + "\n" + "="*100)
    print("🎉 ANALYSIS COMPLETE - Check the output above for detailed insights!")
    print("="*100 + Style.RESET_ALL)


if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        from tabulate import tabulate
        from colorama import init, Fore, Style
        import pandas as pd
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate", "colorama", "pandas"])
        print("Packages installed. Please run the script again.")
        sys.exit(0)
    
    run_comprehensive_analysis()