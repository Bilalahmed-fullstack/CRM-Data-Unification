import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class DataUnificationPhase1:
    def __init__(self, data_dir='../../Amazon Sales Dataset/raw data/'):
        self.data_dir = Path(data_dir).expanduser().resolve()
        print(f"📁 Data directory set to: {self.data_dir}")
        # self.data_dir = Path(data_dir)
        # print(self.data_dir)
        self.datasets = {}
        self.profiles = {}
        
    def load_all_datasets(self):
        """Load all 9 datasets from 3 systems"""
        print("📥 Loading datasets from 3 systems...")
        
        # SYSTEM 1: LEGACY (CSV)
        legacy_path = self.data_dir / 'legacy customers'
        self.datasets['legacy_customers'] = pd.read_csv(legacy_path / 'legacy_customers.csv')
        self.datasets['legacy_products'] = pd.read_csv(legacy_path / 'legacy_products.csv')
        self.datasets['legacy_sales'] = pd.read_csv(legacy_path / 'legacy_sales.csv')
        
        # SYSTEM 2: MODERN (CSV)
        modern_path = self.data_dir / 'modern customers'
        self.datasets['modern_customers'] = pd.read_csv(modern_path / 'modern_customers.csv')
        self.datasets['modern_products'] = pd.read_csv(modern_path / 'modern_products.csv')
        self.datasets['modern_sales'] = pd.read_csv(modern_path / 'modern_sales.csv')
        
        # SYSTEM 3: THIRD (JSON)
        third_path = self.data_dir / 'third customers'
        with open(third_path / 'third_system_customers.json', 'r') as f:
            self.datasets['third_customers'] = pd.DataFrame(json.load(f))
        with open(third_path / 'third_system_products.json', 'r') as f:
            self.datasets['third_products'] = pd.DataFrame(json.load(f))
        with open(third_path / 'third_system_sales.json', 'r') as f:
            self.datasets['third_sales'] = pd.DataFrame(json.load(f))
        
        print(f"✅ Loaded {len(self.datasets)} datasets")
        return self
    
    def analyze_schema_differences(self):
        """Analyze schema differences across systems"""
        print("\n📊 Analyzing Schema Differences...")
        
        schema_comparison = {}
        for name, df in self.datasets.items():
            schema_comparison[name] = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'row_count': len(df)
            }
        
        # Display schema comparison
        schema_df = pd.DataFrame({
            'Dataset': list(schema_comparison.keys()),
            'Columns': [', '.join(info['columns']) for info in schema_comparison.values()],
            'Rows': [info['row_count'] for info in schema_comparison.values()]
        })
        
        print(schema_df.to_string(index=False))
        return schema_comparison
    
    def profile_data_quality(self):
        """Profile data quality issues for each dataset"""
        print("\n🔍 Profiling Data Quality Issues...")
        
        quality_report = {}
        for name, df in self.datasets.items():
            report = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'unique_values_per_column': {col: df[col].nunique() for col in df.columns}
            }
            
            # Check for specific data quality issues
            issues = []
            
            # Check for empty strings that might represent missing data
            for col in df.select_dtypes(include=['object']).columns:
                empty_strings = (df[col] == '').sum() + (df[col].astype(str).str.strip() == '').sum()
                if empty_strings > 0:
                    issues.append(f"{col}: {empty_strings} empty strings")
            
            # Check date columns for invalid formats
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'created' in col.lower()]
            for col in date_columns:
                if col in df.columns:
                    invalid_dates = pd.to_datetime(df[col], errors='coerce').isna().sum()
                    if invalid_dates > 0:
                        issues.append(f"{col}: {invalid_dates} invalid dates")
            
            # Check numeric columns for outliers/zeros
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in df.columns:
                    zeros = (df[col] == 0).sum()
                    negatives = (df[col] < 0).sum() if col != 'premium_member' else 0
                    if zeros > len(df) * 0.5:  # More than 50% zeros
                        issues.append(f"{col}: {zeros} zero values")
                    if negatives > 0:
                        issues.append(f"{col}: {negatives} negative values")
            
            report['specific_issues'] = issues
            quality_report[name] = report
        
        # Print summary
        for name, report in quality_report.items():
            print(f"\n📋 {name}:")
            print(f"   Rows: {report['total_rows']}, Columns: {report['total_columns']}")
            print(f"   Missing values: {report['missing_values']} ({report['missing_percentage']:.1f}%)")
            print(f"   Duplicate rows: {report['duplicate_rows']}")
            if report['specific_issues']:
                print(f"   Issues: {len(report['specific_issues'])} found")
                for issue in report['specific_issues'][:3]:  # Show first 3 issues
                    print(f"     - {issue}")
        
        return quality_report
    
    def analyze_identifier_formats(self):
        """Analyze different ID formats across systems"""
        print("\n🔑 Analyzing Identifier Formats...")
        
        id_patterns = {}
        for name, df in self.datasets.items():
            id_cols = [col for col in df.columns if 'id' in col.lower() or 'sku' in col.lower() 
                      or 'asin' in col.lower() or 'code' in col.lower()]
            if id_cols:
                patterns = {}
                for col in id_cols:
                    sample_values = df[col].head(5).tolist()
                    patterns[col] = {
                        'sample': sample_values,
                        'unique_count': df[col].nunique(),
                        'format_pattern': self._detect_pattern(sample_values[0] if len(sample_values) > 0 else '')
                    }
                id_patterns[name] = patterns
        
        # Display ID patterns
        print("Identifier Patterns Found:")
        for dataset, patterns in id_patterns.items():
            print(f"\n{dataset}:")
            for col, info in patterns.items():
                print(f"  {col}: {info['format_pattern']} (unique: {info['unique_count']})")
                print(f"    Sample: {info['sample']}")
        
        return id_patterns
    
    def _detect_pattern(self, value):
        """Detect pattern of an identifier"""
        if pd.isna(value):
            return "Empty"
        value = str(value)
        if value.startswith('CUST-'):
            return "CUST-###"
        elif value.startswith('PROD-'):
            return "PROD-A###"
        elif value.startswith('ORD-'):
            return "ORD-L####"
        elif value.startswith('USER-'):
            return "USER-####"
        elif value.startswith('B0') and len(value) == 10:
            return "ASIN (B0########)"
        elif value.startswith('TXN-'):
            return "TXN-M####"
        elif value.startswith('ACC-'):
            return "ACC-#####"
        elif value.startswith('ITEM-'):
            return "ITEM-####"
        elif value.startswith('SALE-'):
            return "SALE-#####"
        else:
            return "Unknown"
    
    def generate_summary_statistics(self):
        """Generate overall summary statistics"""
        print("\n📈 Generating Summary Statistics...")
        
        summary = {
            'total_datasets': len(self.datasets),
            'total_records': sum(len(df) for df in self.datasets.values()),
            'system_breakdown': {}
        }
        
        # Group by system
        systems = {}
        for name, df in self.datasets.items():
            system = name.split('_')[0]  # legacy, modern, third
            if system not in systems:
                systems[system] = {'count': 0, 'tables': []}
            systems[system]['count'] += len(df)
            systems[system]['tables'].append(name)
        
        summary['system_breakdown'] = systems
        
        # Print summary
        print(f"Total Datasets: {summary['total_datasets']}")
        print(f"Total Records: {summary['total_records']:,}")
        print("\nSystem Breakdown:")
        for system, info in systems.items():
            print(f"  {system.title()} System: {info['count']:,} records")
            print(f"    Tables: {', '.join(info['tables'])}")
        
        return summary
    
    def identify_data_types_inconsistencies(self):
        """Identify inconsistent data types across systems"""
        print("\n⚡ Identifying Data Type Inconsistencies...")
        
        inconsistencies = {}
        
        # Check common fields across systems
        common_fields = {
            'customer_emails': {
                'legacy': 'email_address',
                'modern': 'email',
                'third': 'email_addr'
            },
            'customer_names': {
                'legacy': ['first_name', 'last_name'],
                'modern': 'full_name',
                'third': 'customer_name'
            },
            'phone_numbers': {
                'legacy': 'phone_number',
                'modern': 'phone',
                'third': 'contact_number'
            },
            'dates': {
                'legacy': 'registration_date',
                'modern': 'account_created',
                'third': 'signup_date'
            }
        }
        
        for field_type, field_map in common_fields.items():
            print(f"\n🔍 {field_type.replace('_', ' ').title()}:")
            for system, field in field_map.items():
                if isinstance(field, list):
                    dataset_name = f"{system}_customers"
                    if dataset_name in self.datasets:
                        df = self.datasets[dataset_name]
                        sample = []
                        for f in field:
                            if f in df.columns:
                                sample.append(df[f].iloc[0] if len(df) > 0 else 'N/A')
                        print(f"  {system}: {field} → Sample: {sample}")
                else:
                    dataset_name = f"{system}_customers"
                    if dataset_name in self.datasets:
                        df = self.datasets[dataset_name]
                        if field in df.columns:
                            sample = df[field].iloc[0] if len(df) > 0 else 'N/A'
                            print(f"  {system}: {field} → Sample: {sample}")
                        else:
                            print(f"  {system}: {field} → NOT FOUND")
        
        return inconsistencies
    
    def save_phase1_report(self, output_dir='reports'):
        """Save Phase 1 analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f'phase1_analysis_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA UNIFICATION PROJECT - PHASE 1 ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. DATASETS LOADED:\n")
            for name, df in self.datasets.items():
                f.write(f"   {name}: {len(df)} rows, {len(df.columns)} columns\n")
            
            f.write("\n2. SCHEMA DIFFERENCES:\n")
            for name, df in self.datasets.items():
                f.write(f"\n   {name}:\n")
                f.write(f"     Columns: {', '.join(df.columns)}\n")
            
            f.write("\n3. DATA QUALITY ISSUES:\n")
            quality_report = self.profile_data_quality()
            for name, report in quality_report.items():
                f.write(f"\n   {name}:\n")
                f.write(f"     Missing values: {report['missing_values']} ({report['missing_percentage']:.1f}%)\n")
                f.write(f"     Duplicate rows: {report['duplicate_rows']}\n")
                if report['specific_issues']:
                    f.write(f"     Issues found: {len(report['specific_issues'])}\n")
                    for issue in report['specific_issues']:
                        f.write(f"       - {issue}\n")
        
        print(f"\n📄 Report saved to: {report_file}")
        return report_file
    
    def run_full_analysis(self):
        """Run complete Phase 1 analysis"""
        print("=" * 80)
        print("DATA UNIFICATION PROJECT - PHASE 1: DATA ACQUISITION & ANALYSIS")
        print("=" * 80)
        
        # Execute all analysis steps
        self.load_all_datasets()
        self.analyze_schema_differences()
        self.profile_data_quality()
        self.analyze_identifier_formats()
        self.generate_summary_statistics()
        self.identify_data_types_inconsistencies()
        
        # Save comprehensive report
        report_path = self.save_phase1_report()
        
        print("\n" + "=" * 80)
        print("PHASE 1 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return self


# Main execution
if __name__ == "__main__":
    # Initialize and run Phase 1
    analyzer = DataUnificationPhase1(data_dir='../../Amazon Sales Dataset/raw data/')
    
    # Run complete analysis
    analyzer.run_full_analysis()
    
    # Access loaded data for next phase
    print("\n📂 Datasets available for next phase:")
    for name, df in analyzer.datasets.items():
        print(f"  - {name}: {len(df)} rows")
    
    # Quick preview of first few rows
    print("\n👀 Sample data from legacy_customers:")
    print(analyzer.datasets['legacy_customers'].head(3))
    
    print("\n👀 Sample data from modern_customers:")
    print(analyzer.datasets['modern_customers'].head(3))
    
    print("\n👀 Sample data from third_customers:")
    print(analyzer.datasets['third_customers'].head(3))