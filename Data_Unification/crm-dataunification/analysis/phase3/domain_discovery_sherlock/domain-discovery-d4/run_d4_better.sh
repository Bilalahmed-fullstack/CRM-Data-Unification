#!/bin/bash
# complete_d4_v2.sh - Complete D4 v2 pipeline with error checking

JAR="target/D4-jar-with-dependencies.jar"
echo "🔧 Completing D4 v2 pipeline with error checking..."
echo "=" * 50

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo "✅ Found: $1"
        return 0
    else
        echo "❌ Missing: $1"
        return 1
    fi
}

# Function to run step only if needed
run_step() {
    echo ""
    echo "Step $1: $2"
    if check_file "$3"; then
        echo "   Already exists, skipping..."
    else
        echo "   Running..."
        eval "$4"
        if [ $? -eq 0 ] && [ -f "$3" ]; then
            echo "   ✅ Success"
        else
            echo "   ❌ Failed to create $3"
            return 1
        fi
    fi
    return 0
}

# ========== STEP 1: Check what exists ==========
echo "📋 Checking existing files..."
check_file "customers_columns_v2/" && echo "   (directory)"
check_file "customers_term-index_v2.txt.gz"
check_file "customers_eqs_v2.txt.gz"
check_file "customers_signatures_v2.txt.gz"
check_file "customers_expanded-columns_v2.txt.gz"
check_file "customers_local-domains_v2.txt.gz"
check_file "customers_strong-domains_v2.txt.gz"

echo ""
echo "🚀 Starting from first missing step..."
echo ""

# ========== STEP 2: Run missing steps ==========
# Step 1: Columns (directory)
if [ ! -d "customers_columns_v2" ]; then
    echo "1. Converting CSV files to column files..."
    java -jar $JAR columns \
      --input=data_pool/customers \
      --output=customers_columns_v2 \
      --threads=2
fi

# Step 2: Term index
run_step "2" "Creating term index" \
  "customers_term-index_v2.txt.gz" \
  "java -jar $JAR term-index --input=customers_columns_v2 --output=customers_term-index_v2.txt.gz --textThreshold='GT0.3'"

# Step 3: Equivalence classes
run_step "3" "Generating equivalence classes" \
  "customers_eqs_v2.txt.gz" \
  "java -jar $JAR eqs --input=customers_term-index_v2.txt.gz --output=customers_eqs_v2.txt.gz"

# Step 4: Signatures
run_step "4" "Generating signatures" \
  "customers_signatures_v2.txt.gz" \
  "java -jar $JAR signatures --eqs=customers_eqs_v2.txt.gz --signatures=customers_signatures_v2.txt.gz --sim=TF-ICF --robustifier=IGNORE-LAST --ignoreMinorDrop=false"

# Step 5: Expand columns
run_step "5" "Expanding columns" \
  "customers_expanded-columns_v2.txt.gz" \
  "java -jar $JAR expand-columns --eqs=customers_eqs_v2.txt.gz --signatures=customers_signatures_v2.txt.gz --columns=customers_expanded-columns_v2.txt.gz --expandThreshold='GT0.15' --decrease=0.02 --iterations=8 --trimmer=LIBERAL"

# Step 6: Local domains
run_step "6" "Discovering local domains" \
  "customers_local-domains_v2.txt.gz" \
  "java -jar $JAR local-domains --eqs=customers_eqs_v2.txt.gz --columns=customers_expanded-columns_v2.txt.gz --signatures=customers_signatures_v2.txt.gz --localdomains=customers_local-domains_v2.txt.gz"

# Step 7: Strong domains
run_step "7" "Discovering strong domains" \
  "customers_strong-domains_v2.txt.gz" \
  "java -jar $JAR strong-domains --eqs=customers_eqs_v2.txt.gz --localdomains=customers_local-domains_v2.txt.gz --strongdomains=customers_strong-domains_v2.txt.gz --domainOverlap='GT0.3' --supportFraction=0.15"

# ========== STEP 3: Export results ==========
echo ""
echo "📦 Exporting results..."

if check_file "customers_strong-domains_v2.txt.gz"; then
    # Check columns.tsv location
    if [ -f "columns.tsv" ]; then
        COLUMNS_FILE="columns.tsv"
    elif [ -f "customers_columns_v2.tsv" ]; then
        COLUMNS_FILE="customers_columns_v2.tsv"
    else
        echo "⚠️  Looking for columns.tsv..."
        find . -name "*.tsv" -type f | head -5
        echo "Please specify columns.tsv path:"
        read COLUMNS_FILE
    fi
    
    echo "8. Exporting domains using $COLUMNS_FILE..."
    java -jar $JAR export \
      --eqs=customers_eqs_v2.txt.gz \
      --terms=customers_term-index_v2.txt.gz \
      --columns=$COLUMNS_FILE \
      --domains=customers_strong-domains_v2.txt.gz \
      --output=customers_domains_v2_export_complete \
      --sampleSize=50
    
    if [ $? -eq 0 ]; then
        echo "✅ Export successful!"
        echo "📁 Results in: customers_domains_v2_export_complete/"
        
        # Show results
        echo ""
        echo "🎯 Discovered domains:"
        if [ -d "customers_domains_v2_export_complete" ]; then
            JSON_COUNT=$(ls customers_domains_v2_export_complete/*.json 2>/dev/null | wc -l)
            echo "   Found $JSON_COUNT domain files"
            
            # Quick preview
            for file in customers_domains_v2_export_complete/domain_*.json; do
                [ -f "$file" ] || continue
                echo ""
                echo "=== $(basename $file) ==="
                python3 -c "
import json
try:
    with open('$file', 'r') as f:
        data = json.load(f)
    cols = data.get('columns', [])
    print(f'Columns: {len(cols)}')
    for col in cols[:2]:
        print(f'  • {col.get(\"dataset\", \"?\")}.{col.get(\"name\", \"?\")}')
except:
    print('Could not parse JSON')
"
                break  # Just show first one
            done
        fi
    else
        echo "❌ Export failed"
    fi
else
    echo "❌ Cannot export: strong-domains file missing"
fi

echo ""
echo "=" * 50
echo "🎉 D4 pipeline complete!"