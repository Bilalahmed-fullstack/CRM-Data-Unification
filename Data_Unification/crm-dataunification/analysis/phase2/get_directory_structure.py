import os
import pandas as pd
import json

def get_file_columns(directory):
    file_columns_dict = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.csv', '.json')):
                file_path = os.path.join(root, file)
                
                try:
                    if file.endswith('.csv'):
                        # Read CSV file and get columns
                        df = pd.read_csv(file_path, nrows=0)  # Read only headers
                        columns = list(df.columns)
                    elif file.endswith('.json'):
                        # Read JSON file and get columns
                        # For JSON, we need to read the first object to determine structure
                        with open(file_path, 'r') as f:
                            # Try different JSON formats
                            data = json.load(f)
                            
                            # Handle different JSON structures
                            if isinstance(data, list) and len(data) > 0:
                                # JSON array of objects
                                columns = list(data[0].keys())
                            elif isinstance(data, dict):
                                # Single JSON object or nested structure
                                columns = list(data.keys())
                            else:
                                columns = ["Could not determine columns"]
                    
                    file_columns_dict[file] = columns
                    
                except Exception as e:
                    file_columns_dict[file] = [f"Error reading file: {str(e)}"]
    
    return file_columns_dict
def print_structure(file_columns_dict):
    # Print the dictionary
    for file_name, columns in file_columns_dict.items():
        print(f"{file_name}:")
        print(f"  Columns: {columns}")
        print("-" * 50)

