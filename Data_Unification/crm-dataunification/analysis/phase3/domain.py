# Import the csv_detective package
from csv_detective import routine
import os # for this example only

# Replace by your file path
file_path = os.path.join('.', 'legacy_customers.csv')

# Open your file and run csv_detective
inspection_results = routine(
  file_path, # or file URL
  num_rows=-1, # Value -1 will analyze all lines of your file, you can change with the number of lines you wish to analyze
  save_results=False, # Default False. If True, it will save result output into the same directory as the analyzed file, using the same name as your file and .json extension
  output_profile=True, # Default False. If True, returned dict will contain a property "profile" indicating profile (min, max, mean, tops...) of every column of your csv
  output_schema=True, # Default False. If True, returned dict will contain a property "schema" containing basic [tableschema](https://specs.frictionlessdata.io/table-schema/) of your file. This can be used to validate structure of other csv which should match same structure. 
  tags=["fr"],  # Default None. If set as a list of strings, only performs checks related to the specified tags (you can see the available tags with FormatsManager().available_tags())
)
print(inspection_results)