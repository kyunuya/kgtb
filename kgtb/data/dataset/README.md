# 🗺️ CA & NYC Location Dataset

A structured geospatial dataset for (CA), (NYC), and (TKY) designed for POI (Point of Interest) analysis, graph modeling, and spatial machine learning tasks. Download the processed dataset from `https://drive.google.com/drive/folders/18XANnxh8ziYklCjISJDh5xh20JjS4eJH?usp=drive_link`

---

## 📂 Directory Structure
```
dataset/
├── CA/
│   ├── data/ # Processed data for input to LLM
│   ├── graph/ # Graph catname, regison, user
│   ├── raw/ # Original unprocessed source files
│   ├── split_data/ # Train/test splits for training codebook
│   ├── catname_mapping_CA.csv # Category ID → human-readable name
│   ├── data_mapping_CA.csv # Core data-to-internal-ID mapping
│   ├── filtered_CA_train.csv # Filtered training set
│   ├── filtered_CA_val.csv # Filtered validation set
│   ├── filtered_CA_test.csv # Filtered test set
│   ├── pid_mapping_CA.csv # POI ID ↔ canonical ID mapping
│   ├── poi_info_CA.csv # POI metadata (name, lat, lon, category, etc.)
│   ├── region_mapping_CA.csv # Region/neighborhood ID mapping
│   └── uid_mapping_CA.csv # User ID ↔ anonymized ID mapping
│
├── NYC/
│   ├── data/
|   ├── graph/
|   ├── raw/
|   ├── split_data/
|   ├── catname_mapping_NYC.csv
|   ├── data_mapping_NYC.csv
|   ├── filtered_NYC_train.csv
|   ├── filtered_NYC_val.csv
|   ├── filtered_NYC_test.csv
|   ├── pid_mapping_NYC.csv
|   ├── poi_info_NYC.csv
|   ├── region_mapping_NYC.csv
|   └── uid_mapping_NYC.csv
|
├── TKY/
│   ├── data/ 
│   ├── graph/ 
│   ├── raw/ 
│   ├── split_data/ 
│   ├── catname_mapping_TKY.csv 
│   ├── data_mapping_TKY.csv 
│   ├── filtered_TKY_train.csv 
│   ├── filtered_TKY_val.csv
│   ├── filtered_TKY_test.csv 
│   ├── pid_mapping_TKY.csv 
│   ├── poi_info_TKY.csv 
│   ├── region_mapping_TKY.csv 
│   └── uid_mapping_TKY.csv
```