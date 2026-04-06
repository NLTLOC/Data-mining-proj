# Flexible Data Preprocessing Guide

## Overview
The updated `Uni_Students_prj.py` contains flexible preprocessing functions that automatically adapt to any dataset structure without requiring manual column configuration.

---

## Key Features

### 1. **Automatic Column Detection**
- Automatically detects numerical vs. categorical columns
- No need to specify column names or types manually
- Works with any CSV file structure

### 2. **Flexible Missing Data Handling**
- **Numerical columns**: Fills with median (robust to outliers)
- **Categorical columns**: Fills with mode (most frequent value)
- Automatically adapts based on column data type

### 3. **Automatic Duplicate Detection**
- General duplicate row removal
- ID column detection and duplicate removal (supports any column with "id" in name)
- Logs removed duplicates

### 4. **Outlier Detection & Handling**
- **IQR Method** (Interquartile Range)
  - Uses Q1 and Q3 quartiles
  - Threshold: 1.5 × IQR
  - Replaces outliers with median
- **Z-Score Method** (optional)
  - Threshold: Z-score > 3
  - Replaces outliers with median

### 5. **Inconsistent Data Standardization**
- Strips leading/trailing whitespace
- Converts case appropriately
- Handles special cases for common categorical types

---

## Usage Examples

### **Example 1: Quick Preprocessing (Default Settings)**
```python
from Uni_Students_prj import preprocess_dataset

# Simple one-line preprocessing
cleaned_df = preprocess_dataset("my_dataset.csv")

# Save to new file
cleaned_df.to_csv("my_dataset_cleaned.csv", index=False)
```

### **Example 2: Preprocess with Custom Options**
```python
# Preprocess with specific columns for outlier handling
cleaned_df = preprocess_dataset(
    file_path="customer_data.csv",
    handle_missing=True,
    handle_dups=True,
    handle_outliers_flag=True,
    outlier_columns=['Age', 'Income', 'Purchase_Amount'],
    verbose=True
)
```

### **Example 3: Skip Specific Preprocessing Steps**
```python
# Skip outlier handling if dataset is already clean
cleaned_df = preprocess_dataset(
    file_path="student_scores.csv",
    handle_missing=True,
    handle_dups=True,
    handle_outliers_flag=False,  # Don't change values
    handle_inconsistent=True,
    verbose=True
)
```

### **Example 4: Use Individual Functions**
```python
import pandas as pd
from Uni_Students_prj import (
    handle_missing_data,
    handle_duplicates,
    handle_outliers,
    handle_inconsistent_data,
    detect_column_types
)

# Load data
df = pd.read_csv("raw_data.csv")

# Apply preprocessing step by step
df = handle_missing_data(df)
df = handle_duplicates(df)
df = handle_inconsistent_data(df)
df = handle_outliers(df, method='iqr')

# Check column types
numerical, categorical = detect_column_types(df)
print(f"Numerical: {numerical}")
print(f"Categorical: {categorical}")
```

### **Example 5: Preprocess Multiple Datasets**
```python
from Uni_Students_prj import preprocess_dataset

datasets = [
    "student_dataset_100rows.csv",
    "customer_data.csv",
    "CleanedStudentData.csv"
]

for dataset in datasets:
    cleaned = preprocess_dataset(
        file_path=dataset,
        verbose=True
    )
    output_name = dataset.replace(".csv", "_cleaned.csv")
    cleaned.to_csv(output_name, index=False)
    print(f"✓ Saved: {output_name}\n")
```

---

## Available Functions

### `detect_column_types(data)`
**Purpose**: Automatically detect numerical and categorical columns
```python
numerical_cols, nonNum = detect_column_types(df)
```

### `handle_missing_data(data, verbose=True)`
**Purpose**: Handle missing values based on column type
```python
df_clean = handle_missing_data(df)
```

### `handle_duplicates(data, verbose=True)`
**Purpose**: Remove duplicate rows and ID-based duplicates
```python
df_clean = handle_duplicates(df)
```

### `handle_outliers(data, columns=None, method='iqr', verbose=True)`
**Purpose**: Detect and replace outliers
```python
# Automatic detection of numerical columns
df_clean = handle_outliers(df)

# Specific columns only
df_clean = handle_outliers(df, columns=['Age', 'Score'], method='iqr')
```

### `handle_inconsistent_data(data, verbose=True)`
**Purpose**: Standardize categorical values
```python
df_clean = handle_inconsistent_data(df)
```

### `preprocess_dataset(file_path, **options)`
**Purpose**: Complete preprocessing pipeline
```python
df_clean = preprocess_dataset(
    file_path="data.csv",
    handle_missing=True,
    handle_dups=True,
    handle_outliers_flag=True,
    handle_inconsistent=True,
    outlier_columns=None,
    verbose=True
)
```

---

## Current Dataset Results

**File**: `Uni_student_dataset_proj.csv`

**Preprocessing Summary**:
- ✓ No missing values found
- ✓ No duplicate rows
- ✓ Final_Result column standardized
- ✓ All numerical columns checked for outliers

**Output**: `Uni_student_dataset_cleaned.csv`

**Dataset Info**:
- Shape: 120 rows × 8 columns
- Numerical columns: 7 (Attendance, Test_Score, LMS_Time_Hours, etc.)
- Categorical columns: 1 (Final_Result)

---

## Workflow for New Datasets

1. **Place CSV file** in the working directory
2. **Run preprocessing**:
   ```python
   df = preprocess_dataset("your_file.csv")
   ```
3. **Review output** - Check for warnings/messages
4. **Save cleaned data**:
   ```python
   df.to_csv("your_file_cleaned.csv", index=False)
   ```
5. **Done!** - File is ready for analysis

---

## Tips & Best Practices

### ✓ **DO**
- Use verbose=True to monitor what's happening
- Review the console output before further analysis
- Test with a small dataset first
- Save cleaned data with a new filename

### ✗ **DON'T**
- Modify the original data files directly
- Skip the preprocessing step (data quality matters!)
- Change outlier threshold for IQR without understanding impact
- Process files with encoding issues without trying different encodings

---

## Troubleshooting

### Problem: UnicodeDecodeError
```python
# Solution: Function automatically tries UTF-8 then Latin-1
# Or specify encoding in preprocess_dataset
```

### Problem: Column names not being detected
```python
# Solution: Check CSV structure
import pandas as pd
df = pd.read_csv("file.csv")
print(df.columns)  # See actual column names
```

### Problem: Too many outliers being replaced
```python
# Solution: Use z-score method instead of IQR
cleaned = preprocess_dataset(
    "file.csv",
    handle_outliers_flag=False  # Skip automatic
)
# Then apply carefully:
df = handle_outliers(cleaned, method='zscore', columns=['specific_col'])
```

---

## Integration with Analysis

Once preprocessed, use your data with analysis tools:

```python
import pandas as pd
from Uni_Students_prj import preprocess_dataset

# 1. Preprocess
df = preprocess_dataset("raw_data.csv")

# 2. Exploratory Analysis
print(df.describe())
print(df.corr())

# 3. Visualization
import matplotlib.pyplot as plt
df.hist(figsize=(12, 8))
plt.show()

# 4. Machine Learning
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
```

---

**Last Updated**: March 2026
