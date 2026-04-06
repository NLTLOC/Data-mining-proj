import numpy as np
import pandas as pd
from itertools import chain

def detect_column_types(data):
    """
    Phân loại cột, kiểu số và ký tự.
    
    Returns:
        tuple chứa 2 danh sách: (num_cols, nonNum_cols)
    """
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    nonNum_cols = data.select_dtypes(include=['object']).columns.tolist()
    return num_cols, nonNum_cols

def handle_missing_data(data):
    """
    Xử lý dữ liệu thiếu dựa trên loại cột.
    - Num columns: fill with median
    - Non-Num columns: fill with mode
    - Unknown columns: fill with 'Unknown'
    
    Returns:
        DataFrame đã được xử lý thiếu dữ liệu
    """
    data = data.copy()
    num_cols, nonNum_cols = detect_column_types(data)
    # Handle numerical columns
    for col in num_cols:
        missing_count = data[col].isna().sum()
        if missing_count > 0:
            data[col] = data[col].fillna(data[col].median())
            print(f"{col}: Filled {missing_count} missing values with median")
    
    # Handle non-numerical columns
    for col in nonNum_cols:
        missing_count = data[col].isna().sum()
        if missing_count > 0:
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
            print(f"{col}: Filled {missing_count} missing values with mode")
    return data

def handle_duplicates(data):
    """
    Xử lý dữ liệu trùng lặp, tự phát hiện ID.
    
    Returns:
        DataFrame không trùng lặp
    """
    data = data.copy()
    initial_rows = len(data)
    
    #Remove exact duplicates
    data = data.drop_duplicates(keep='first')
    exact_dup_removed = initial_rows - len(data)
    
    # Find and remove ID-based duplicates
    id_dup_removed = 0
    id_cols = [col for col in data.columns if 'id' in col.lower()]
    if id_cols:
        initial_rows_id = len(data)
        for id_col in id_cols:
            data = data.drop_duplicates(subset=[id_col], keep='first')
        id_dup_removed = initial_rows_id - len(data)
        if id_dup_removed > 0:
            print(f"Removed {id_dup_removed} duplicates based on ID columns: {id_cols}")
    
        if exact_dup_removed > 0 or id_dup_removed > 0:
            print(f"Total duplicates removed: {exact_dup_removed + id_dup_removed}")
        else:
            print(f"No duplicates found")
    return data

def handle_outliers(data, columns=None):
    """
    Xử lý ngoại lai bằng IQR.
    Thay bằng median 
    Args:
        data: DataFrame
        columns: List of columns (None = all numerical)
    Returns:
        DataFrame đã được xử lý ngoại lai
    """
    data = data.copy()
    numerical_cols, _ = detect_column_types(data)
    
    if columns is None:
        columns = numerical_cols
    else:
        columns = [col for col in columns if col in numerical_cols]
    
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outliers > 0:
                median = data[col].median()
                data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = median
                print(f"{col}: Replaced {outliers} outliers with median ({median:.2f})")
    return data

def handle_inconsistent_data(data):
    """
    Chuẩn hóa dữ liệu không nhất quán:
    - Bỏ whitespace
    - Convert case for column là string
    Args:
        data: DataFrame
    Returns:
        DataFrame đã được chuẩn hóa
    """
    data = data.copy()
    _, nonNum = detect_column_types(data)
    
    for col in nonNum:
        if data[col].dtype == 'object':
            # Strip whitespace and standardize
            data[col] = data[col].str.strip()
            # Check if column looks like it should be title case
            if col.lower() not in ['gender', 'status', 'result', 'grade', 'category']:
                data[col] = data[col].str.title()
                print(f"{col}: Standardized inconsistent values")
    return data


def preprocess_dataset(file_path, handle_missing=True, handle_dups=True, 
                    handle_outliers_flag=True, handle_inconsistent=True, 
                    outlier_columns=None):
    """
    Tách tiền xử lý thành một hàm tổng quát.
    Args:
        file_path: Path to CSV file
        handle_missing: Whether to handle missing values
        handle_dups: Whether to handle duplicates
        handle_outliers_flag: Whether to handle outliers
        handle_inconsistent: Whether to handle inconsistent data
        outlier_columns: Specific columns for outlier detection (None = all numerical)
    Returns:
        Cleaned DataFrame
    """
    # Load data
    data = pd.read_csv(file_path, encoding="utf-8")
    print(f"PREPROCESSING DATASET: {file_path}")
    print(f"{'='*60}")
    print(f"Initial dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Missing values:\n{data.isna().sum()}\n")
    
    # Apply preprocessing steps
    if handle_missing:
            print("STEP 1: Handling Missing Data")
            data = handle_missing_data(data)

    if handle_dups:
            print("\nSTEP 2: Handling Duplicates")
            data = handle_duplicates(data)
    
    if handle_inconsistent:
            print("\nSTEP 3: Standardizing Inconsistent Data")
            data = handle_inconsistent_data(data)
    
    if handle_outliers_flag:
            print("\nSTEP 4: Handling Outliers (IQR Method)")
            data = handle_outliers(data, columns=outlier_columns)

    # Summary
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Final dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Remaining missing values:\n{data.isna().sum()}\n")
    return data

def support(itemset, set_data):
    """
    Tính support của itemset
    Support = (number of transactions containing itemset) / (total transactions)
    """
    count = 0
    for t in set_data:
        if itemset.issubset(t):
            count += 1
    return count / len(set_data) if len(set_data) > 0 else 0

def get_data_info(data):
    print("DATASET INFORMATION")
    print("="*60)
    print(f"\nDataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"\nColumn names and types:")
    for col in data.columns:
        print(f"  • {col}: {data[col].dtype}")
    print(f"\nBasic statistics:")
    print(data.describe())
    print(f"\nData types summary:")
    print(data.dtypes)
    print("="*60 + "\n")


def main():
    """
    Main preprocessing.
    """
    # Step 1: Preprocess the dataset
    cleaned_df = preprocess_dataset(
        "Uni_Stu_ds_proj.csv",
        handle_missing=True,
        handle_dups=True,
        handle_outliers_flag=True,
        handle_inconsistent=True,
    )
    # Step 2: Display data information
    get_data_info(cleaned_df)
    
    # Step 3: Save cleaned dataset
    output_file = "Uni_Stu_cleaned.csv"
    cleaned_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Cleaned dataset saved to: {output_file}\n")
    
    # Step 4: Prepare data for association rules
    set_data = [set(row) for row in cleaned_df.values]
    
    # Optional: Display sample statistics
    print("Sample data (first 5 rows):")
    print(cleaned_df.head())
    return cleaned_df

if __name__ == "__main__":
    main()