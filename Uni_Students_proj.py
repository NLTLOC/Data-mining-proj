import numpy as np
import pandas as pd
from itertools import chain, combinations
import os

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

def confidence(itemset, antecedent, set_data):
    """
    Tính confidence của rule: antecedent => itemset - antecedent
    """
    if not antecedent:
        return 0
    count_antecedent = 0
    count_both = 0
    for t in set_data:
        if antecedent.issubset(t):
            count_antecedent += 1
            if itemset.issubset(t):
                count_both += 1
    return count_both / count_antecedent if count_antecedent > 0 else 0

def lift(itemset, antecedent, set_data):
    """
    Tính lift của rule: antecedent => itemset - antecedent
    Lift = confidence / support of consequent
    """
    if not antecedent:
        return 0
    confidence_value = confidence(itemset, antecedent, set_data)
    consequent = itemset - antecedent
    support_consequent = support(consequent, set_data)
    return confidence_value / support_consequent if support_consequent > 0 else 0

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

def cleaned_data_info(file_path=None):
    """
    Main preprocessing.
    """
    # Use script directory if no file path provided
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "Uni_Stu_ds_proj.csv")
    # Step 1: Preprocess the dataset
    cleaned_df = preprocess_dataset(
        file_path,
        handle_missing=True,
        handle_dups=True,
        handle_outliers_flag=True,
        handle_inconsistent=True,
    )
    # Step 2: Display data information
    get_data_info(cleaned_df)
    
    # Step 3: Save cleaned dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "Uni_Stu_cleaned.csv")
    cleaned_df.to_csv(output_file, index=False, encoding="utf-8", float_format='%.1f')
    print(f"Cleaned dataset saved to: {output_file}\n")
    
    # Step 4: Prepare data for association rules
    set_data = [set(row) for row in cleaned_df.values]
    
    # Optional: Display sample statistics
    print("Sample data (first 5 rows):")
    print(cleaned_df.head())
    return cleaned_df

def discretize_data(data):
    """
    Convert numerical data to non-numerical itemsets.
    Creates meaningful items like 'Attendance=High', 'Test_Score=Low', etc.
    """
    discretized = []
    for idx, row in data.iterrows():
        items = set()
        for col in data.columns:
            value = row[col]
            if col == 'Final_Result':
                # Use result directly
                items.add(f"Result={value}")
            elif isinstance(value, (int, float)):
                # Discretize numerical values using median
                median = data[col].median()
                if value >= median:
                    items.add(f"{col}=High")
                else:
                    items.add(f"{col}=Low")
            else:
                items.add(f"{col}={value}")
        discretized.append(items)
    return discretized

def generate_frequent_itemsets(set_data, min_support):
    """
    Generate frequent itemsets of all sizes using Apriori approach.
    """
    # Start with 1-itemsets
    items = set()
    for itemset in set_data:
        items.update(itemset)
    
    frequent = []
    candidates = [{item} for item in items]
    
    while candidates:
        # Filter candidates by support
        current_frequent = []
        for candidate in candidates:
            if support(candidate, set_data) >= min_support:
                current_frequent.append(candidate)
                frequent.append(candidate)
        
        if not current_frequent:
            break
        
        # Generate new candidates (2-itemsets, 3-itemsets, etc.)
        next_candidates = []
        for i, itemset1 in enumerate(current_frequent):
            for itemset2 in current_frequent[i+1:]:
                union = itemset1 | itemset2
                if len(union) == len(itemset1) + 1:
                    next_candidates.append(union)
        
        candidates = next_candidates
    
    return frequent

def generate_LKH(data, min_support=0.2, min_confidence=0.5, min_lift=1.0):
    """
    Tạo ra các luật kết hợp dùng Apriori.
    Args:
        data: DataFrame đã được làm sạch
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
    Returns:
        List of strong association rules
    """
    # Step 1: Discretize data
    print("Discretizing data...")
    set_data = discretize_data(data)
    print(f"Sample discretized transaction: {list(set_data[0])}")
    
    # Step 2: Generate frequent itemsets
    print(f"\nGenerating frequent itemsets (min_support={min_support})...")
    frequent_itemsets = generate_frequent_itemsets(set_data, min_support)
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    # Step 3: Generate rules from frequent itemsets
    print(f"\nGenerating rules (min_confidence={min_confidence}, min_lift={min_lift})...")
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue  # Skip 1-itemsets
        
        # Generate all possible antecedents for this itemset
        itemset_list = list(itemset)
        for i in range(1, len(itemset_list)):
            for antecedent_combo in combinations(itemset_list, i):
                antecedent = frozenset(antecedent_combo)
                consequent = itemset - set(antecedent)
                
                conf = confidence(itemset, antecedent, set_data)
                lf = lift(itemset, antecedent, set_data)
                
                if conf >= min_confidence and lf >= min_lift:
                    rules.append({
                        'antecedent': set(antecedent),
                        'consequent': consequent,
                        'confidence': conf,
                        'lift': lf,
                        'support': support(itemset, set_data)
                    })
    
    print(f"\nGenerated {len(rules)} strong association rules")
    return rules

def main():
    cleaned_df = cleaned_data_info()
    rules = generate_LKH(cleaned_df)
    
    if rules:
        print("\n" + "="*60)
        print("STRONG ASSOCIATION RULES")
        print("="*60)
        for i, rule in enumerate(rules[:10], 1):  # Show first 10 rules
            print(f"\nRule {i}:")
            print(f"  {set(rule['antecedent'])} => {rule['consequent']}")
            print(f"  Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f}, Support: {rule['support']:.2%}")
    else:
        print("\nNo strong association rules found. Try lowering min_support or min_confidence.")

if __name__ == "__main__":
    main()