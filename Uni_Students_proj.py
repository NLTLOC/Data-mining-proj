import numpy as np
import pandas as pd
from itertools import combinations
import os

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def detect_column_types(data):
    data = data.copy()
    num_cols    = data.select_dtypes(include=[np.number]).columns.tolist()
    nonNum_cols = data.select_dtypes(include=['object']).columns.tolist()
    return num_cols, nonNum_cols

def handle_missing_data(data):
    data = data.copy()
    num_cols, nonNum_cols = detect_column_types(data)
    for col in num_cols:
        n = data[col].isna().sum()
        if n > 0:
            data[col] = data[col].fillna(data[col].median())
            print(f"  {col}: filled {n} missing values with median")
    for col in nonNum_cols:
        n = data[col].isna().sum()
        if n > 0:
            mode = data[col].mode()
            data[col] = data[col].fillna(mode[0] if not mode.empty else 'Unknown')
            print(f"  {col}: filled {n} missing values with mode")
    return data

def handle_duplicates(data):
    data = data.copy()
    before = len(data)
    data = data.drop_duplicates(keep='first')
    removed = before - len(data)
    id_cols = [c for c in data.columns if 'id' in c.lower()]
    id_removed = 0
    if id_cols:
        before2 = len(data)
        for col in id_cols:
            data = data.drop_duplicates(subset=[col], keep='first')
        id_removed = before2 - len(data)
    total = removed + id_removed
    print(f"  Removed {total} duplicate rows" if total else "  No duplicates found")
    return data

def handle_outliers(data, columns=None):
    data = data.copy()
    num_cols, _ = detect_column_types(data)
    cols = columns if columns else num_cols
    cols = [c for c in cols if c in num_cols]
    for col in cols:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            mask = (data[col] < lo) | (data[col] > hi)
            n = mask.sum()
            if n > 0:
                median = data[col].median()
                data.loc[mask, col] = median
                print(f"  {col}: replaced {n} outliers with median ({median:.2f})")
    return data

def handle_inconsistent_data(data):
    data = data.copy()
    _, nonNum = detect_column_types(data)
    for col in nonNum:
        data[col] = data[col].str.strip()
        data[col] = data[col].str.title()
        print(f"  {col}: title-cased")
    return data

def support_(itemset, transactions):
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)

def support_count(itemset, transactions):
    return sum(1 for t in transactions if itemset.issubset(t))

def confidence_(itemset, antecedent, transactions):
    count_ant  = support_count(antecedent, transactions)
    count_both = support_count(itemset, transactions)
    return count_both / count_ant if count_ant else 0

def lift_(itemset, antecedent, transactions):
    conf = confidence_(itemset, antecedent, transactions)
    consequent = itemset - set(antecedent)
    sup_cons = support_(consequent, transactions)
    return conf / sup_cons if sup_cons else 0

def fix_negative_values(data):
    data = data.copy()
    keywords = ['Attendance', 'Test_Score', 'LMS_Time_Hours', 'Assignments_Submitted', 'Forum_Interactions', 'Study_Hours_Per_Week', 'Previous_GPA']
    cols = [c for c in data.select_dtypes(include=[np.number]).columns
            if any(kw in c.lower() for kw in keywords)]
    for col in cols:
        n = (data[col] < 0).sum()
        if n > 0:
            data[col] = data[col].clip(lower=0)
            print(f"  {col}: clipped {n} negative values to 0")
    return data

def preprocess_dataset(file_path, handle_missing=True, handle_dups=True,
                    handle_outliers_flag=True, handle_inconsistent=True,
                    outlier_columns=None):
    data = pd.read_csv(file_path, encoding="utf-8")
    print(f"{'='*60}")
    print(f"PREPROCESSING: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    missing = data.isna().sum()
    if missing.any():
        print(f"Missing values:\n{missing[missing > 0]}\n")

    print("\n[Step 1] Cleaning negative values")
    data = fix_negative_values(data)
    if handle_missing:
        print("\n[Step 2] Handling missing data")
        data = handle_missing_data(data)
    if handle_dups:
        print("\n[Step 3] Handling duplicates")
        data = handle_duplicates(data)
    if handle_inconsistent:
        print("\n[Step 4] Standardizing inconsistent data")
        data = handle_inconsistent_data(data)
    if handle_outliers_flag:
        print("\n[Step 5] Handling outliers (IQR)")
        data = handle_outliers(data, columns=outlier_columns)

    print(f"\n{'='*60}")
    print(f"Final shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"{'='*60}\n")
    return data

def identify_at_risk_students(data):
    """
    Identify students with poor performance after preprocessing.
    Criteria:
    - Final_Result = Poor (definitely at-risk)
    OR
    - Final_Result = Average AND multiple low indicators
    Returns:
        DataFrame of at-risk students
    """
    df = data.copy()
    at_risk_flags = []
    thresholds = {}
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        thresholds[col] = df[col].quantile(0.25)
        
    for idx, row in df.iterrows():
        risk_score = 0
        
        # Rule 1: Result is Poor → definitely at risk
        if 'Final_Result' in df.columns and str(row['Final_Result']) == 'Poor':
            at_risk_flags.append(True)
            continue
        
        # Rule 2: Result is Average AND has multiple weak indicators → at risk
        if 'Final_Result' in df.columns and str(row['Final_Result']) == 'Average':
            # Count weak indicators
            for col, threshold in thresholds.items():
                if row[col] <= threshold:
                    risk_score += 1
            
            # If at least half the features are in bottom 25% → at risk
            if risk_score >= len(thresholds) // 2:
                at_risk_flags.append(True)
            else:
                at_risk_flags.append(False)
        else:
            # Good or Excellent → not at risk
            at_risk_flags.append(False)
    
    df['At_Risk'] = at_risk_flags
    
    # Extract only risky students and drop the flag column
    at_risk_df = df[df['At_Risk'] == True].drop(columns=['At_Risk'])

    print("\n" + "="*60)
    print("AT-RISK STUDENTS DETECTED")
    print("="*60)
    return at_risk_df

def get_data_info(preprocessed_dataset):
    data = preprocessed_dataset.round(2)
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
    print(f"\nSample data:")
    print(data.head(10))
    print("="*60 + "\n")

# ─────────────────────────────────────────────
# APRIORI
# ─────────────────────────────────────────────

def discretize_data(data):
    data = data.copy()
    discretized = []
    for idx, row in data.iterrows():
        items = set()
        for col in data.columns:
            value = row[col]
            if col == 'Final_Result':
                items.add(f"Final_Result={value}")
            elif isinstance(value, (int, float)):
                low  = data[col].quantile(0.25)
                high = data[col].quantile(0.75)
                if value <= low:
                    items.add(f"{col}=Low")
                elif value >= high:
                    items.add(f"{col}=High")
                else:
                    items.add(f"{col}=Medium")
            else:
                items.add(f"{col}={value}")
        discretized.append(frozenset(items))
    return discretized

#-----------------------------------
# NEED EXPLANATION FOR THIS FUNCTION
#-----------------------------------
def generate_frequent_itemsets(transactions, min_support):
    """Standard Apriori level-by-level generation. Returns {frozenset: support}."""
    N = len(transactions)
    min_count = min_support * N

    # 1-itemsets
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1
            
    freq = {}
    current = []
    for item, cnt in item_counts.items():
        if cnt >= min_count:
            fs = frozenset([item])
            freq[fs] = cnt / N
            current.append(fs)
    print(f"  Size 1: {len(current)} frequent itemsets")

    k = 2
    while current:
        candidates = set()
        for i, a in enumerate(current):
            for b in current[i+1:]:
                union = a | b
                if len(union) == k:
                    candidates.add(union)
        next_freq = []
        for candidate in candidates:
            cnt = support_count(candidate, transactions)
            if cnt >= min_count:
                freq[candidate] = cnt / N
                next_freq.append(candidate)
        print(f"  Size {k}: {len(next_freq)} frequent itemsets")
        current = next_freq
        k += 1
    return freq

def generate_LKH(data, min_support, min_confidence, min_lift):
    """
    Generate association rules using Apriori.
    Only produces rules where the consequent is Final_Result.
    Args:
        data:            cleaned DataFrame
        min_support:     minimum support threshold  (0-1)
        min_confidence:  minimum confidence threshold (0-1)
        min_lift:        minimum lift threshold
    Returns:
        List of rule dicts sorted by lift descending.
    """
    print("Discretising data...")
    transactions = discretize_data(data)
    N = len(transactions)
    print(f"Transactions: {N}")
    
    print(f"Generating frequent itemsets (min_support={min_support})...")
    freq_itemsets = generate_frequent_itemsets(transactions, min_support)

    print(f"Generating rules (min_confidence={min_confidence}, min_lift={min_lift})...")
    rules = []

    for itemset, sup in freq_itemsets.items():
        if len(itemset) < 2:
            continue

        result_items = frozenset(i for i in itemset if i.startswith("Final_Result="))
        if not result_items:
            continue

        antecedent = itemset - result_items
        if not antecedent:
            continue

        count_ant  = support_count(antecedent, transactions)
        count_both = support_count(itemset, transactions)
        conf = count_both / count_ant if count_ant else 0

        sup_cons = support_count(result_items, transactions) / N
        lf = conf / sup_cons if sup_cons else 0

        if conf >= min_confidence and lf >= min_lift:
            rules.append({
                'antecedent':   antecedent,
                'consequent':   result_items,
                'Final_Result': next(iter(result_items)).split('=')[1],
                'support':      sup,
                'confidence':   conf,
                'lift':         lf,
            })
    rules.sort(key=lambda r: r['lift'], reverse=True)
    print(f"Strong rules found: {len(rules)}\n")
    return rules

def generate_atrisk_rules(data, min_support, min_confidence, min_lift):
    """
    Generate rules for at-risk students.
    Targets both Poor and Average results since at-risk includes both.
    Uses a lower min_support since at-risk students are a minority.
    """
    set_data = discretize_data(data)
    frequent_itemsets = generate_frequent_itemsets(set_data, min_support)
    
    # Target both Poor and Average as at-risk indicators
    target_consequents = [frozenset(["Final_Result=Poor"]), frozenset(["Final_Result=Average"])]
    rules = []

    for target_consequent in target_consequents:
        print(f"\n  Generating rules for: {next(iter(target_consequent))}")
        target_count = 0
        
        for itemset, sup in frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            # Only care about rules that conclude with this target
            if not target_consequent.issubset(itemset):
                continue

            antecedent = itemset - target_consequent
            if not antecedent:
                continue

            N = len(set_data)
            count_ant  = support_count(antecedent, set_data)
            count_both = support_count(itemset, set_data)
            conf = count_both / count_ant if count_ant else 0

            sup_cons = support_count(target_consequent, set_data) / N
            lf = conf / sup_cons if sup_cons else 0

            if conf >= min_confidence and lf >= min_lift:
                rules.append({
                    'antecedent': antecedent,
                    'consequent': target_consequent,
                    'confidence': conf,
                    'lift':       lf,
                    'support':    sup,
                })
                target_count += 1
        print(f"Found {target_count} rules")
    rules.sort(key=lambda r: r['lift'], reverse=True)
    return rules

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def cleaned_data_info(file_path=None):
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "Uni_Stu_ds_proj.csv")

    cleaned_df = preprocess_dataset(file_path)
    get_data_info(cleaned_df)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "Uni_Stu_cleaned.csv")
    cleaned_df.to_csv(out, index=False, encoding="utf-8", float_format='%.2f')
    print(f"Cleaned data saved -> {out}\n")
    return cleaned_df

def at_risk_data_info(cleaned_df):
    at_risk_df = identify_at_risk_students(cleaned_df)
    print(f"At-risk students: {len(at_risk_df)}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Uni_Stu_at_risk.csv")
    at_risk_df.to_csv(out, index=False, encoding="utf-8", float_format='%.2f')
    print(f"At-risk data saved -> {out}\n")
    return at_risk_df

def main():
    cleaned_df = cleaned_data_info()
    at_risk_df = at_risk_data_info(cleaned_df)

    rules = generate_LKH(
        cleaned_df,
        min_support=0.10,    
        min_confidence=0.40,
        min_lift=1.25,
    )

    rules_at_risk = generate_atrisk_rules(
        at_risk_df,
        min_support=0.08,
        min_confidence=0.35,
        min_lift=1.35,
    )

    if rules:
        print("=" * 60)
        print("STRONG ASSOCIATION RULES")
        print("=" * 60)
        for i, rule in enumerate(rules, 1):
            ant = " AND ".join(sorted(rule['antecedent']))
            con = next(iter(rule['consequent']))
            print(f"\nRule {i}:")
            print(f"  IF   {ant}")
            print(f"  THEN {con}")
            print(f"  Support={rule['support']:.2f}  "
                f"Confidence={rule['confidence']:.2f}  "
                f"Lift={rule['lift']:.2f}")
            print("-" * 60)
        print("="*60)
    else:
        print("No rules found — try lowering thresholds:")

    if rules_at_risk:
        print("\n" + "=" * 60)
        print("AT-RISK ASSOCIATION RULES")
        print("=" * 60)
        for i, rule in enumerate(rules_at_risk, 1):
            ant = " AND ".join(sorted(rule['antecedent']))
            con = next(iter(rule['consequent']))
            print(f"\nRule {i}:")
            print(f"  IF   {ant}")
            print(f"  THEN {con}")
            print(f"  Support={rule['support']:.2f}  "
                f"Confidence={rule['confidence']:.2f}  "
                f"Lift={rule['lift']:.2f}")
            print("-" * 60)
        print("="*60)
    else:
        print("No at-risk rules found — try lowering thresholds:")

if __name__ == "__main__":
    main()