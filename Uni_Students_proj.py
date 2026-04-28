import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
            if any(kw.lower() in c.lower() for kw in keywords)]
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

    print("\nCleaning negative values")
    data = fix_negative_values(data)
    if handle_missing:
        print("\n[Step 1] Handling missing data")
        data = handle_missing_data(data)
    if handle_dups:
        print("\n[Step 2] Handling duplicates")
        data = handle_duplicates(data)
    if handle_inconsistent:
        print("\n[Step 3] Standardizing inconsistent data")
        data = handle_inconsistent_data(data)
    if handle_outliers_flag:
        print("\n[Step 4] Handling outliers (IQR)")
        data = handle_outliers(data, columns=outlier_columns)

    print(f"\n{'='*60}")
    print(f"Final shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print("\nBasic statistics:")
    print(data.describe())
    if 'Final_Result' in data.columns:
        print("\nAVERAGES BY FINAL RESULT")
        print(f"{'='*60}")
        num_cols, _ = detect_column_types(data)
        result_order = ['Excellent', 'Good', 'Average', 'Poor']
        present_results = [r for r in result_order if r in data['Final_Result'].values]
        averages = data.groupby('Final_Result')[num_cols].mean()
        for result in present_results:
            print(f"\n  [{result}]  (n={len(data[data['Final_Result'] == result])})")
            for col in num_cols:
                print(f"    {col:<30} {averages.loc[result, col]:.2f}")
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
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    thresholds = {col: df[col].quantile(0.25) for col in numeric_cols}
    
    at_risk_mask = df['Final_Result'] == 'Poor'
    
    # Check if average students have weak indicators
    avg_mask = df['Final_Result'] == 'Average'
    if avg_mask.any():
        weak_counts = (df[numeric_cols] <= pd.Series(thresholds)).sum(axis=1)
        avg_students_atrisk = avg_mask & (weak_counts >= len(numeric_cols) // 2)
        at_risk_mask = at_risk_mask | avg_students_atrisk
    
    # Extract only risky students
    at_risk_df = df[at_risk_mask].copy()

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
    quantiles = {}
    for col in data.columns:
        if col != 'Final_Result' and isinstance(data[col].iloc[0] if len(data) > 0 else 0, (int, float)):
            quantiles[col] = {
                'low': data[col].quantile(0.25),
                'high': data[col].quantile(0.75)
            }
    discretized = []
    for idx, row in data.iterrows():
        items = set()
        for col in data.columns:
            value = row[col]
            if col == 'Final_Result':
                items.add(f"Final_Result={value}")
            elif col in quantiles:
                low = quantiles[col]['low']
                high = quantiles[col]['high']
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
        min_support:     minimum support threshold
        min_confidence:  minimum confidence threshold
        min_lift:        minimum lift threshold
    Returns:
        List of rule dicts sorted by lift descending.
    """
    print("Discrettizing data...")
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
    cleaned_df = cleaned_df.sort_values(by='Final_Result')
    get_data_info(cleaned_df)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "Uni_Stu_cleaned.csv")
    cleaned_df.to_csv(out, index=False, encoding="utf-8", float_format='%.2f')
    return cleaned_df

def at_risk_data_info(cleaned_df):
    at_risk_df = identify_at_risk_students(cleaned_df)
    at_risk_df = at_risk_df.sort_values(by='Final_Result')
    print(f"At-risk students: {len(at_risk_df)}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Uni_Stu_at_risk.csv")
    at_risk_df.to_csv(out, index=False, encoding="utf-8", float_format='%.2f')
    return at_risk_df

# ─────────────────────────────────────────────
# VISUALIZATION & CLASSIFICATION
# ─────────────────────────────────────────────

def create_visualizations(cleaned_df):
    """Create visualizations: distribution, correlation heatmap, etc."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Final_Result Distribution
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Plot 1: Final Result distribution
        cleaned_df['Final_Result'].value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Distribution of Final Results', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Final Result')
        axes[0, 0].set_ylabel('Count')
        
        # Plot 2: Attendance distribution (histogram)
        if 'Attendance' in cleaned_df.columns:
            axes[1, 0].hist(cleaned_df['Attendance'], bins=20, color='lightgreen', edgecolor='black')
            axes[1, 0].set_title('Attendance Distribution', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Attendance %')
            axes[1, 0].set_ylabel('Frequency')
        
        # Plot 3: Test Score distribution
        if 'Test_Score' in cleaned_df.columns:
            axes[1, 1].hist(cleaned_df['Test_Score'], bins=20, color='coral', edgecolor='black')
            axes[1, 1].set_title('Test Score Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Test Score')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        viz_file = os.path.join(script_dir, "visualizations.png")
        plt.savefig(viz_file, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {viz_file}")
    except Exception as e:
        print(f"  ✗ Visualization error: {e}")

def classify_students(cleaned_df):
    """Compare multiple classification algorithms: Decision Tree, KNN, Naive Bayes."""
    print("\n" + "="*60)
    print("CLASSIFICATION ALGORITHMS COMPARISON")
    print("="*60)
    
    try:
        # Prepare data
        X = cleaned_df.select_dtypes(include=[np.number]).drop('Final_Result', axis=1, errors='ignore')
        y = cleaned_df['Final_Result']
        
        # Encode target labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        classifiers = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        }
        
        results = []
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'Algorithm': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
            })
            print(f"\n  [{name}]")
            print(f"    Accuracy:  {accuracy:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall:    {recall:.4f}")
            print(f"    F1-Score:  {f1:.4f}")
        
        # Export results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_file = os.path.join(script_dir, "classification_results.csv")
        pd.DataFrame(results).to_csv(results_file, index=False, encoding="utf-8")
        print(f"\n  ✓ Classification results saved to: {results_file}")
        
    except Exception as e:
        print(f"  ✗ Classification error: {e}")


def main():
    cleaned_df = cleaned_data_info()
    at_risk_df = at_risk_data_info(cleaned_df)
    
    rules = generate_LKH(
        cleaned_df,
        min_support=0.10,    
        min_confidence=0.40,
        min_lift=1.20,
    )

    rules_at_risk = generate_atrisk_rules(
        at_risk_df,
        min_support=0.08,
        min_confidence=0.45,
        min_lift=1.20,
    )

    if rules:
        print("=" * 60)
        print("STRONG ASSOCIATION RULES")
        print("=" * 60)
        print("-" * 60)
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
        print("-" * 60)
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
    
    # Export rules to CSV
    print("\n" + "="*60)
    print("EXPORTING RULES TO FILES")
    print("="*60)
    
    create_visualizations(cleaned_df)
    
    classify_students(cleaned_df)

if __name__ == "__main__":
    main()