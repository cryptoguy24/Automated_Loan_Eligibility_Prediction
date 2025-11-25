import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr, f_oneway
from itertools import combinations, product
from typing import List, Tuple, Optional
# -------------------------------------------------------------


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of the input DataFrame including total rows, 
    unique values, and null counts for each column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame with the following columns:
        - 'columns': The name of the column.
        - 'total_counts': The total number of rows in the column.
        - 'unique_counts': The number of unique values in the column.
        - 'null_counts': The number of missing (null) values in the column.
    """

    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    print('-'*100)

    cols = df.columns.tolist()
    total_rows = df.shape[0]
    print(f'There are {UNDERLINE}{BOLD}{total_rows}{END} rows with {UNDERLINE}{BOLD}{len(cols)}{END} columns in the DataFrame.')

    print('-'*100)    

    data = []
    for col in cols:
        data.append([col, (~df[col].isna()).sum(), df[col].nunique(), df[col].isna().sum()])

    return pd.DataFrame(data, columns=['columns', 'total_counts', 'unique_counts', 'null_counts'])


    # --------------------------------------------------------------------


def non_float_values(df: pd.DataFrame, column: str):
    """
    Identify non-float values in a specific column of a DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    column : str
        Name of the column to check for non-float values.

    Returns:
    -------
    numpy.ndarray
        An array of unique values from the specified column that 
        cannot be converted to float (excluding the literal string 'None').

    Notes:
    ------
    - The column is converted to string and stripped of whitespace before checking.
    - Real NaN/None values become the string 'None' after conversion,
      so 'None' is filtered out from the final result.
    """

    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False

    # Convert the column to string and strip surrounding whitespace
    col = df[column].astype(str).str.strip()

    # Identify all values that are NOT valid floats
    non_numeric = col[~col.apply(is_float)]

    # Filter out the literal string 'None' (because real None/NaN became the string "None")
    return non_numeric[non_numeric != 'None'].unique()

# --------------------------------------------------------------------------------------------------

def get_significant_pairs(
                            df: pd.DataFrame, 
                            numerical_cols: Optional[List[str]] = None,
                            categorical_cols: Optional[List[str]] = None,
                            alpha: float = 0.05,
                            verbose: bool = False
                        ) -> List[Tuple[str, str]]:
    """
    # Assuming you already separated your columns list
    numerical_cols = [col for col in numeric_cols if col not in ['Loan_ID']]
    categorical_cols = [col for col in categorical_cols if col not in ['Loan_ID']]

    # Just pass them in! The function handles the IDs and the pairing logic.
    sig_pairs = get_significant_pairs(df, numerical_cols, categorical_cols)

    # The function will print a big table, and 'sig_pairs' will contain
    # the list you need for your plotting loops.
    Automatically tests bivariate relationships between columns to find significant associations.
    
    It performs:
    1. Chi-Square Test for Categorical vs Categorical.
    2. Pearson Correlation for Numerical vs Numerical.
    3. ANOVA (One-way) for Numerical vs Categorical.

    It automatically filters out Primary Key/ID columns (where all values are unique).

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data.
    numerical_cols : List[str], optional
        List of numerical column names. 
    categorical_cols : List[str], optional
        List of categorical column names.
    alpha : float, optional (default=0.05)
        The p-value threshold. If p < alpha, the relationship is considered significant.

    Returns:
    --------
    List[Tuple[str, str]]
        A list of pairs (col1, col2) that showed a statistically significant relationship.
    """
    
    significant_pairs = []
    numerical_cols = numerical_cols if numerical_cols else []
    categorical_cols = categorical_cols if categorical_cols else []
    
    # --- 1. Filter out ID columns (High Cardinality) ---
    clean_num = []
    for col in numerical_cols:
        if df[col].nunique() == df.shape[0]:
            if verbose:
                print(f"Ignoring '{col}': Looks like a Primary Key/ID.")
        else:
            clean_num.append(col)
            
    clean_cat = []
    for col in categorical_cols:
        if df[col].nunique() == df.shape[0]:
            if verbose:
                print(f"Ignoring '{col}': Looks like a Primary Key/ID.")
        else:
            clean_cat.append(col)

    if verbose:
        print(f"\n{'Column A':<20} | {'Column B':<20} | {'Test Type':<10} | {'P-Value':<8} | {'Result'}")
        print("-" * 85)

    # --- 2. CAT vs CAT (Chi-Square) ---
    if len(clean_cat) > 1:
        for col1, col2 in combinations(clean_cat, 2):
            crosstab = pd.crosstab(df[col1], df[col2])
            chi2, p, _, _ = chi2_contingency(crosstab)
            
            if p < alpha:
                significant_pairs.append((col1, col2))
                if verbose:
                    print(f"{col1:<20} | {col2:<20} | {'Chi-Sq':<10} | {p:.4f}   | Related")

    # --- 3. NUM vs NUM (Pearson Correlation) ---
    if len(clean_num) > 1:
        for col1, col2 in combinations(clean_num, 2):
            # Drop NaNs for correlation
            temp_df = df[[col1, col2]].dropna()
            if len(temp_df) < 2: continue # Skip if empty
            
            corr, p = pearsonr(temp_df[col1], temp_df[col2])
            
            if p < alpha:
                significant_pairs.append((col1, col2))
                if verbose:
                    print(f"{col1:<20} | {col2:<20} | {'Pearson':<10} | {p:.4f}   | Related")

    # --- 4. NUM vs CAT (ANOVA) ---
    if len(clean_num) > 0 and len(clean_cat) > 0:
        # Product creates pairs between two different lists
        for cat_col, num_col in product(clean_cat, clean_num):
            # Group the numeric data by the categorical column
            groups = [df[df[cat_col] == group][num_col].dropna() for group in df[cat_col].unique()]
            
            # ANOVA requires at least 2 groups with data
            clean_groups = [g for g in groups if len(g) > 0]
            
            if len(clean_groups) < 2:
                continue

            f_stat, p = f_oneway(*clean_groups)
            
            if p < alpha:
                significant_pairs.append((cat_col, num_col))
                if verbose:
                    print(f"{cat_col:<20} | {num_col:<20} | {'ANOVA':<10} | {p:.4f}   | Related")

    if verbose:
        print(f"\nTotal Significant Pairs Found: {len(significant_pairs)}")
    return significant_pairs

