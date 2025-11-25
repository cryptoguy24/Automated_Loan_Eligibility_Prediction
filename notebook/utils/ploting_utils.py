# Filename: bivariate_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate Plotting Utilities
# ----------------------------
# Numerical Univariate Plotting Utility
# --------------------------------------
def plot_numerical_univariate(df, numeric_cols, n_cols=2, figsize_per_subplot=(5,4), save_path=None):
    """
    Plots vertical histograms of numerical columns with skewness annotated, arranged in symmetrical subplots.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing numeric columns.
    numeric_cols : list
        List of numeric columns to plot.
    n_cols : int, default 2
        Number of subplots per row.
    figsize_per_subplot : tuple, default (5,4)
        Width and height per subplot.
    save_path : str or None
        Folder path to save the figure. If None, figure is not saved.
    """

    # Auto-calculate number of rows (YOUR ADDED LOGIC)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  

    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_subplot[0] * n_cols,
                                                      figsize_per_subplot[1] * n_rows))
    
    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot histograms
    for ax, col in zip(axes, numeric_cols):
        skewness = df[col].skew()
        sns.histplot(df[col], kde=True, ax=ax)  # removed fixed color to keep it flexible
        ax.set_title(f"{col} | Skewness: {skewness:.5f}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    # Remove empty subplots to keep symmetry
    for ax in axes[len(numeric_cols):]:
        ax.remove()

    # Title for whole figure
    plt.suptitle('Behavior of Numeric Variables in Loan Applications', fontsize=20)

    plt.tight_layout()

    # Save option
    if save_path:
        plt.savefig(f"{save_path}/numerical_univariate_histograms.png", dpi=300)

    plt.show()





# Bivariate Plotting Utilities
# ----------------------------
# Numerical vs Numerical Plotting Utility
# --------------------------------------

def plot_numerical_vs_numerical(df, numerical_cols, 
                                figsize_per_subplot=(5, 4), 
                                add_corr=True, save_path=None):
    """
    Plots scatterplots for pairwise numerical column comparisons with automatic symmetrical layout.
    """

    from itertools import combinations
    import math

    # Generate all unique numeric pairs
    num_pairs = list(combinations(numerical_cols, 2))
    total_plots = len(num_pairs)

    # --- AUTOMATIC SYMMETRICAL GRID ---
    n_cols = int(math.ceil(math.sqrt(total_plots)))
    n_rows = int(math.ceil(total_plots / n_cols))

    # Create subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows)
    )
    axes = axes.flatten()

    # Plot each pair
    for ax, (x_col, y_col) in zip(axes, num_pairs):
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        sns.regplot(x=df[x_col], y=df[y_col], scatter=False, ax=ax)

        ax.set_title(f'{x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        if add_corr:
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            ax.text(0.05, 0.9, f'Corr: {corr:.3f}', transform=ax.transAxes, fontsize=9)

    # Remove unused subplots
    for ax in axes[total_plots:]:
        ax.remove()

    plt.suptitle('Numerical vs Numerical Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(f'{save_path}/numerical_vs_numerical.png', dpi=300)

    plt.show()


# Numerical vs Categorical Plotting Utility
# --------------------------------------


def plot_numerical_vs_categorical(df, numerical_cols, categorical_cols, 
                                  n_cols=None, figsize_per_subplot=(5,4), 
                                  horizontal=True, save_path=None):

    import math

    # Auto-select n_cols for symmetry
    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(len(numerical_cols))))

    orient = 'h' if horizontal else 'v'

    for cat_col in categorical_cols:

        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(figsize_per_subplot[0]*n_cols, figsize_per_subplot[1]*n_rows)
        )
        axes = axes.flatten()

        for ax, num_col in zip(axes, numerical_cols):
            sns.boxplot(
                x=num_col if horizontal else cat_col, 
                y=cat_col if horizontal else num_col, 
                data=df,
                orient=orient,
                ax=ax
            )
            ax.set_title(f'{num_col} vs {cat_col}')

        # Remove empty plots for symmetry
        for ax in axes[len(numerical_cols):]:
            ax.remove()

        plt.suptitle(f'Numerical vs Categorical Analysis: {cat_col}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(f"{save_path}/{cat_col}_numerical_vs_categorical.png", dpi=300)

        plt.show()


# Categorical vs Categorical Plotting Utility
# --------------------------------------

def plot_categorical_vs_categorical(df, cat_cols, hue_col=None,
                                    n_cols=2, figsize_per_plot=(5,4), save_path=None):
    """
    Plots countplots for multiple categorical columns using symmetrical subplots.
    """

    # Auto number of rows
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols

    # Figure & axes
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_plot[0] * n_cols,
                                      figsize_per_plot[1] * n_rows))
    axes = axes.flatten()  # make iterable

    # Plot each categorical column
    for ax, col in zip(axes, cat_cols):
        sns.countplot(x=col, hue=hue_col, data=df, ax=ax)
        ax.set_title(f"{col} vs {hue_col}" if hue_col else f"{col} Countplot")

        # Rotate xticks if too many categories
        if df[col].nunique() > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Remove unused axes for symmetry
    for ax in axes[len(cat_cols):]:
        ax.remove()

    plt.tight_layout()
    plt.suptitle("Categorical vs Categorical Analysis", fontsize=16, y=1.02)

    # Save if requested
    if save_path:
        plt.savefig(f"{save_path}/categorical_vs_categorical_subplots.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\yashd\OneDrive\Desktop\DS_projects\earthquake_data_tsunami\data\raw\earthquake_data_tsunami.csv')
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    plot_numerical_univariate(df, num_cols, n_cols=3, figsize_per_subplot=(5,4), save_path=None)
    pass