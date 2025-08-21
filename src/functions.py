
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


def exclude_outliers_by_trend(df, group_col, value_col,sort_by='FLOPs', window_size=5, num_std=2):
    """
    Excludes outliers from a specified value column within groups based on the moving average trend.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): The name of the column to group the data by (e.g., 'source').
        value_col (str): The name of the column containing the values to check for trend outliers (e.g., 'HEF').
        window_size (int): The window size for the moving average.
        num_std (float): The number of standard deviations to use as the threshold for outlier detection.

    Returns:
        pd.DataFrame: A new DataFrame with outliers excluded based on the trend within each group.
    """
    df_processed = []
    if group_col!='':
        for group_name, group_data in df.groupby(group_col):
            group_sorted = group_data.sort_values(by=sort_by).copy()  # Assuming 'FLOPs' is the sequence column
    
            group_sorted.loc[:, 'rolling_mean'] = group_sorted[value_col].rolling(window=window_size, center=True).mean()
            group_sorted.loc[:, 'difference'] = np.abs(group_sorted[value_col] - group_sorted['rolling_mean'])
            std_dev_diff = group_sorted['difference'].std()
            threshold = num_std * std_dev_diff
    
            trend_following_group = group_sorted[group_sorted['difference'] <= threshold].copy()
            df_processed.append(trend_following_group)
    
        df_filtered = pd.concat(df_processed, ignore_index=True)
    else:
        group_sorted = df.sort_values(by=sort_by).copy()  # Assuming 'FLOPs' is the sequence column
        group_sorted.loc[:, 'rolling_mean'] = group_sorted[value_col].rolling(window=window_size, center=True).mean()
        group_sorted.loc[:, 'difference'] = np.abs(group_sorted[value_col] - group_sorted['rolling_mean'])
        std_dev_diff = group_sorted['difference'].std()
        threshold = num_std * std_dev_diff

        trend_following_group = group_sorted[group_sorted['difference'] <= threshold].copy()
        df_processed.append(trend_following_group)

        df_filtered = pd.concat(df_processed, ignore_index=True)
    return df_filtered

# Example custom function to model exponential decay with saturation



def exponential_decay(x, L, k,d):
    return L * (1 - np.exp(-k * x**d))

# Custom plotting function to be used with FacetGrid

def plot_exponential_decay_fit(data, x_col, y_col, **kwargs):
    source=data['source'].values[-1]
    op=data['component'].values[-1]
    # Extract x and y from the dataframe subset
    x = data[x_col].values
    y = data[y_col].values
    initial_guess = [60, 1,5]
    # Fit the custom exponential function to this subset
    popt, _ = curve_fit(exponential_decay, x, y, p0=initial_guess, maxfev=100000)
    L, k,d=popt[0],popt[1],popt[2]    

    # Plot the raw data
    plt.scatter(x*1e12, y, **kwargs)
    # Plot the fitted curve (using the same x values)
    fitted_y = exponential_decay(x, *popt)

    plt.scatter(x*1e12, fitted_y, color="red", marker='x',
            label=rf"$\eta_h = {L:.2f} \cdot \left(1 - e^{{-{k:.2f} \cdot C^{{{d:.2f}}}}}\right)$")

    #######################################################################################
    # Add legend if desired (each Facet gets its own legend)
    plt.legend()
    print(f"Source={source}, coef {op}, :{popt}")
    return popt

def remove_outliers(df: pd.DataFrame, column_name: str,) -> pd.DataFrame:
    """
    Exclut les valeurs extrêmes (outliers) d'une colonne spécifiée dans un DataFrame Pandas.

    Args:
        df (pd.DataFrame): Le DataFrame Pandas contenant les données.
        column_name (str): Le nom de la colonne (variable) sur laquelle appliquer la détection des outliers.
                           Cette colonne doit contenir des données numériques.
    Returns:
        pd.DataFrame: Un nouveau DataFrame Pandas avec les lignes contenant les outliers
                      dans la colonne spécifiée supprimées.

    Raises:
        ValueError: Si la colonne spécifiée n'existe pas, ne contient pas de données numériques,
                    ou si une méthode non reconnue est spécifiée.
    """
    if column_name not in df.columns:
        raise ValueError(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")

    data_series = df[column_name]

    if not pd.api.types.is_numeric_dtype(data_series):
        raise ValueError(f"La colonne '{column_name}' doit contenir des données numériques.")

    original_len = len(df)
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtrer le DataFrame en entier basé sur les limites de la colonne spécifiée
    filtered_df = df[(data_series >= lower_bound) & (data_series <= upper_bound)]
    print(f"Méthode IQR: Limite inférieure={lower_bound:.2f}, Limite supérieure={upper_bound:.2f}")


    print(f"Nombre de lignes originales: {original_len}")
    print(f"Nombre de lignes après exclusion des outliers: {len(filtered_df)}")
    print(f"Nombre de lignes exclues (outliers): {original_len - len(filtered_df)}")

    return filtered_df
