import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import os

def convert_to_float(df, column_name):
    """
    Converts a column in a pandas DataFrame to float type.
    
    Parameters:
    - df : pd.DataFrame
        The DataFrame containing the column to convert.
    - column_name : str
        The name of the column to convert.
    
    Returns:
    - pd.DataFrame
        The DataFrame with the column converted to float.
    """
    try:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df
    except KeyError:
        raise KeyError(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")
    except Exception as e:
        raise Exception(f"Erreur lors de la conversion : {e}")
    

def preprocess_data(df):
    """
    Preprocess the input DataFrame by performing several data cleaning and transformation steps.
    This function performs the following operations:
    1. Creates a copy of the input DataFrame.
    2. Converts specified columns to float type.
    3. Drops unwanted columns.
    4. Removes features with high correlation.
    5. Identifies and drops columns with mixed data types.
    6. Fills missing values.
    7. Splits a date column into multiple components.
    8. Normalizes and encodes features.
    Parameters:
    df (pandas.DataFrame): The input DataFrame to preprocess.
    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """

    df = df.copy()
    for column_name in ["insee_%_agri", "insee_med_living_level", "insee_%_ind", "insee_%_const"]:
        convert_to_float(df,column_name)
    # find columns with mixt types (int and string)
    drop_unwanted_columns(df)
    remove_high_correlation_features(df)
    mixed_type_columns = [
        col for col in df.columns 
        if df[col].apply(type).nunique() > 1
    ]
    print(f"Columns with mixed types: {mixed_type_columns}")
    df.drop(columns=mixed_type_columns, inplace=True)
    fill_missing_values(df)
    split_date_column(df,'piezo_measurement_date')
    df = normalize_and_encode_features(df)

    return df

def normalize_and_encode_features(df):
    """
    Normalize numeric features and encode categorical features in a DataFrame.
    Parameters:
    df (pd.DataFrame): Input DataFrame with numeric and categorical features.
    Returns:
    pd.DataFrame: DataFrame with normalized numeric features and encoded categorical features.
    """

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Initialisez des objets pour la normalisation et l'encodage ordinal
    scaler = MinMaxScaler()
    ordinal_encoder = OrdinalEncoder()

    # Normalisation des colonnes numériques
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encodage des colonnes catégoriques en fonction du nombre de valeurs uniques
    for col in categorical_cols:
        unique_values = df[col].nunique()
        
        if unique_values <= 30:
            # One-Hot Encoding pour moins de 30 catégories uniques
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            # Encodage ordinal ou naturel pour plus de 30 catégories uniques
            df[col] = ordinal_encoder.fit_transform(df[[col]])
    return df

def split_date_column(df, column_name):
    """
    Splits a date column in the DataFrame into separate year, month, and day columns.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the date column.
    column_name (str): The name of the date column to split.
    Returns:
    None: The DataFrame is modified in place.
    """

    df.loc[:, [column_name+'_year', column_name+'_month', column_name+'_day']] = df[column_name].str.split('-', expand=True).astype('Int32').rename(columns={0: 'year', 1: 'month', 2: 'day'})
    df.drop(columns=[column_name], inplace=True)

def drop_unwanted_columns(df):
    """
    Remove predefined unwanted columns from the DataFrame if they exist.
    Parameters:
    df (pandas.DataFrame): The DataFrame from which to drop columns.
    Returns:
    None: The function modifies the DataFrame in place.
    """

    UselessFeatures = ["piezo_station_commune_name","piezo_station_department_name","piezo_station_bss_code","piezo_station_pe_label","piezo_station_bdlisa_codes","piezo_producer_name","piezo_measure_nature_name","piezo_station_pe_label", "meteo_name", "hydro_status_label", "hydro_method_label", "hydro_method_label"]
    #remove UselessFeatures that are in the dataset
    UselessFeatures = [col for col in UselessFeatures if col in df.columns]
    df.drop(columns=UselessFeatures, inplace = True)

def remove_high_correlation_features(df):
    """
    Remove features from the DataFrame that have a high correlation with other features.
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    Returns:
    None: The function modifies the DataFrame in place.
    """

    numeric_data = df.select_dtypes(include=[np.number])

    corr_matrix = numeric_data.corr().abs() 

    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool) 
    high_corr_pairs = corr_matrix.where(upper_triangle).stack().reset_index()
    high_corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']

    threshold = 0.9
    redundant_features = high_corr_pairs[high_corr_pairs['Correlation'] > threshold]['Feature2'].unique()

    df.drop(columns=redundant_features, inplace=True)

def fill_missing_values(df):
    """
    Fills missing values in a DataFrame.
    For numeric columns, fills missing values with the median.
    For categorical columns, fills missing values with the mode.
    Prints columns with remaining missing values, if any.
    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    Returns:
    None: The function modifies the DataFrame in place.
    """

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    #check if there is still a na
    if df.isna().sum().sum() > 0:
        #find columns with na
        cols_with_na = df.columns[df.isna().any()].tolist()
        print(f"Columns with missing values: {cols_with_na}")
    else:
        print("No missing values in the dataset")

def save_to_csv(model, X_test,mapping_int_to_string,row_index_test,path):
    predictions = [mapping_int_to_string[pred] for pred in model.predict(X_test)]
    df_predictions = pd.DataFrame(
        predictions, 
        index=row_index_test,  # Set the index
        columns=["piezo_groundwater_level_category"]  # Set the column name
    )
    df_predictions.to_csv(get_csv_path(path))

def get_csv_path(path):
    #if path is a csv return it, otherwist add .csv to the path
    if not path.endswith('.csv'):
        path= path + '.csv'
    # if path is not in output folder, add it
    if not path.startswith('output/'):
        path = 'output/' + path
    #if csv allready exists, add a number to the end of the path, starting from adding nothing
    updated_path = path 
    i = 1
    while os.path.exists(updated_path):
        updated_path = path.replace('.csv',f'_({i}).csv')
        i += 1
    return updated_path

def split_train_test(datasets,  date_column, split):
    """
    Splits the dataset into training and test sets based on a date range.

    Parameters:
    datasets array of (pd.DataFrame): The datasets to split.
    date_column (pd.DataFrame): The column containing datetime values.
    split (str,str): The start date of the test set (inclusive) and  The end date of the test set (inclusive)

    Returns:
    an array of (pd.DataFrame): The training and test sets.
    """
    # Get the start and end dates for the test set
    start_date, end_date = split

    # find the indexes of the train and test sets
    train_indexes = date_column.index[(date_column < start_date) | (date_column > end_date)]
    test_indexes = date_column.index[(date_column >= start_date) & (date_column <= end_date)]

    # return a tuple of size 2 * len(datasets) containing the train and test sets
    output= [[dataset.loc[train_indexes], dataset.loc[test_indexes]] for dataset in datasets]
    flatten_output=[j for i in output for j in i]
    return tuple(flatten_output)
