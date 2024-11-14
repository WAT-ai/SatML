import pandas as pd

def get_easy_ids(file_path):
    """
    Reads a CSV file and returns a list of IDs where the 'difficulty' column is 'easy'.
    
    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of IDs where the difficulty is 'easy'.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Filter rows where 'difficulty' is 'easy' (case-insensitive)
        easy_ids = df.loc[df['difficulty'].str.lower().str.strip() == 'easy', 'id'].tolist()
        
    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")
    
    return easy_ids