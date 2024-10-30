import csv

def get_easy_ids(file_path):
    """
    Reads a CSV file and returns a list of IDs where the 'difficulty' column is 'easy'.
    
    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of IDs where the difficulty is 'easy'.
    """
    easy_ids = []
    try:
        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['difficulty'].strip().lower() == 'easy':
                    easy_ids.append(row['id'])
    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")
    
    return easy_ids