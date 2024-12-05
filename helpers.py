def read_labels(file_path: str): 
    """
    helper for load labels.txt
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [lines.strip() for lines in f.readlines()]
        