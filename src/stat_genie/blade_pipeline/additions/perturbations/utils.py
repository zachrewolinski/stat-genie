# imports
import json


def read_json(json_path):
    """
    Reads a JSON file from the given file path and returns the parsed metadata.
    
    Args:
        json_path: Path to the JSON file to read
        
    Returns:
        Dictionary or list containing the parsed JSON metadata
    """
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata