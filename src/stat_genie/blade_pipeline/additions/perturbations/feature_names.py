# imports
import json
import random
from copy import deepcopy

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

def anonymize_variable_names(json_metadata):
    """
    Takes a JSON (in dictionary format) and replaces variable names 
    with non-descriptive names like "feature1", "feature2", etc.
    
    Args:
        json_metadata: Dictionary containing the JSON metadata
        
    Returns:
        Dictionary with anonymized variable names
    """
    
    # create a copy of the metadata to work with
    json_metadata = deepcopy(json_metadata)
    
    # get all unique variable names from fields
    if "data_desc" in json_metadata and "fields" in json_metadata["data_desc"]:
        variable_names = [field["column"] for field in json_metadata["data_desc"]["fields"]]
    else:
        # fallback: try to get from field_names if fields structure is different
        variable_names = json_metadata.get("data_desc", {}).get("field_names", [])
    
    # create mapping from old names to new names (feature1, feature2, etc.)
    name_mapping = {old_name: f"feature{i+1}" for i, old_name in enumerate(variable_names)}
    
    # replace column names in fields
    if "data_desc" in json_metadata and "fields" in json_metadata["data_desc"]:
        for field in json_metadata["data_desc"]["fields"]:
            if "column" in field and field["column"] in name_mapping:
                field["column"] = name_mapping[field["column"]]
    
    # replace names in field_names array
    if "data_desc" in json_metadata and "field_names" in json_metadata["data_desc"]:
        json_metadata["data_desc"]["field_names"] = [
            name_mapping.get(old_name, old_name) 
            for old_name in json_metadata["data_desc"]["field_names"]
        ]
    
    return json_metadata

def shuffle_feature_order(json_metadata, seed=42):
    """
    Shuffles the order of features in the JSON metadata.
    
    Args:
        json_metadata: Dictionary containing the JSON metadata
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with shuffled feature order
    """
    
    # create a copy of the metadata to work with
    json_metadata = deepcopy(json_metadata)
    
    # set random seed for reproducibility
    random.seed(seed)
    
    # shuffle the fields array if it exists
    if "data_desc" in json_metadata and "fields" in json_metadata["data_desc"]:
        fields = json_metadata["data_desc"]["fields"]
        random.shuffle(fields)
        json_metadata["data_desc"]["fields"] = fields
    
    # shuffle the field_names array if it exists
    if "data_desc" in json_metadata and "field_names" in json_metadata["data_desc"]:
        field_names = json_metadata["data_desc"]["field_names"]
        random.shuffle(field_names)
        json_metadata["data_desc"]["field_names"] = field_names
    
    return json_metadata

def shuffle_feature_names(json_metadata, seed=42):
    """
    Randomly shuffles the feature names in the JSON metadata.
    
    Args:
        json_metadata: Dictionary containing the JSON metadata
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with shuffled feature names
    """
    
    # create a copy of the metadata to work with
    json_metadata = deepcopy(json_metadata)
    
    # set random seed for reproducibility
    random.seed(seed)
    
    # get all variable names from fields
    if "data_desc" in json_metadata and "fields" in json_metadata["data_desc"]:
        variable_names = [field["column"] for field in json_metadata["data_desc"]["fields"]]
    else:
        # fallback: try to get from field_names if fields structure is different
        variable_names = json_metadata.get("data_desc", {}).get("field_names", [])
    
    # create a shuffled copy of the names
    shuffled_names = variable_names.copy()
    random.shuffle(shuffled_names)
    
    # create mapping from original names to shuffled names
    name_mapping = {old_name: new_name for old_name, new_name in zip(variable_names, shuffled_names)}
    
    # replace column names in fields
    if "data_desc" in json_metadata and "fields" in json_metadata["data_desc"]:
        for field in json_metadata["data_desc"]["fields"]:
            if "column" in field and field["column"] in name_mapping:
                field["column"] = name_mapping[field["column"]]
    
    # replace names in field_names array
    if "data_desc" in json_metadata and "field_names" in json_metadata["data_desc"]:
        json_metadata["data_desc"]["field_names"] = [
            name_mapping.get(old_name, old_name) 
            for old_name in json_metadata["data_desc"]["field_names"]
        ]
    
    return json_metadata
    