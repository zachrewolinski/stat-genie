# imports
import random
from copy import deepcopy
from stat_genie.blade_pipeline.additions.perturbations.utils import read_json

class FeaturePerturbation:
    """
    Centralized object for perturbing feature names in JSON metadata. There
    are currently three supported actions:
    1. Anonymize variable names - replaces variable names with generic
       names like "feature1", "feature2", etc.
    2. Shuffle feature order - randomly shuffles the order of features
       in the metadata.
    3. Shuffle feature names - randomly shuffles the feature names among the
       features.
    If multiple types of feature perturbations are desired, their order will
    follow the order listed above.
    """
    
    def __init__(self, anonymize: bool = False,
                 shuffle_order: bool = False, shuffle_names: bool = False,
                 shuffle_order_seed: int = 42, shuffle_names_seed: int = 42):
        
        self.anonymize = anonymize
        self.shuffle_order = shuffle_order
        self.shuffle_names = shuffle_names
        self.shuffle_order_seed = shuffle_order_seed
        self.shuffle_names_seed = shuffle_names_seed

    def anonymize_variable_names(self, json_metadata: dict) -> None:
        """
        Takes a JSON (in dictionary format) and replaces variable names 
        with non-descriptive names like "feature1", "feature2", etc.
        
        Args:
            json_metadata: Dictionary containing the JSON metadata
            
        Returns:
            Nothing, modifies self.json_metadata in place
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

    def shuffle_feature_order(self, json_metadata: dict) -> None:
        """
        Shuffles the order of features in the JSON metadata.
        
        Args:
            json_metadata: Dictionary containing the JSON metadata
            seed: Random seed for reproducibility
        
        Returns:
            Nothing, modifies self.json_metadata in place
        """
        
        # create a copy of the metadata to work with
        json_metadata = deepcopy(json_metadata)
        
        # set random seed for reproducibility
        random.seed(self.shuffle_order_seed)
        
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

    def shuffle_feature_names(self, json_metadata: dict) -> None:
        """
        Randomly shuffles the feature names in the JSON metadata.
        
        Args:
            json_metadata: Dictionary containing the JSON metadata
            seed: Random seed for reproducibility
        
        Returns:
            Nothing, modifies self.json_metadata in place
        """
        
        # create a copy of the metadata to work with
        json_metadata = deepcopy(json_metadata)
        
        # set random seed for reproducibility
        random.seed(self.shuffle_names_seed)
        
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
    
    def perturb(self, json_path: str) -> dict:
        """
        Applies the selected perturbations to the JSON metadata in the
        following order:
        1. Anonymize variable names
        2. Shuffle feature order
        3. Shuffle feature names
        
        Returns:
            The perturbed JSON metadata as a dictionary
        """
        
        # read in json metadata
        json_metadata = read_json(json_path)
        
        if self.anonymize:
            json_metadata = self.anonymize_variable_names(json_metadata)
        
        if self.shuffle_order:
            json_metadata = self.shuffle_feature_order(json_metadata)
        
        if self.shuffle_names:
            json_metadata = self.shuffle_feature_names(json_metadata)
        
        return json_metadata