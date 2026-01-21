# imports
import os
import random
import pandas as pd
import numpy as np
from copy import deepcopy

from stat_genie.blade_pipeline.additions.perturbations.utils import read_json
from stat_genie.blade_pipeline.utils import (
    get_dataset_info_path,
    get_dataset_csv_path,
    list_datasets,
)

class FeaturePerturbation:
    """
    Centralized object for perturbing features in JSON metadata. There
    are currently four supported actions:
    1. Add random feature to the dataset from a different BLADE dataset.
    2. Anonymize variable names - replaces variable names with generic
       names like "feature1", "feature2", etc.
    3. Shuffle feature order - randomly shuffles the order of features
       in the metadata.
    4. Shuffle feature names - randomly shuffles the feature names among the
       features.
    If multiple types of feature perturbations are desired, their order will
    follow the order listed above.
    """
    
    def __init__(self, anonymize: bool = False,
                 shuffle_order: bool = False, shuffle_names: bool = False,
                 add_random_features: int = 0, shuffle_order_seed: int = 42,
                 shuffle_names_seed: int = 42, random_features_seed: int = 42):
        self.anonymize = anonymize
        self.shuffle_order = shuffle_order
        self.shuffle_names = shuffle_names
        self.add_num_features = add_random_features
        self.shuffle_order_seed = shuffle_order_seed
        self.shuffle_names_seed = shuffle_names_seed
        self.random_features_seed = random_features_seed
        
    def anonymize_variable_names(self,
                                 json_metadata: dict,
                                 df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """
        Takes a JSON (in dictionary format) and replaces variable names 
        with non-descriptive names like "feature1", "feature2", etc.
        
        Args:
            json_metadata: Dictionary containing the JSON metadata
            dataset_name: Name of the current dataset to avoid selecting its
                          features.
            df: The current dataset as a pandas DataFrame
            
        Returns:
            json_metadata: The perturbed JSON metadata as a dictionary
            df: The perturbed dataframe with anonymized feature names as a pandas DataFrame
        """
        
        # create a copy of the metadata to work with
        json_metadata = deepcopy(json_metadata)
        
        # copy the dataframe
        df = deepcopy(df)
        
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
            
        # replace column names in the dataframe
        df.rename(columns=name_mapping, inplace=True)
            
        return json_metadata, df

    def shuffle_feature_order(self,
                              json_metadata: dict,
                              df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """
        Shuffles the order of features in the JSON metadata.
        
        Args:
            json_metadata: Dictionary containing the JSON metadata
            df: The current dataset as a pandas DataFrame
        
        Returns:
            json_metadata: The perturbed JSON metadata as a dictionary
            df: The dataframe (unchanged) as a pandas DataFrame
        """
        
        # create a copy of the metadata to work with
        json_metadata = deepcopy(json_metadata)
        
        # copy the dataframe
        df = deepcopy(df)
        
        # use numpy RNG for a single consistent permutation
        rng = np.random.default_rng(self.shuffle_order_seed)
        idxs = rng.permutation(df.shape[1])
                
        has_fields = "data_desc" in json_metadata and "fields" in json_metadata["data_desc"]
        has_field_names = "data_desc" in json_metadata and "field_names" in json_metadata["data_desc"]
        
        # shuffle the fields array if it exists
        if has_fields:
            fields = json_metadata["data_desc"]["fields"]
            fields = [fields[i] for i in idxs]
            json_metadata["data_desc"]["fields"] = fields
        
        # shuffle the field_names array if it exists
        if has_field_names:
            field_names = json_metadata["data_desc"]["field_names"]
            field_names = [field_names[i] for i in idxs]
            json_metadata["data_desc"]["field_names"] = field_names
            
        # shuffle the columns in the dataframe to match the new order of field_names if it exists
        df = df.iloc[:, idxs]
            
        return json_metadata, df

    def shuffle_feature_names(self,
                              json_metadata: dict,
                              df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """
        Randomly shuffles the feature names in the JSON metadata.
        
        Args:
            json_metadata: Dictionary containing the JSON metadata
            df: The current dataset as a pandas DataFrame
        
        Returns:
            json_metadata: The perturbed JSON metadata as a dictionary
            df: The dataframe (unchanged) as a pandas DataFrame
        """
        
        # create a copy of the metadata to work with
        json_metadata = deepcopy(json_metadata)
        
        # copy the dataframe
        df = deepcopy(df)
        
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
            
        # replace column names in the dataframe to match the new shuffled names
        df.rename(columns=name_mapping, inplace=True)
        
        return json_metadata, df

    def add_random_features(self,
                            json_metadata: dict,
                            dataset_name: str,
                            df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """
        Adds randomly selected feature(s) from a different randomly selected
        BLADE dataset to the JSON metadata.
        
        Args:
            json_metadata: Dictionary containing the JSON metadata
            dataset_name: Name of the current dataset to avoid selecting its
                          features.
            df: The current dataset as a pandas DataFrame
        Returns:
            The perturbed JSON metadata as a dictionary
            The perturbed dataframe with added features as a pandas DataFrame
        """
        
        # create a copy of the metadata to work with
        json_metadata = deepcopy(json_metadata)
        
        # copy dataframe
        df = deepcopy(df)
        
        # set random seed for reproducibility
        random.seed(self.random_features_seed)
        
        # get list of available datasets.
        available_datasets = list_datasets()
        
        # remove the current dataset if it's in the list
        if dataset_name in available_datasets:
            available_datasets.remove(dataset_name)
            
        # read in the features for each dataset
        full_feature_set = [] # list of tuples (dataset_name, feature_dict)
        for dataset in available_datasets:
            dataset_info_path = get_dataset_info_path(dataset)
            dataset_metadata = read_json(dataset_info_path)  # just to ensure the dataset is readable
            
            # extract features from this dataset
            if "data_desc" in dataset_metadata and "fields" in dataset_metadata["data_desc"]:
                full_feature_set.extend([(dataset, feature) for feature in dataset_metadata["data_desc"]["fields"]])

        # randomly select features to add
        features_to_add = random.sample(full_feature_set, 
                                        min(self.add_num_features,
                                            len(full_feature_set)))
        # add selected features to the current metadata
        if "data_desc" not in json_metadata:
            json_metadata["data_desc"] = {}
        if "fields" not in json_metadata["data_desc"]:
            json_metadata["data_desc"]["fields"] = []
        json_metadata["data_desc"]["fields"].extend([feature for _, feature in features_to_add])
        # add names to field_names array
        if "field_names" not in json_metadata["data_desc"]:
            json_metadata["data_desc"]["field_names"] = []
        json_metadata["data_desc"]["field_names"].extend(
            [feature[1]["column"] for feature in features_to_add if "column" in feature[1]]
        )
        
        # add selected features to the dataframe
        for dataset, feature in features_to_add:
            feature_name = feature["column"]
            feature_df = pd.read_csv(get_dataset_csv_path(dataset),
                                     usecols=[feature_name])
            # make feature_df the correct length by shortening or repeating rows
            if len(feature_df) > len(df):
                feature_df = feature_df.sample(n=len(df),
                                               random_state=self.random_features_seed).reset_index(drop=True)
            elif len(feature_df) < len(df):
                repeats = (len(df) // len(feature_df)) + 1
                feature_df = pd.concat([feature_df] * repeats,
                                       ignore_index=True).iloc[:len(df)]
            # add the feature column to the main dataframe
            df[feature_name] = feature_df[feature_name]
                
        return json_metadata, df
    
    def perturb(self, json_path: str) -> tuple[dict, pd.DataFrame]:
        """
        Applies the selected perturbations to the JSON metadata in the
        following order:
        1. Add random feature(s) to the dataset from a different BLADE dataset.
        2. Anonymize variable names
        3. Shuffle feature order
        4. Shuffle feature names
        
        Returns:
        - The perturbed JSON metadata as a dictionary
        - The perturbed dataframe with added features as a pandas DataFrame
          (if add_random_features > 0), otherwise None
        """
        
        # read in json metadata
        json_metadata = read_json(json_path)
        # get name of dataset from path
        dataset_name = os.path.basename(os.path.dirname(json_path))
        
        # read in dataset
        df_path = get_dataset_csv_path(dataset_name)
        df = pd.read_csv(df_path)
        
        if self.add_num_features > 0:
            json_metadata, df = self.add_random_features(json_metadata,
                                                         dataset_name,
                                                         df)
        
        if self.anonymize:
            json_metadata, df = self.anonymize_variable_names(json_metadata,
                                                              df)
        
        if self.shuffle_order:
            json_metadata, df = self.shuffle_feature_order(json_metadata, df)
        
        if self.shuffle_names:
            json_metadata, df = self.shuffle_feature_names(json_metadata, df)
        
        return json_metadata, df