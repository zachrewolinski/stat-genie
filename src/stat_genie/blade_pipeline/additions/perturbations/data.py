# imports
import os
import random
import pandas as pd
import numpy as np

from stat_genie.blade_pipeline.utils import (
    get_dataset_csv_path,
)

class DataPerturbation:
    """
    Centralized object for perturbing the entries of BLADE datasets. There
    is currently one supported action:
    1. Replace the dataset's features with independent random variables.
        - Columns with data type 'float' will be replaced with Normal r.v.s
          centered at (max-min)/2 with std dev of the original column.
        - Columns with data type 'int' will be replaced with Uniform r.v.s
          over the same range as the original column.
        - Columns with data type 'object' will be replaced with random samples
          from the original column.
    If multiple types of feature perturbations are desired, their order will
    follow the order listed above.
    """
    
    def __init__(self, replace_features: bool = False,
                 replace_features_seed: int = 42):
        
        self.replace_features = replace_features
        self.replace_features_seed = replace_features_seed

    def replace_with_rvs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces the features of the dataset with independent random variables.
        Importantly, the changes are not reflected in the JSON metadata, to make
        this perturbation bigger in scale and more likely to flip downstream
        model predictions.
        
        Args:
            df: The dataframe whose features are to be replaced.
            
        Returns:
        - The perturbed dataframe
        """
        
        # set random seed for reproducibility
        random.seed(self.replace_features_seed)
        
        # go through each column and replace with random variables
        for col in df.columns:
            old_values = df[col]
            if pd.api.types.is_float_dtype(old_values.dtype):
                # get mean and std for the new values
                loc = (old_values.max() - old_values.min()) / 2.0
                scale = old_values.std(ddof=0)
                # handle edge case
                if pd.isna(scale) or scale == 0:
                    scale = 1.0
                # replace with normal r.v.s
                df[col] = np.random.normal(loc=loc, scale=scale, size=len(df))
            elif pd.api.types.is_integer_dtype(old_values.dtype):
                # get range for the new values
                cmin = int(old_values.min())
                cmax = int(old_values.max())
                # handle edge case
                if cmin == cmax:
                    df[col] = [cmin] * len(df)
                # replace with uniform r.v.s
                else:
                    df[col] = np.random.randint(
                        low=cmin,
                        high=cmax + 1,
                        size=len(df)
                    )
            else:
                # object / categorical: sample original values with replacement
                df[col] = old_values.sample(n=len(df), replace=True,
                    random_state=self.replace_features_seed).reset_index(
                        drop=True
                    )

        return df
    
    def perturb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the selected perturbations to the data in the following order:
        1. Replace the dataset's features with independent random variables.
        
        Returns:
            The perturbed dataset as a dataframe
        """
        # in case no perturbations are selected
        perturbed_df = df
        
        if self.replace_features:
            perturbed_df = self.replace_with_rvs(df)
        
        return perturbed_df