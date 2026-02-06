# imports
import pandas as pd
from copy import deepcopy

class DataPerturbation:
    """
    Centralized object for perturbing the entries of BLADE datasets. There
    is currently one supported action:
    1. Shuffle the values of each column independently of each other.
        - Any patterns within the data should be broken.
    If multiple types of feature perturbations are desired, their order will
    follow the order listed above.
    """
    
    def __init__(self, shuffle_values: bool = False,
                 shuffle_values_seed: int = 42):
        
        self.shuffle_values = shuffle_values
        self.shuffle_values_seed = shuffle_values_seed
        
    def shuffle_values_in_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffles the values in each column of the dataframe independently.
        This should flip downstream model predictions to make the answer "No",
        if it is not already.
        
        Args:
            df: The dataframe whose values are to be shuffled.
            
        Returns:
        - The perturbed dataframe
        """
        
        # set random seed for reproducibility
        seed = self.shuffle_values_seed
        
        # copy the dataframe to avoid modifying the original one
        df = deepcopy(df)
        
        # go through each column and randomly shuffle its values,
        # independently of how the other columns were shuffled.
        for col in df.columns:
            df[col] = df[col].sample(n=df.shape[0],
                                     random_state=seed).reset_index(drop=True)
            seed += 1  # change seed for the next column for different shuffling

        return df
    
    def perturb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the selected perturbations to the data in the following order:
        1. Shuffle the values of each column independently of each other.
        
        Returns:
            The perturbed dataset as a dataframe
        """
        # in case no perturbations are selected
        perturbed_df = deepcopy(df)
        
        if self.shuffle_values:
            perturbed_df = self.shuffle_values_in_cols(perturbed_df)
        
        return perturbed_df