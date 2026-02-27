# imports
import numpy as np
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

    def __init__(
        self,
        shuffle_values: bool = False,
        shuffle_values_seed: int = 42,
        set_pve: bool = False,
        pve: float = None,
        iv_idxs: list[int] = None,
        dv_idxs: list[int] = None,
        set_pve_seed: int = 42,
    ):
        # ensure that both shuffle and pve perturbations are not selected at
        # the same time, since they will conflict with each other
        if shuffle_values and set_pve:
            raise ValueError(
                "Cannot apply both shuffle and PVE perturbations simultaneously."
            )
        # check that necessary parameters are provided if set_pve is True
        if set_pve:
            if pve is None or iv_idxs is None or dv_idxs is None:
                raise ValueError(
                    "If set_pve is True, pve, iv_idxs, & dv_idxs are required."
                )

        self.shuffle_values = shuffle_values
        self.shuffle_values_seed = shuffle_values_seed
        self.control_pve = set_pve
        self.pve = pve
        self.iv_idxs = iv_idxs
        self.dv_idxs = dv_idxs
        self.set_pve_seed = set_pve_seed

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
            df[col] = (
                df[col]
                .sample(n=df.shape[0], random_state=seed)
                .reset_index(drop=True)
            )
            seed += 1  # change seed for the next column for different shuffling

        return df

    def set_pve(
        self,
        json_metadata: dict,
        df: pd.DataFrame,
        pve: float,
        iv_idxs: list[int],
        dv_idxs: list[int],
        seed: int = 42,
    ) -> tuple[dict, pd.DataFrame]:
        """
        Modifies each dependent variable column so that the independent
        variables explain exactly `pve` proportion of its variance.

        The new DV is constructed as:
            $Z = \hat{Y} + \epsilon$,  where $\hat{Y}$ = OLS fit of IVs on DV
        with $\sigma_\epsilon$ chosen so that $Var(\hat{Y}) / Var(Z)$ = pve.

        Args:
            json_metadata: Dictionary containing the JSON metadata.
            df: Input dataset.
            pve: Target proportion of variance explained (0 to 1 inclusive).
            iv_idxs: Column indices of the independent variables.
            dv_idxs: Column indices of the dependent variables to modify.
            seed: Random seed for reproducibility.

        Returns:
            The perturbed metadata
            A copy of `df` with the DV columns modified.
        """

        # check that the provided pve is valid
        if pve < 0 or pve > 1:
            raise ValueError("PVE must be between 0 and 1.")

        # check that the provided iv and dv indices are valid
        num_cols = df.shape[1]
        if any(idx < 0 or idx >= num_cols for idx in iv_idxs + dv_idxs):
            raise ValueError(
                "IV and DV column indices are out of range for the dataframe."
            )

        # create a copy of the metadata to work with
        json_metadata = deepcopy(json_metadata)

        # copy the dataframe to avoid modifying the original one
        df = deepcopy(df)

        # control randomness for reproducibility
        rng = np.random.default_rng(seed)

        # build design matrix with intercept column; string/categorical IVs
        # are one-hot encoded (drop_first to avoid multicollinearity)
        iv_df = df.iloc[:, iv_idxs]
        encoded_parts = []
        for col in iv_df.columns:
            if pd.api.types.is_string_dtype(
                iv_df[col]
            ) or pd.api.types.is_categorical_dtype(iv_df[col]):
                dummies = pd.get_dummies(iv_df[col], drop_first=True).astype(
                    float
                )
                encoded_parts.append(dummies.values)
            else:
                encoded_parts.append(
                    iv_df[col].values.astype(float).reshape(-1, 1)
                )
        X = np.column_stack([np.ones(len(df))] + encoded_parts)

        for dv_idx in dv_idxs:
            y = df.iloc[:, dv_idx].values
            y_numeric = y.astype(float)

            # OLS fitted values are the "signal" component from the IVs
            coeffs, _, _, _ = np.linalg.lstsq(X, y_numeric, rcond=None)
            y_hat = X @ coeffs
            var_signal = np.var(y_hat)

            if pve > 0 and np.isclose(var_signal, 0, atol=1e-10):
                raise ValueError(
                    f"DV at column index {dv_idx}: the independent variables "
                    "have no linear variation and cannot achieve PVE > 0."
                )

            # construct latent variable Z = signal + noise
            # derivation: $Var(\hat{Y}) / (Var(\hat{Y}) + \sigma_\epsilon^2)$ = pve
            #             => $\sigma_\epsilon^2 = sqrt(Var(\hat{Y})$ * (1 - pve) / pve)
            if pve == 0:
                z = rng.normal(0, 1, size=len(y))
            elif pve == 1:
                z = y_hat
            else:
                sigma_noise = np.sqrt(var_signal * (1 - pve) / pve)
                z = y_hat + rng.normal(0, sigma_noise, size=len(y))

            # replace the original DV column with the new Z values
            df.iloc[:, dv_idx] = z

            # update field metadata to reflect the perturbed DV's distribution
            field_props = json_metadata["data_desc"]["fields"][dv_idx][
                "properties"
            ]
            sample_idxs = rng.choice(len(z), size=3, replace=False)
            field_props["std"] = float(np.std(z))
            field_props["min"] = float(np.min(z))
            field_props["max"] = float(np.max(z))
            field_props["samples"] = [float(z[i]) for i in sample_idxs]
            field_props["num_unique_values"] = int(len(np.unique(z)))

        return json_metadata, df

    def perturb(
        self, json_metadata: dict, df: pd.DataFrame
    ) -> tuple[dict, pd.DataFrame]:
        """
        Applies the selected perturbations to the data in the following order:
        1a. Shuffle the values of each column independently of each other.
        1b. Set the PVE of the IVs on the DVs to the specified value.

        Perturbations 1a and 1b cannot be applied at the same time since they
        will conflict with each other.

        Args:
            json_metadata: Metadata information for the dataset.
            df: The dataset to perturb.

        Returns:
            updated_metadata, perturbed_df
        """

        # in case no perturbations are selected
        perturbed_df = deepcopy(df)
        updated_metadata = deepcopy(json_metadata)

        if self.shuffle_values:
            perturbed_df = self.shuffle_values_in_cols(perturbed_df)

        if self.control_pve:
            updated_metadata, perturbed_df = self.set_pve(
                updated_metadata,
                perturbed_df,
                pve=self.pve,
                iv_idxs=self.iv_idxs,
                dv_idxs=self.dv_idxs,
                seed=self.set_pve_seed,
            )

        return updated_metadata, perturbed_df
