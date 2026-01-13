from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/replace_with_rvs_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Drop rows missing key variables required for modeling
    df = df.dropna(subset=['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child'])

    # Remove or flag rows with non-positive hours (cannot be used as exposure)
    df = df[df['hours'] > 0]

    # Derive total number of people in the group (adults + children)
    df['persons_total'] = df['persons'] + df['child']

    # Mean-center persons_total to improve interpretability of intercepts
    df['persons_total_centered'] = df['persons_total'] - df['persons_total'].mean()

    # Descriptive rate: fish per hour (useful for summaries and plotting)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Log of hours to use as an offset (exposure) in GLM count models
    df['log_hours'] = np.log(df['hours'])

    # Ensure binary predictors are integers (0/1)
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Return transformed dataframe with all columns required by the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Ensure necessary transformed columns exist and drop any remaining missing rows
    df = df.dropna(subset=['fish_caught', 'log_hours', 'livebait', 'camper', 'persons_total_centered'])

    # Response and predictors
    y = df['fish_caught']
    X = df[['livebait', 'camper', 'persons_total_centered']]
    X = sm.add_constant(X)

    # Offset (log of exposure hours)
    offset = df['log_hours']

    # Fit Poisson GLM with log link and offset
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()

    # Assess overdispersion using the Pearson chi-square / df
    mu = poisson_model.predict(X, offset=offset)
    # Avoid division by zero; mu should be > 0 for well-defined predictions
    eps = 1e-8
    pearson_chi2 = np.sum(((y - mu) ** 2) / (mu + eps))
    dispersion = pearson_chi2 / (df.shape[0] - X.shape[1])

    print(f'Poisson dispersion estimate (Pearson chi2 / df): {dispersion:.3f}')

    # If there is substantial overdispersion, fit a Negative Binomial GLM
    # Threshold of 1.5 is a rule-of-thumb; adjust if desired
    if dispersion > 1.5:
        try:
            nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()
            print('Overdispersion detected; fitted Negative Binomial GLM (log link, hours as offset).')
            print(nb_model.summary())
            return nb_model
        except Exception as exc:
            print('Negative Binomial model failed to converge or threw an error; returning Poisson fit. Error:', exc)
            print(poisson_model.summary())
            return poisson_model
    else:
        print('No substantial overdispersion detected; returning Poisson GLM result (log link, hours as offset).')
        print(poisson_model.summary())
        return poisson_model


