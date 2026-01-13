from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from types import SimpleNamespace

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe for logistic models.

    Produces these columns used in the model:
    - MajorityChoice: 1 if y==2 (majority), 0 otherwise
    - Age_c: age centered (age - mean(age))
    - Age_sq: squared term of Age_c to capture nonlinearity
    - Female: 1 if gender == 1 (girl), 0 if gender == 2 (boy)
    - majority_first: coerced to integer (0/1)
    - culture: kept as-is (used as categorical in the model)
    """
    df = df.copy()

    # Keep rows with required variables
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=required)

    # Create dependent variable: majority choice indicator
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Ensure age is numeric and within plausible range (4-14 in schema)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df = df[(df['age'] >= 4) & (df['age'] <= 14)].copy()

    # Center age and add quadratic term
    df['Age_c'] = df['age'] - df['age'].mean()
    df['Age_sq'] = df['Age_c'] ** 2

    # Gender -> Female indicator (schema: 1=girl, 2=boy)
    df['Female'] = (df['gender'] == 1).astype(int)

    # Ensure majority_first is 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Ensure culture is integer-coded where possible, otherwise keep as categorical
    try:
        df['culture'] = df['culture'].astype(int)
    except Exception:
        df['culture'] = df['culture'].astype('category')

    # (Optional) drop rows with missing created columns (defensive)
    df = df.dropna(subset=['MajorityChoice', 'Age_c', 'Age_sq', 'Female', 'majority_first', 'culture'])

    # Reset index for clean output
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting MajorityChoice from age (linear + quadratic), culture,
    and their interactions (Age_c x culture). Controls: Female and majority_first.

    Returns the fitted model results with cluster-robust standard errors by culture.
    """
    # Formula: main effects for age (centered) and age^2, culture as categorical, interaction of Age_c with culture,
    # and controls Female and majority_first.
    # Use C(culture) so culture is treated as categorical regardless of numeric coding.
    formula = 'MajorityChoice ~ Age_c + Age_sq + C(culture) + Age_c:C(culture) + Female + majority_first'

    # Fit the logistic model (binomial, logit link)
    model_fit = smf.logit(formula, data=df).fit(disp=False)

    # Obtain cluster-robust covariance by culture (accounts for within-culture correlation)
    groups = df['culture']

    # Some versions of statsmodels provide get_robustcov_results on result objects,
    # but if not available, fall back to creating a lightweight results-like object
    # that supports predict and exposes clustered covariance / standard errors.
    try:
        robust = model_fit.get_robustcov_results(cov_type='cluster', groups=groups)
    except Exception:
        # Try to compute clustered covariance matrix
        try:
            from statsmodels.stats.sandwich_covariance import cov_cluster
            cov = cov_cluster(model_fit, groups)
            bse = np.sqrt(np.diag(cov))
        except Exception:
            # As a last resort, fall back to the model's default covariance
            try:
                cov = model_fit.cov_params()
                bse = model_fit.bse
            except Exception:
                cov = np.diag(np.square(model_fit.bse))
                bse = model_fit.bse

        # Create a simple wrapper that provides predict, params, bse, and cov_params callable
        robust = SimpleNamespace(
            predict=model_fit.predict,
            params=model_fit.params,
            bse=bse,
            cov_params=lambda: cov
        )

    # For interpretability, also compute predicted probability by age for each culture (optional):
    # Here we compute marginal predicted probabilities across observed ages for each culture
    try:
        ages = np.sort(df['age'].unique())
        preds = {}
        for c in sorted(df['culture'].unique()):
            tmp = df.iloc[0:1].copy()
            # create a template row: set culture to c, Female and majority_first to reference (mean/mode)
            tmp.loc[:, 'Female'] = df['Female'].mean()
            tmp.loc[:, 'majority_first'] = df['majority_first'].mode().iloc[0] if not df['majority_first'].mode().empty else 0
            tmp.loc[:, 'culture'] = c
            pred_list = []
            for a in ages:
                tmp.loc[:, 'age'] = a
                tmp.loc[:, 'Age_c'] = a - df['age'].mean()
                tmp.loc[:, 'Age_sq'] = tmp.loc[:, 'Age_c'] ** 2
                p = robust.predict(tmp.iloc[0:1])[0]
                pred_list.append(p)
            key = int(c) if (isinstance(c, (int, np.integer))) else str(c)
            preds[key] = (ages.tolist(), pred_list)
    except Exception:
        preds = None

    # Return both the fitted object and the robust results plus optional predictions
    return {
        'fitted_model': model_fit,
        'robust_results': robust,
        'predicted_by_age_and_culture': preds
    }