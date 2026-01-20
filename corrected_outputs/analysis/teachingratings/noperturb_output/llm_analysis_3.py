from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/noperturb_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Drop observations missing the core variables needed for modeling
    df = df.dropna(subset=['beauty', 'eval', 'gender', 'age', 'students', 'prof'])

    # Standardize the main independent variable (beauty)
    # Use population-style std (ddof=0) for z-scoring; consistent scaling is important
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std(ddof=0)

    # Center age to aid interpretation (coefficients reflect effect per year from sample mean)
    df['age_c'] = df['age'] - df['age'].mean()

    # Transform class size to reduce skew and heteroskedasticity
    df['log_students'] = np.log1p(df['students'].astype(float))

    # Map categorical factors to numeric 0/1 indicators for modeling
    df['gender_female'] = df['gender'].map({'female': 1, 'male': 0})
    df['minority_yes'] = df['minority'].map({'yes': 1, 'no': 0})
    df['tenure_yes'] = df['tenure'].map({'yes': 1, 'no': 0})
    df['native_yes'] = df['native'].map({'yes': 1, 'no': 0})
    df['division_upper'] = df['division'].map({'upper': 1, 'lower': 0})
    df['credits_more'] = df['credits'].map({'more': 1, 'single': 0})

    # Replace any unmapped values with 0 (defensive); if mapping failed this will keep data usable
    for col in ['gender_female', 'minority_yes', 'tenure_yes', 'native_yes', 'division_upper', 'credits_more']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Interaction term to test moderation by gender
    df['beauty_x_female'] = df['beauty_z'] * df['gender_female']

    # Ensure professor id is integer for clustering
    df['prof'] = df['prof'].astype(int)

    # Final check: drop any rows with NA in the created columns (should be uncommon)
    df = df.dropna(subset=['beauty_z', 'age_c', 'log_students', 'gender_female', 'prof', 'eval'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf

    # Base formula with controls
    formula_base = (
        'eval ~ beauty_z + gender_female + age_c + log_students '
        '+ minority_yes + tenure_yes + native_yes + division_upper + credits_more'
    )

    # Fit base OLS
    m1 = smf.ols(formula_base, data=df).fit()
    # Clustered standard errors by professor id (prof)
    m1_clust = m1.get_robustcov_results(cov_type='cluster', groups=df['prof'])

    # Model with interaction to test whether beauty effect differs by gender
    formula_int = formula_base + ' + beauty_x_female'
    m2 = smf.ols(formula_int, data=df).fit()
    m2_clust = m2.get_robustcov_results(cov_type='cluster', groups=df['prof'])

    # Optional robustness: model with professor fixed effects (uncomment if desired)
    # This can soak up instructor-level unobserved heterogeneity but is costly in degrees of freedom
    # formula_fe = formula_base + ' + C(prof)'
    # m_fe = smf.ols(formula_fe, data=df).fit()
    # m_fe_clust = m_fe.get_robustcov_results(cov_type='cluster', groups=df['prof'])

    # Return fitted results (cluster-robust objects) so caller can examine .summary() or coefficients
    results = {
        'base_model_clustered': m1_clust,
        'interaction_model_clustered': m2_clust,
        # 'fixed_effects_clustered': m_fe_clust
    }

    return results


