from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/replace_with_rvs_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Steps:
    - copy dataframe and drop rows missing key fields (rater1/rater2, redCards, games)
    - create SkinToneAvg (mean of rater1 and rater2)
    - create SkinGroup (Light/Medium/Dark) using thresholds on the 0-1 normalized ratings
    - restrict the sample to the extremes (Light vs Dark) to match the research question comparing dark vs light
    - create SkinDark binary (1 = Dark, 0 = Light)
    - parse birthday and compute age at mid-season reference date (2013-01-01)
    - compute offset = log(games) for use as exposure in count model
    - coerce numeric controls to numeric and drop rows with non-finite values
    """

    df = df.copy()

    # Required columns existence check (will raise KeyError if missing)
    required = ['rater1', 'rater2', 'redCards', 'games', 'birthday', 'height', 'weight', 'meanIAT', 'meanExp', 'position', 'leagueCountry', 'refNum']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows where essential values are missing
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games'])

    # Skin tone: average the two rater scores (they are scaled 0..1)
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define groups: Light (<=0.25), Dark (>=0.75), Medium (between). These thresholds map to the extreme categories of the 5-point scale.
    df['SkinGroup'] = pd.cut(df['SkinToneAvg'], bins=[-1e9, 0.25, 0.75, 1e9], labels=['Light', 'Medium', 'Dark'])

    # Keep only Light and Dark to directly compare extremes (research Q: dark vs light)
    df = df[df['SkinGroup'].isin(['Light', 'Dark'])].copy()

    # Binary indicator: 1 = Dark, 0 = Light
    df['SkinDark'] = (df['SkinGroup'] == 'Dark').astype(int)

    # Parse birthday (format dd.mm.yyyy). Coerce errors; drop rows where birthday missing after parse
    df['birthday_parsed'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    df = df.dropna(subset=['birthday_parsed'])

    # Compute age at mid-season date (use 2013-01-01 as a reference midpoint of the 2012-2013 season)
    season_ref = pd.to_datetime('2013-01-01')
    df['age'] = (season_ref - df['birthday_parsed']).dt.days / 365.25

    # Exposure offset: log(number of games). Ensure games > 0
    df = df[df['games'] > 0].copy()
    df['offset'] = np.log(df['games'].astype(float))

    # Coerce numeric controls to numeric, drop rows with missing or non-finite values in controls
    for col in ['height', 'weight', 'meanIAT', 'meanExp', 'redCards']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing/infinite values in model columns
    model_cols = ['redCards', 'SkinDark', 'SkinToneAvg', 'SkinGroup', 'games', 'offset', 'position', 'age', 'height', 'weight', 'meanIAT', 'meanExp', 'leagueCountry', 'refNum']
    df = df.dropna(subset=model_cols)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial GLM for redCards with exposure offset = log(games).

    Model specification:
    - Dependent variable: redCards (count)
    - Independent variable of interest: SkinDark (1 = dark, 0 = light)
    - Controls: age, height, weight, meanIAT, meanExp, categorical position, categorical leagueCountry
    - Exposure: games included as an offset (log scale)
    - Clustered (robust) standard errors at the referee level (refNum)

    Returns an object exposing the fitted results and clustered standard errors.
    """

    # Copy to avoid modifying input
    df = df.copy()

    # Formula: categorical variables handled with C(...)
    formula = (
        'redCards ~ SkinDark + age + height + weight + meanIAT + meanExp + '
        'C(position) + C(leagueCountry)'
    )

    # Build and fit Negative Binomial GLM with offset
    model_glm = sm.GLM.from_formula(formula,
                                   data=df,
                                   family=sm.families.NegativeBinomial(),
                                   offset=df['offset'])

    results = model_glm.fit()

    # Obtain cluster-robust covariance matrix clustered by referee (refNum)
    try:
        # Use statsmodels' sandwich covariance utility to compute clustered covariance
        clustered_cov = sm.stats.sandwich_covariance.cov_cluster(results, df['refNum'])
    except Exception:
        # Fallback: if cov_cluster is unavailable or fails, fall back to the original covariance
        clustered_cov = results.cov_params()

    # Build a lightweight wrapper that exposes clustered standard errors and a summary
    class ClusteredResults:
        def __init__(self, base_results, clustered_cov, groups):
            self._results = base_results
            self.cov_cluster = clustered_cov
            self.cluster_groups = groups
            # prepare commonly used attributes derived from clustered covariance
            self.params = self._results.params
            self.bse = np.sqrt(np.diag(self.cov_cluster))
            # two-sided z tests
            self.zvalues = self.params / self.bse
            self.pvalues = 2 * (1 - sps.norm.cdf(np.abs(self.zvalues)))
            z_crit = sps.norm.ppf(0.975)
            self.conf_int = np.vstack((self.params - z_crit * self.bse, self.params + z_crit * self.bse)).T

        def summary(self):
            # Print the original summary first for reference
            print(self._results.summary())
            # Then print clustered SE table
            header = f"\nClustered standard errors (clustered by refNum). Number of clusters: {len(pd.unique(self.cluster_groups))}\n"
            print(header)
            # Build a small table
            rows = []
            fmt = "{:30s} {:>12s} {:>12s} {:>12s} {:>12s}"
            print(fmt.format("Parameter", "Coef.", "Std.Err", "z", "P>|z|"))
            for name, coef, se, z, p in zip(self.params.index, self.params.values, self.bse, self.zvalues, self.pvalues):
                print(fmt.format(str(name), f"{coef:12.4f}", f"{se:12.4f}", f"{z:12.4f}", f"{p:12.4g}"))
            # Print confidence intervals
            print("\nParameter confidence intervals (clustered):")
            for name, ci in zip(self.params.index, self.conf_int):
                print(f"{name:30s} [{ci[0]:.4f}, {ci[1]:.4f}]")

        def __getattr__(self, item):
            # Delegate attribute access to the original results where appropriate
            return getattr(self._results, item)

        def __repr__(self):
            return f"<ClusteredResults wrapper; clustered by refNum; base_results={repr(self._results)}>"

    clustered_results = ClusteredResults(results, clustered_cov, df['refNum'].values)

    # Print a brief summary with clustered SEs
    clustered_results.summary()

    return clustered_results