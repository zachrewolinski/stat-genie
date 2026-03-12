import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/crofoot/alt/anonymize/run13/crofoot.csv"
df = pd.read_csv(path)

# Define variables
# Relative group size: log ratio to make symmetric
# Add small epsilon just in case
rel_size = np.log(df["feature7"] / df["feature8"])

# Contest location advantage: positive when contest is closer to focal group center
loc_adv = df["feature6"] - df["feature5"]

# Outcome
y = df["feature4"]

# Prepare design matrix with intercept
X = pd.DataFrame({
    "rel_size": rel_size,
    "loc_adv": loc_adv,
})
X = sm.add_constant(X)

# Fit logistic regression
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Also fit reduced models for comparison
model_size = sm.GLM(y, sm.add_constant(X[["rel_size"]]), family=sm.families.Binomial()).fit()
model_loc = sm.GLM(y, sm.add_constant(X[["loc_adv"]]), family=sm.families.Binomial()).fit()

# Likelihood ratio tests
# Comparing full vs intercept-only
model_null = sm.GLM(y, sm.add_constant(pd.DataFrame({"intercept": np.ones(len(y))})), family=sm.families.Binomial()).fit()

from scipy import stats

def lr_test(full, reduced):
    lr = 2 * (full.llf - reduced.llf)
    df_diff = full.df_model - reduced.df_model
    p = stats.chi2.sf(lr, df_diff)
    return lr, df_diff, p

lr_full_vs_null = lr_test(result, model_null)
lr_size_vs_null = lr_test(model_size, model_null)
lr_loc_vs_null = lr_test(model_loc, model_null)

# Collect summary stats
summary = {
    "n": int(len(df)),
    "coef": result.params.to_dict(),
    "se": result.bse.to_dict(),
    "pvalues": result.pvalues.to_dict(),
    "llf": result.llf,
    "aic": result.aic,
    "lr_full_vs_null": lr_full_vs_null,
    "lr_size_vs_null": lr_size_vs_null,
    "lr_loc_vs_null": lr_loc_vs_null,
}

print(summary)
