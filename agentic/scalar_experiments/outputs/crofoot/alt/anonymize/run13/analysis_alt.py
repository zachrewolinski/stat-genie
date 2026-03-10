import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

path = "/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/crofoot/alt/anonymize/run13/crofoot.csv"
df = pd.read_csv(path)

y = df["feature4"]

# Define alternative predictors
pred_sets = {}

pred_sets["log_ratio"] = {
    "rel_size": np.log(df["feature7"] / df["feature8"]),
    "loc_adv": df["feature6"] - df["feature5"],
}

pred_sets["diff"] = {
    "rel_size": df["feature7"] - df["feature8"],
    "loc_adv": df["feature6"] - df["feature5"],
}

pred_sets["ratio_dist"] = {
    "rel_size": np.log(df["feature7"] / df["feature8"]),
    "loc_adv": np.log(df["feature6"] / df["feature5"]),
}

pred_sets["scaled_dist"] = {
    "rel_size": np.log(df["feature7"] / df["feature8"]),
    "loc_adv": (df["feature6"] - df["feature5"]) / (df["feature6"] + df["feature5"]),
}

# Null model
model_null = sm.GLM(y, sm.add_constant(pd.DataFrame({"intercept": np.ones(len(y))})), family=sm.families.Binomial()).fit()


def lr_test(full, reduced):
    lr = 2 * (full.llf - reduced.llf)
    df_diff = full.df_model - reduced.df_model
    p = stats.chi2.sf(lr, df_diff)
    return lr, df_diff, p

results = {}
for name, preds in pred_sets.items():
    X = sm.add_constant(pd.DataFrame(preds))
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    lr = lr_test(res, model_null)
    results[name] = {
        "coef": res.params.to_dict(),
        "pvalues": res.pvalues.to_dict(),
        "lr_full_vs_null": lr,
        "aic": res.aic,
    }

print(results)
