import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv("affairs.csv")
children = df["feature6"].astype(str).str.lower()
affairs = df["feature2"].astype(float)

X = pd.DataFrame({
    "children_yes": (children == "yes").astype(int),
    "gender_male": (df["feature3"].astype(str).str.lower() == "male").astype(int),
    "age": df["feature4"].astype(float),
    "years_married": df["feature5"].astype(float),
    "religiousness": df["feature7"].astype(float),
    "education": df["feature8"].astype(float),
    "occupation": df["feature9"].astype(float),
    "marriage_rating": df["feature10"].astype(float),
})
X = sm.add_constant(X, has_constant="add")

ols_model = sm.OLS(affairs, X)
ols_res = ols_model.fit(cov_type="HC3")

out = {
    "ols_coef_children_yes": float(ols_res.params["children_yes"]),
    "ols_se_children_yes": float(ols_res.bse["children_yes"]),
    "ols_p_children_yes": float(ols_res.pvalues["children_yes"]),
}
print(json.dumps(out, indent=2))
