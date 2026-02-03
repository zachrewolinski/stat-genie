import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("amtl.csv")

# Column mapping inferred from values:
# sockets -> tooth class (Anterior/Posterior/Premolar)
# tooth_class -> genus (Homo sapiens, Pan, Pongo, Papio)
# genus -> count of missing teeth (AMTL) for that class
# age -> count of observable sockets (trials)
# pop -> estimated age at death
# stdev_age -> probability male (sex estimate)

# Prepare modeling data
model_df = df[df["age"] > 0].copy()
model_df["amtl_rate"] = model_df["genus"] / model_df["age"]

# Use Homo sapiens as reference for genus
model_df["tooth_class"] = pd.Categorical(
    model_df["tooth_class"],
    categories=["Homo sapiens", "Pan", "Pongo", "Papio"],
    ordered=False,
)
model_df["sockets"] = pd.Categorical(
    model_df["sockets"],
    categories=["Anterior", "Premolar", "Posterior"],
    ordered=False,
)

formula = "amtl_rate ~ C(tooth_class) + pop + stdev_age + C(sockets)"
model = smf.glm(
    formula=formula,
    data=model_df,
    family=sm.families.Binomial(),
    var_weights=model_df["age"],
).fit()

# Extract coefficients for genera vs Homo sapiens
params = model.params
conf = model.conf_int()

results = []
for genus in ["Pan", "Pongo", "Papio"]:
    term = f"C(tooth_class)[T.{genus}]"
    if term in params.index:
        coef = params[term]
        ci_low, ci_high = conf.loc[term]
        results.append((genus, coef, ci_low, ci_high, model.pvalues[term]))

# Predicted mean AMTL rate by genus at average covariates
avg_pop = model_df["pop"].mean()
avg_male = model_df["stdev_age"].mean()

pred_rows = []
for genus in ["Homo sapiens", "Pan", "Pongo", "Papio"]:
    for socket in ["Anterior", "Premolar", "Posterior"]:
        pred_rows.append(
            {
                "tooth_class": genus,
                "pop": avg_pop,
                "stdev_age": avg_male,
                "sockets": socket,
                "age": model_df["age"].median(),
                "amtl_rate": 0.0,
            }
        )

pred_df = pd.DataFrame(pred_rows)
pred_df["pred_rate"] = model.predict(pred_df)
pred_summary = pred_df.groupby("tooth_class")["pred_rate"].mean()

print("GLM (binomial) coefficients vs Homo sapiens:")
for genus, coef, ci_low, ci_high, pval in results:
    print(f"{genus}: coef={coef:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f}), p={pval:.3g}")
print("\nPredicted AMTL rate by genus (avg covariates, mean over tooth classes):")
print(pred_summary)
