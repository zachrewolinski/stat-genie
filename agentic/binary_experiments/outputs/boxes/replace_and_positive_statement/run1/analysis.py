import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(model_restricted, model_full):
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def summarize_logit(label, m0, m1, m2):
    lr_culture = lr_test(m0, m1)
    lr_inter = lr_test(m1, m2)
    print(f"\n=== {label} ===")
    print("Base model: age only")
    age_p = m0.pvalues.get("age_c", np.nan)
    print(f"  llf={m0.llf:.3f}, df_model={int(m0.df_model)}, age_p={age_p:.4g}")
    print("Add culture:")
    print(
        f"  LR={lr_culture[0]:.3f}, df={int(lr_culture[1])}, p={lr_culture[2]:.4g}"
    )
    print("Add age*culture interaction:")
    print(f"  LR={lr_inter[0]:.3f}, df={int(lr_inter[1])}, p={lr_inter[2]:.4g}")


# Load data
_df = pd.read_csv("boxes.csv")
_df["age_c"] = _df["age"] - _df["age"].mean()

# 1) Reliance on social information: chose demonstrated option (majority or minority)
_df["use_social"] = (_df["y"] != 1).astype(int)

m0_social = smf.logit("use_social ~ age_c", data=_df).fit(disp=0)
m1_social = smf.logit("use_social ~ age_c + C(culture)", data=_df).fit(disp=0)
m2_social = smf.logit("use_social ~ age_c * C(culture)", data=_df).fit(disp=0)

summarize_logit("Reliance on social information", m0_social, m1_social, m2_social)

# 2) Majority preference among those who chose a demonstrated option
_df_demo = _df[_df["y"].isin([2, 3])].copy()
_df_demo["majority"] = (_df_demo["y"] == 2).astype(int)

m0_maj = smf.logit("majority ~ age_c", data=_df_demo).fit(disp=0)
m1_maj = smf.logit("majority ~ age_c + C(culture)", data=_df_demo).fit(disp=0)
m2_maj = smf.logit("majority ~ age_c * C(culture)", data=_df_demo).fit(disp=0)

summarize_logit("Preference for majority cues (among demonstrated choices)", m0_maj, m1_maj, m2_maj)

# Provide a small descriptive table to help interpret differences
# Social reliance rate by culture and age group (4-7, 8-11, 12-14)
_df["age_group"] = pd.cut(_df["age"], bins=[3, 7, 11, 14], labels=["4-7", "8-11", "12-14"])

desc_social = (
    _df.groupby(["culture", "age_group"])["use_social"]
    .mean()
    .reset_index()
    .pivot(index="culture", columns="age_group", values="use_social")
)

print("\nSocial reliance rate by culture and age group (mean use_social):")
print(desc_social.round(3))

# Majority preference rate by culture and age group
_desc_demo = _df_demo.copy()
_desc_demo["age_group"] = pd.cut(_desc_demo["age"], bins=[3, 7, 11, 14], labels=["4-7", "8-11", "12-14"])

maj_rate = (
    _desc_demo.groupby(["culture", "age_group"])["majority"]
    .mean()
    .reset_index()
    .pivot(index="culture", columns="age_group", values="majority")
)

print("\nMajority preference rate by culture and age group (mean majority):")
print(maj_rate.round(3))
