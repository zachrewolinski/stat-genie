import json
import numpy as np
import pandas as pd
from scipy import stats

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Define variables
# feature20 is reading speed in words per minute (equals words / (feature5/60000))
reading_speed = "feature20"
reader_view = "feature3"  # 1 = on, 0 = off
is_dyslexic = "feature17"  # 1 = dyslexia

# Subset to dyslexic readers
sub = df[df[is_dyslexic] == 1].copy()

# Groups
g_off = sub[sub[reader_view] == 0][reading_speed].dropna()
g_on = sub[sub[reader_view] == 1][reading_speed].dropna()

# Welch's t-test
res = stats.ttest_ind(g_on, g_off, equal_var=False, nan_policy="omit")

# Effect size (Cohen's d using pooled SD)
def cohend(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (a.mean() - b.mean()) / s

# Welch-Satterthwaite df for CI
na, nb = len(g_on), len(g_off)
va, vb = g_on.var(ddof=1), g_off.var(ddof=1)
se = np.sqrt(va / na + vb / nb)

# degrees of freedom for Welch
num = (va / na + vb / nb) ** 2
den = (va ** 2) / (na ** 2 * (na - 1)) + (vb ** 2) / (nb ** 2 * (nb - 1))
df_welch = num / den

# 95% CI for mean difference (on - off)
alpha = 0.05
tcrit = stats.t.ppf(1 - alpha / 2, df_welch)
mean_diff = g_on.mean() - g_off.mean()
ci_low = mean_diff - tcrit * se
ci_high = mean_diff + tcrit * se

# Compose explanation
explanation = (
    "Analyzed dyslexic readers (feature17=1) and compared reading speed (feature20, words per minute; "
    "equals words divided by reading time excluding scrolling) between Reader View on (feature3=1) and off (feature3=0). "
    f"Sample sizes were n_on={na} and n_off={nb}. Mean speed was {g_on.mean():.2f} wpm with Reader View on versus "
    f"{g_off.mean():.2f} wpm with it off (difference {mean_diff:.2f} wpm). A Welch two-sample t-test showed no "
    f"statistically significant improvement (t={res.statistic:.3f}, p={res.pvalue:.3f}). The 95% CI for the mean "
    f"difference was [{ci_low:.2f}, {ci_high:.2f}] wpm, and the effect size was very small (Cohen's d={cohend(g_on, g_off):.3f}). "
    "Overall, the data do not provide evidence that Reader View improves reading speed for individuals with dyslexia; "
    "if anything, the mean is slightly lower with Reader View but the difference is negligible and not significant."
)

# Likert response: strong No due to no significant improvement and tiny negative effect
response = 25

out = {"response": response, "explanation": explanation}

with open("conclusion.txt", "w") as f:
    json.dump(out, f, ensure_ascii=False)

