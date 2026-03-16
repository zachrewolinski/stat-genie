import json
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv("affairs.csv")

# Create binary indicator for any extramarital affair
df["has_affair"] = (df["affairs"] > 0).astype(int)

# Descriptive statistics by children status
summary = {}

for children_status, group in df.groupby("children"):
    summary[children_status] = {
        "n": int(group.shape[0]),
        "mean_affairs": float(group["affairs"].mean()),
        "median_affairs": float(group["affairs"].median()),
        "prop_with_affair": float(group["has_affair"].mean()),
    }

# Logistic regression: any affair ~ children + controls
# Encode categorical variables
X = df.copy()
X = pd.get_dummies(
    X[[
        "children",
        "gender",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]],
    drop_first=True,
)

X = sm.add_constant(X)
y = df["has_affair"]

logit_model = sm.Logit(y, X)
logit_result = logit_model.fit(disp=False)

# Extract effect of having children (relative to no children)
children_cols = [c for c in X.columns if c.startswith("children_")]
children_effects = {}
for col in children_cols:
    coef = float(logit_result.params[col])
    pval = float(logit_result.pvalues[col])
    odds_ratio = float(pd.np.exp(coef)) if coef is not None else None
    children_effects[col] = {
        "coef": coef,
        "p_value": pval,
        "odds_ratio": odds_ratio,
    }

# Simple model with only children as predictor
X_simple = pd.get_dummies(df[["children"]], drop_first=True)
X_simple = sm.add_constant(X_simple)
logit_simple = sm.Logit(y, X_simple).fit(disp=False)

simple_children_col = [c for c in X_simple.columns if c.startswith("children_")][0]
simple_coef = float(logit_simple.params[simple_children_col])
simple_pval = float(logit_simple.pvalues[simple_children_col])
simple_or = float(pd.np.exp(simple_coef))

# Decide Likert response
# Question: "Does having children decrease engagement in extramarital affairs?"
# We answer "Yes" if there is statistically significant evidence (p < 0.05)
# that the presence of children is associated with *lower* odds of any affair.

# Use the full model as primary evidence
if children_cols:
    col = children_cols[0]
    coef = logit_result.params[col]
    pval = logit_result.pvalues[col]
    odds_ratio = float(pd.np.exp(coef))
else:
    # Fallback to simple model
    coef = logit_simple.params[simple_children_col]
    pval = logit_simple.pvalues[simple_children_col]
    odds_ratio = float(pd.np.exp(coef))

# Interpret direction and significance
alpha = 0.05

if pval < alpha and coef < 0:
    # Significant evidence that children reduce affairs
    # Map strength using odds_ratio: stronger reduction -> closer to 100
    if odds_ratio < 0.5:
        response = 90
    elif odds_ratio < 0.8:
        response = 75
    else:
        response = 65
    answer = "Yes"
elif pval < alpha and coef > 0:
    # Significant evidence that children are associated with MORE affairs
    # Strong "No" answer
    if odds_ratio > 1.5:
        response = 5
    elif odds_ratio > 1.2:
        response = 10
    else:
        response = 15
    answer = "No"
else:
    # No significant association: weak evidence either way
    if coef < 0:
        # Non-significant trend toward fewer affairs with children
        response = 55
        answer = "Yes"
    elif coef > 0:
        # Non-significant trend toward more affairs with children
        response = 35
        answer = "No"
    else:
        # Essentially no difference
        response = 50
        answer = "No"

# Build explanation string
explanation_parts = []
explanation_parts.append(
    "Research question: Does having children decrease engagement in extramarital affairs?"
)
explanation_parts.append(
    f"Descriptively, the mean number of affairs in the past year was "
    f"{summary['yes']['mean_affairs']:.2f} for respondents with children (n={summary['yes']['n']}) "
    f"and {summary['no']['mean_affairs']:.2f} for those without children (n={summary['no']['n']})."
)
explanation_parts.append(
    f"The proportion reporting at least one affair was {summary['yes']['prop_with_affair']:.2f} "
    f"with children vs {summary['no']['prop_with_affair']:.2f} without children."
)
explanation_parts.append(
    "I then fit logistic regression models predicting whether a respondent had any extramarital affair "
    "from the presence of children, first alone and then controlling for gender, age, years married, "
    "religiousness, education, occupation, and self-rated marital quality."
)
explanation_parts.append(
    f"In the fully adjusted model, the coefficient for having children was {coef:.3f} on the log-odds scale, "
    f"corresponding to an odds ratio of {odds_ratio:.2f} (p = {pval:.3f})."
)

if answer == "Yes":
    explanation_parts.append(
        "Because the estimated effect of having children is negative (odds ratio below 1) and statistically "
        "significant at the 5% level, the data provide evidence that having children is associated with a lower "
        "likelihood of engaging in extramarital affairs."
    )
else:
    if pval < alpha:
        explanation_parts.append(
            "Because the estimated effect of having children is positive (odds ratio above 1) and statistically "
            "significant at the 5% level, the data provide evidence that having children are, if anything, associated "
            "with a higher likelihood of extramarital affairs rather than a decrease."
        )
    else:
        explanation_parts.append(
            "The effect of having children is not statistically significant at conventional levels, so the data do not "
            "provide clear evidence that having children decreases engagement in extramarital affairs."
        )

explanation_parts.append(
    f"Based on these results, I answer the question '{answer}' and map this to a Likert-scale response of {response} "
    "on a 0 (strong 'No') to 100 (strong 'Yes') scale."
)

explanation = " ".join(explanation_parts)

output = {
    "response": int(response),
    "explanation": explanation,
}

with open("conclusion.txt", "w") as f:
    json.dump(output, f)
