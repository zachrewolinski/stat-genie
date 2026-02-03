import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("mortgage.csv")
    # Drop index-like column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Basic cleaning: drop rows with any missing values in model columns
    outcome = "accept"
    predictors = [
        "female",
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
        "denied_PMI",
    ]

    model_df = df[[outcome] + predictors].dropna().copy()

    y = model_df[outcome]
    # Unadjusted model: accept ~ female (female is noisy/continuous here)
    X_unadj = sm.add_constant(model_df[["female"]])
    unadj_model = sm.Logit(y, X_unadj)
    unadj_result = unadj_model.fit(disp=False)

    # Logistic regression with controls
    X = sm.add_constant(model_df[predictors])
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    female_coef = result.params["female"]
    female_p = result.pvalues["female"]

    # Predicted probability difference at mean covariates
    mean_vals = X.mean()
    mean_male = mean_vals.copy()
    mean_female = mean_vals.copy()
    mean_male["female"] = 0
    mean_female["female"] = 1
    pred_male = result.predict(mean_male)[0]
    pred_female = result.predict(mean_female)[0]
    pred_diff = pred_female - pred_male

    # Print key results for reference
    print("Unadjusted logit coefficient for female:", unadj_result.params["female"])
    print("Unadjusted logit p-value for female:", unadj_result.pvalues["female"])
    print("\nAdjusted logit coefficient for female:", female_coef)
    print("Adjusted logit p-value for female:", female_p)
    print("Predicted acceptance (male at means):", pred_male)
    print("Predicted acceptance (female at means):", pred_female)
    print("Predicted acceptance difference (female - male):", pred_diff)


if __name__ == "__main__":
    main()
