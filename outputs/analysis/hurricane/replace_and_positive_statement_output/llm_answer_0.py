def extract_final_answer(model_output):
    """
    Extracts key statistics for the primary independent variables from a fitted statsmodels GLMResultsWrapper
    (negative binomial model was used in the example). Returns a dictionary with numeric results and a
    short interpretation relative to the hypothesis.

    Returns:
      {
        "object": {
          "masfem_std": {
            "coef": float,
            "se": float,
            "pvalue": float,
            "ci_lower": float,
            "ci_upper": float,
            "irr": float,               # exp(coef)
            "irr_ci_lower": float,
            "irr_ci_upper": float,
            "significant": bool
          },
          "gender_female": { ... } or a message if not present
        },
        "description": "A short interpretation of the masfem_std result relative to the hypothesis..."
      }
    """
    import numpy as np
    import pandas as pd

    # Variables of interest
    vars_of_interest = ["masfem_std", "gender_female"]

    # Prepare output containers
    extracted = {}
    missing_vars = []

    # Basic checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    params = model_output.params
    pvalues = getattr(model_output, "pvalues", None)
    bse = getattr(model_output, "bse", None)

    # confidence intervals: statsmodels' conf_int() returns a DataFrame/ndarray
    try:
        ci = model_output.conf_int()
        # Make sure it's a DataFrame with variable names as index if possible
        if not isinstance(ci, pd.DataFrame):
            ci = pd.DataFrame(ci, index=params.index, columns=[0, 1])
    except Exception:
        # fallback: compute from params +/- 1.96*bse if bse available
        if bse is not None:
            ci = pd.DataFrame({
                0: params - 1.96 * bse,
                1: params + 1.96 * bse
            }, index=params.index)
        else:
            ci = None

    for v in vars_of_interest:
        if v not in params.index:
            missing_vars.append(v)
            extracted[v] = {"error": f"Variable '{v}' not present in model params."}
            continue

        coef = float(params.loc[v])
        se = float(bse.loc[v]) if bse is not None and v in bse.index else None
        pv = float(pvalues.loc[v]) if pvalues is not None and v in pvalues.index else None

        if ci is not None and v in ci.index:
            ci_lower = float(ci.loc[v, 0])
            ci_upper = float(ci.loc[v, 1])
        else:
            ci_lower = None
            ci_upper = None

        # Incidence rate ratio and its CI (on multiplicative scale)
        irr = float(np.exp(coef))
        irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

        significant = (pv is not None) and (pv < 0.05)

        extracted[v] = {
            "coef": coef,
            "se": se,
            "pvalue": pv,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "irr": irr,
            "irr_ci_lower": irr_ci_lower,
            "irr_ci_upper": irr_ci_upper,
            "significant": bool(significant)
        }

    # Short interpretation for the primary hypothesis (masfem_std)
    if "masfem_std" in extracted and "error" not in extracted["masfem_std"]:
        e = extracted["masfem_std"]
        # Coefficient is on log count scale; IRR >1 means higher expected fatalities as masfem_std increases.
        if e["pvalue"] is None:
            conclusion = ("Could not determine statistical significance for masfem_std (p-value missing). "
                          "Estimated IRR = {:.3f} (95% CI [{:.3f}, {:.3f}])."
                          .format(e["irr"],
                                  e["irr_ci_lower"] if e["irr_ci_lower"] is not None else float("nan"),
                                  e["irr_ci_upper"] if e["irr_ci_upper"] is not None else float("nan")))
        else:
            if e["significant"]:
                if e["coef"] > 0:
                    conclusion = ("Result consistent with the hypothesis: higher femininity (masfem_std) is "
                                  "associated with higher fatalities. "
                                  "Estimated coef = {:.4f} (SE = {:.4f}, p = {:.3g}), IRR = {:.3f} "
                                  "(95% CI [{:.3f}, {:.3f}])."
                                  .format(e["coef"], e["se"], e["pvalue"], e["irr"],
                                          e["irr_ci_lower"], e["irr_ci_upper"]))
                else:
                    conclusion = ("Result contradicts the hypothesis: higher femininity (masfem_std) is "
                                  "associated with lower fatalities. "
                                  "Estimated coef = {:.4f} (SE = {:.4f}, p = {:.3g}), IRR = {:.3f} "
                                  "(95% CI [{:.3f}, {:.3f}])."
                                  .format(e["coef"], e["se"], e["pvalue"], e["irr"],
                                          e["irr_ci_lower"], e["irr_ci_upper"]))
            else:
                conclusion = ("No statistically significant evidence that name femininity (masfem_std) is associated "
                              "with fatalities at alpha=0.05. "
                              "Estimated coef = {:.4f} (SE = {:.4f}, p = {:.3g}), IRR = {:.3f} "
                              "(95% CI [{:.3f}, {:.3f}])."
                              .format(e["coef"], e["se"], e["pvalue"], e["irr"],
                                      e["irr_ci_lower"], e["irr_ci_upper"]))
    else:
        conclusion = "masfem_std not available in model output; cannot draw conclusion."

    # Compose final description including note on scale/interpretation
    description = (
        "Extracted coefficients are from a Negative Binomial GLM predicting total fatalities (alldeaths).\n"
        "- Coefficients are on the log count scale. Exponentiated coefficients (IRR = exp(coef)) indicate "
        "multiplicative change in expected fatalities for a one-unit increase in the predictor.\n\n"
        "Primary finding (masfem_std):\n" + conclusion +
        "\n\nSecondary variable (gender_female): similar interpretation; positive coef => female-named storms "
        "have higher expected fatalities compared to male-named storms (IRR > 1).\n"
        "If you want additional fields (e.g., other covariates or a different significance threshold), "
        "call this function and adapt the vars_of_interest list."
    )

    return {"object": extracted, "description": description}