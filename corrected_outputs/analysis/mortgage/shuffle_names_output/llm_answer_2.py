def extract_final_answer(model_output):
    """
    Extracts statistics for the 'Female' coefficient from a fitted statsmodels Logit result.

    Returns:
      {
        "object": {
           "coef": float,                # log-odds coefficient for Female
           "std_err": float or None,     # standard error
           "p_value": float or None,     # p-value
           "odds_ratio": float,          # exp(coef)
           "odds_ratio_pct_change": float, # (odds_ratio - 1) * 100
           "ci_odds_ratio": [low, high], # 95% CI for odds ratio
           "nobs": int or None,          # number of observations used
           "significant_0.05": bool or None
        },
        "description": "Interpretation text..."
      }
    If the model output does not contain the 'Female' parameter, returns a descriptive error.
    """
    import numpy as np

    try:
        res = model_output

        # params, bse, pvalues should be available on a statsmodels result
        params = getattr(res, "params", None)
        if params is None:
            return {
                "object": None,
                "description": "Model output has no 'params' attribute. Expected a statsmodels results object."
            }

        # Ensure 'Female' is present
        if "Female" not in params.index:
            return {
                "object": None,
                "description": "The fitted model does not contain a parameter named 'Female'."
            }

        coef = float(params.loc["Female"])
        # standard error
        bse = getattr(res, "bse", None)
        std_err = float(bse.loc["Female"]) if (bse is not None and "Female" in bse.index) else None
        # p-value
        pvalues = getattr(res, "pvalues", None)
        p_value = float(pvalues.loc["Female"]) if (pvalues is not None and "Female" in pvalues.index) else None

        # odds ratio and CI: compute from coef and conf_int()
        odds_ratio = float(np.exp(coef))
        ci = None
        try:
            conf_int = res.conf_int()
            # conf_int may be a DataFrame or ndarray; handle both
            if hasattr(conf_int, "loc") and "Female" in conf_int.index:
                ci_low, ci_high = float(conf_int.loc["Female", 0]), float(conf_int.loc["Female", 1])
            else:
                # fallback if conf_int is ndarray with same ordering as params
                # find position of Female in params.index
                idx = list(params.index).index("Female")
                ci_low, ci_high = float(conf_int[idx, 0]), float(conf_int[idx, 1])
            ci_odds = [float(np.exp(ci_low)), float(np.exp(ci_high))]
        except Exception:
            ci_odds = [None, None]

        # number of observations if available
        nobs = None
        try:
            nobs = int(getattr(getattr(res, "model", None), "nobs", None))
        except Exception:
            nobs = None

        significant = None
        if p_value is not None:
            significant = (p_value < 0.05)

        # percent change in odds
        odds_pct_change = (odds_ratio - 1.0) * 100.0

        # Build the object to return
        result_object = {
            "coef": coef,
            "std_err": std_err,
            "p_value": p_value,
            "odds_ratio": odds_ratio,
            "odds_ratio_pct_change": odds_pct_change,
            "ci_odds_ratio": ci_odds,
            "nobs": nobs,
            "significant_0.05": significant
        }

        # Short interpretation
        if odds_ratio > 1:
            direction = "higher"
        elif odds_ratio < 1:
            direction = "lower"
        else:
            direction = "no change in"

        sig_text = ""
        if significant is True:
            sig_text = " This effect is statistically significant at the 5% level (p < 0.05)."
        elif significant is False:
            sig_text = " This effect is not statistically significant at the 5% level (p >= 0.05)."

        ci_text = ""
        if ci_odds[0] is not None and ci_odds[1] is not None:
            ci_text = f" The 95% CI for the odds ratio is [{ci_odds[0]:.3f}, {ci_odds[1]:.3f}]."

        description = (
            f"The log-odds coefficient for 'Female' is {coef:.4f}. "
            f"This corresponds to an odds ratio of {odds_ratio:.3f}, i.e. females have "
            f"{abs(odds_pct_change):.1f}% {direction} odds of mortgage approval compared to males.{ci_text}"
            f" The p-value for the coefficient is {p_value:.4g}.{sig_text}"
            + (f" Number of observations used: {nobs}." if nobs is not None else "")
        )

        return {"object": result_object, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting the 'Female' coefficient: {e}"
        }