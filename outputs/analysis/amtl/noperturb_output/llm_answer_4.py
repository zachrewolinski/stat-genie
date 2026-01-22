def extract_final_answer(model_output):
    """
    Extracts statistics for the 'is_human' coefficient from a fitted statsmodels GEE results object
    and returns a structured result and a concise interpretation.

    Returns:
      {
        "object": {
          "coef": float,            # log-odds coefficient for is_human
          "se": float,              # standard error of coef
          "z": float,               # z-statistic (coef / se)
          "p_value": float,         # two-sided p-value
          "ci_lower": float,        # 95% CI lower bound (log-odds)
          "ci_upper": float,        # 95% CI upper bound (log-odds)
          "odds_ratio": float,      # exp(coef)
          "or_ci_lower": float,     # exp(ci_lower)
          "or_ci_upper": float,     # exp(ci_upper)
          "percent_change": float,  # (odds_ratio - 1) * 100
          "humans_higher": bool,    # True if coef>0 and p<0.05 (significantly higher in humans)
        },
        "description": str           # plain-language interpretation
      }
    """
    # Basic validation / support for statsmodels result wrappers
    res = model_output
    # Try to obtain parameter vector, standard errors, pvalues, and conf_int
    try:
        params = getattr(res, "params")
        bse = getattr(res, "bse")
        pvalues = getattr(res, "pvalues")
        ci = res.conf_int()  # expected to be array-like or DataFrame with index of param names
    except Exception as e:
        raise ValueError(f"Provided model_output does not appear to be a fitted statsmodels/results object: {e}")

    # Ensure 'is_human' is present
    if "is_human" not in params.index:
        # sometimes the param name could be 'is_human[T.1]' or similar; try to find a matching name
        possible = [n for n in params.index if "is_human" in str(n)]
        if len(possible) == 1:
            pname = possible[0]
        else:
            raise KeyError("Could not find parameter named 'is_human' in model output parameters. "
                           f"Available parameter names: {list(params.index)}")
    else:
        pname = "is_human"

    coef = float(params[pname])
    se = float(bse[pname])
    # compute z stat and p-value (use pvalues if available)
    z = coef / se if se != 0 else float("nan")
    p = float(pvalues[pname]) if pname in pvalues.index else float("nan")

    # Confidence intervals: res.conf_int() might return numpy array or DataFrame
    try:
        if hasattr(ci, "loc"):
            ci_row = ci.loc[pname]
            ci_lower = float(ci_row[0])
            ci_upper = float(ci_row[1])
        else:
            # assume numpy array with same ordering as params.index
            idx = list(params.index).index(pname)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
    except Exception:
        # fallback: approximate via coef +/- 1.96*se
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

    # Odds ratio and CI on OR scale
    import math
    try:
        odds_ratio = math.exp(coef)
        or_ci_lower = math.exp(ci_lower)
        or_ci_upper = math.exp(ci_upper)
    except Exception:
        odds_ratio = float("nan")
        or_ci_lower = float("nan")
        or_ci_upper = float("nan")

    percent_change = (odds_ratio - 1.0) * 100.0

    # Decision: do modern humans have higher AMTL?
    humans_higher = (coef > 0) and (p < 0.05)

    # Build description
    direction = "higher" if coef > 0 else "lower"
    sig_text = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
    description = (
        f"The model coefficient for 'is_human' (comparing Homo sapiens = 1 to non-human primates = 0) "
        f"is {coef:.4f} (SE = {se:.4f}, z = {z:.2f}, p = {p:.4g}). "
        f"On the odds ratio scale this corresponds to OR = {odds_ratio:.3f} "
        f"(95% CI: {or_ci_lower:.3f} – {or_ci_upper:.3f}), i.e. a {percent_change:.1f}% change in odds. "
        f"The effect is {direction} in humans and is {sig_text} after controlling for age, sex probability, "
        f"and tooth class. "
    )
    if humans_higher:
        description += "Conclusion: modern humans have a significantly higher frequency of AMTL compared to the examined non-human primates, controlling for covariates."
    else:
        description += "Conclusion: there is no statistically significant evidence that modern humans have higher AMTL frequency compared to the examined non-human primates, after accounting for covariates."

    result_object = {
        "coef": coef,
        "se": se,
        "z": z,
        "p_value": p,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": odds_ratio,
        "or_ci_lower": or_ci_lower,
        "or_ci_upper": or_ci_upper,
        "percent_change": percent_change,
        "humans_higher": humans_higher,
        "param_name": pname
    }

    return {"object": result_object, "description": description}