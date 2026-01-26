def extract_final_answer(model_output):
    """
    Extracts the key statistics for the 'SkinDark' variable from the provided model_output.
    Expects model_output to be a dict containing at least:
      - 'results': a statsmodels results object (GLMResultsWrapper)
      - 'irr_table': a pandas DataFrame with IRR and CI indexed by variable name

    Returns:
      {
        "object": {
          "coef": float,            # estimated coefficient (log-IRR)
          "pvalue": float or None,  # p-value for the coefficient (None if unavailable)
          "IRR": float,             # incidence rate ratio (exp(coef))
          "IRR_CI_lower": float,    # lower bound of 95% CI for IRR
          "IRR_CI_upper": float,    # upper bound of 95% CI for IRR
          "significant": bool       # True if pvalue < 0.05 or CI does not include 1 (fallback)
        },
        "description": str          # brief interpretation in context
      }
    """
    # Defensive checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary containing 'results' and 'irr_table'")

    res = model_output.get('results', None)
    irr_table = model_output.get('irr_table', None)

    if res is None and irr_table is None:
        raise ValueError("model_output must contain at least one of 'results' or 'irr_table'")

    # Initialize output container
    out = {
        "coef": None,
        "pvalue": None,
        "IRR": None,
        "IRR_CI_lower": None,
        "IRR_CI_upper": None,
        "significant": None
    }

    # Try to extract coefficient and p-value from results object
    try:
        params = getattr(res, "params", None)
        if params is not None and 'SkinDark' in params.index:
            out["coef"] = float(params.loc['SkinDark'])
        # p-value (may be present when model was fit with cov_type='cluster')
        pvals = getattr(res, "pvalues", None)
        if pvals is not None and 'SkinDark' in pvals.index:
            out["pvalue"] = float(pvals.loc['SkinDark'])
    except Exception:
        # ignore and continue to use irr_table if present
        pass

    # Extract IRR and CI from irr_table if available
    if irr_table is not None:
        try:
            # ensure label exists
            if 'SkinDark' in irr_table.index:
                row = irr_table.loc['SkinDark']
            else:
                # try to find a row that case-insensitively matches
                matches = [idx for idx in irr_table.index if str(idx).lower() == 'skindark'.lower()]
                if matches:
                    row = irr_table.loc[matches[0]]
                else:
                    row = None

            if row is not None:
                # IRR and CI values
                out["IRR"] = float(row['IRR'])
                out["IRR_CI_lower"] = float(row['CI_lower'])
                out["IRR_CI_upper"] = float(row['CI_upper'])
        except Exception:
            pass

    # If coef missing but IRR present, compute log(IRR)
    if out["coef"] is None and out["IRR"] is not None:
        try:
            import math
            out["coef"] = math.log(out["IRR"])
        except Exception:
            pass

    # If p-value missing, try to infer significance from CI (CI for IRR does not include 1)
    if out["pvalue"] is None and out["IRR_CI_lower"] is not None and out["IRR_CI_upper"] is not None:
        out["significant"] = not (out["IRR_CI_lower"] <= 1.0 <= out["IRR_CI_upper"])
    else:
        # If p-value exists, determine significance by p < 0.05
        if out["pvalue"] is not None:
            out["significant"] = out["pvalue"] < 0.05

    # Build a human-readable description
    if out["IRR"] is not None:
        irr_text = f"IRR = {out['IRR']:.3f}"
        if out["IRR_CI_lower"] is not None and out["IRR_CI_upper"] is not None:
            irr_text += f" (95% CI: {out['IRR_CI_lower']:.3f}–{out['IRR_CI_upper']:.3f})"
    else:
        irr_text = "IRR not available"

    if out["pvalue"] is not None:
        p_text = f"p = {out['pvalue']:.3g}"
    else:
        p_text = "p-value not available"

    significance_text = "statistically significant" if out["significant"] else "not statistically significant"

    description = (
        "Effect of SkinDark on red card rate (offset by games, controls included). "
        f"{irr_text}; {p_text}. This means dark-skinned players are estimated to receive "
        f"{'more' if out.get('IRR', 1) > 1 else 'fewer' if out.get('IRR', 1) < 1 else 'the same number of'} "
        f"red cards per game compared to light-skinned players. The effect is {significance_text} "
        "given the model estimates and provided confidence intervals/p-values. "
        "Interpretation is conditional on the model specification (controls included and referee-clustered SEs)."
    )

    return {
        "object": out,
        "description": description
    }