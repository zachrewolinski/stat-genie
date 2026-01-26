def extract_final_answer(model_output):
    """
    Extracts statistics relevant to how reliance on the majority changes with age
    (and across cultures to the extent estimated) from the provided model_output.

    Expects model_output to be the dict returned by the modeling function in the prompt,
    containing keys 'social_choice_model', 'majority_choice_model', and 'mnlogit_model'.

    Returns a dict with keys:
      - "object": dict of extracted numeric results (coefficients, SEs, p-values, ORs, CIs)
      - "description": human-readable interpretation of those results in this study context
    """
    import numpy as np
    import pandas as pd

    out = {"object": None, "description": None}

    # Check for multinomial model presence
    mn = model_output.get('mnlogit_model', None)
    if mn is None:
        out["object"] = {"error": "mnlogit_model not present in model_output"}
        out["description"] = "No multinomial model available to extract age/culture effects."
        return out

    # If mn is an error dict, return that
    if isinstance(mn, dict) and 'error' in mn:
        out["object"] = {"error": mn['error']}
        out["description"] = "Multinomial model failed; cannot extract statistics."
        return out

    # At this point mn should be a statsmodels MNLogit results wrapper
    try:
        params = mn.params  # DataFrame: rows = non-baseline categories, cols = exog names
        pvalues = mn.pvalues
        bse = mn.bse
    except Exception as e:
        out["object"] = {"error": f"Could not read params/pvalues/bse from mnlogit_model: {e}"}
        out["description"] = "Model object does not have expected attributes."
        return out

    # Determine which row corresponds to the 'majority' category.
    # With the original coding y_mn = y - min(y), if min(y)=1 then mapping is:
    # 0 = unchosen (baseline), 1 = majority, 2 = minority.
    # MNLogit reports rows in increasing order of endogenous categories (excluding baseline),
    # so the first row should correspond to majority.
    try:
        # Prefer index value 1 if present, else take first row
        if 1 in params.index:
            maj_row = 1
        else:
            maj_row = params.index[0]
        # minority row if present
        if len(params.index) >= 2:
            if 2 in params.index:
                min_row = 2
            else:
                # if first is majority, second is minority
                min_row = params.index[1]
        else:
            min_row = None
    except Exception:
        maj_row = params.index[0]
        min_row = params.index[1] if len(params.index) >= 2 else None

    results = {}

    # Helper to safely extract a coefficient, se, pval for a given row and variable name
    def _get_stat(row, var):
        try:
            coef = float(params.loc[row, var])
            se_val = float(bse.loc[row, var])
            pval = float(pvalues.loc[row, var])
            z = coef / se_val if se_val != 0 else np.nan
            # 95% Wald CI on log-odds scale
            ci_lower = coef - 1.96 * se_val
            ci_upper = coef + 1.96 * se_val
            # exponentiate to get odds ratio scale for interpretation (approx. multiplicative change in odds)
            or_est = float(np.exp(coef))
            or_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))
            return {
                "coef": coef,
                "se": se_val,
                "z": z,
                "p": pval,
                "ci_95": (ci_lower, ci_upper),
                "OR": or_est,
                "OR_95": or_ci
            }
        except Exception:
            return None

    # Extract age coefficient for majority vs unchosen
    age_var = 'age_centered'
    maj_age_stats = _get_stat(maj_row, age_var)
    results['majority_vs_unchosen_age'] = maj_age_stats

    # Extract age coefficient for minority vs unchosen, if available
    if min_row is not None:
        min_age_stats = _get_stat(min_row, age_var)
    else:
        min_age_stats = None
    results['minority_vs_unchosen_age'] = min_age_stats

    # Robustly identify culture main effect columns for majority vs unchosen.
    # Column labels may not all be plain strings (could be ints, tuples, etc.), so
    # normalize to strings for detection but keep original label for indexing.
    def _col_str(c):
        # Return a readable string for a column label
        if isinstance(c, str):
            return c
        if isinstance(c, (list, tuple)):
            # try to find a string element that looks like a column name
            for part in c:
                if isinstance(part, str):
                    return part
            # fallback to stringifying the first element
            return str(c[0]) if len(c) > 0 else str(c)
        return str(c)

    try:
        culture_cols = [c for c in params.columns if _col_str(c).startswith('culture_')]
    except Exception:
        # If params.columns isn't iterable as expected, set empty
        culture_cols = []

    culture_effects = {}
    for c in culture_cols:
        stat = _get_stat(maj_row, c)
        culture_effects[_col_str(c)] = stat
    results['majority_vs_unchosen_culture_effects'] = culture_effects

    # Note about interactions: the original logistic models included age x culture interactions,
    # but the MNLogit here was fit without interactions (to aid convergence). So we cannot
    # directly assess culture-specific age slopes from this MNLogit.
    # We include a short summary interpretation below.

    out["object"] = results

    # Build description:
    desc_lines = []
    desc_lines.append(
        "Extracted statistics are from the multinomial logistic model predicting choice (unchosen=baseline, "
        "majority, minority). Coefficients are log-odds differences vs the baseline (unchosen)."
    )
    if maj_age_stats is not None:
        desc_lines.append(
            f"Age effect (majority vs unchosen): coef = {maj_age_stats['coef']:.3f}, SE = {maj_age_stats['se']:.3f}, "
            f"z = {maj_age_stats['z']:.3f}, p = {maj_age_stats['p']:.3g}. "
            f"Odds ratio per year (exp(coef)) = {maj_age_stats['OR']:.3f} "
            f"(95% CI on OR = [{maj_age_stats['OR_95'][0]:.3f}, {maj_age_stats['OR_95'][1]:.3f}])."
        )
        if maj_age_stats['p'] < 0.05:
            desc_lines.append("Interpretation: Older children have significantly higher (or lower if OR<1) odds of choosing the majority option vs an unchosen option per year of age (age was mean-centered).")
        else:
            desc_lines.append("Interpretation: The effect of age on choosing the majority option vs an unchosen option is not statistically significant at alpha=0.05.")
    else:
        desc_lines.append("Could not extract the age coefficient for majority vs unchosen.")

    if min_age_stats is not None:
        desc_lines.append(
            f"Age effect (minority vs unchosen): coef = {min_age_stats['coef']:.3f}, SE = {min_age_stats['se']:.3f}, p = {min_age_stats['p']:.3g}."
        )
    else:
        desc_lines.append("Minority vs unchosen equation not available or age coefficient not extractable.")

    # Culture main effects summary (majority vs unchosen)
    if culture_effects:
        non_null = {k: v for k, v in culture_effects.items() if v is not None}
        if non_null:
            desc_lines.append("Culture main effects on majority vs unchosen (coefficients are log-odds relative to reference site):")
            for k, v in non_null.items():
                desc_lines.append(f"  {k}: coef={v['coef']:.3f}, p={v['p']:.3g}, OR={v['OR']:.3f}")
            desc_lines.append(
                "Note: Interactions age x culture were not included in this MNLogit model, so these are baseline cross-cultural differences in majority choice, not culture-specific developmental slopes."
            )
        else:
            desc_lines.append("Culture dummy coefficients were not extractable.")
    else:
        desc_lines.append("No culture dummy columns found in model output.")

    # Mention that original logistic models with interactions failed to return robust results
    sc_err = model_output.get('social_choice_model')
    maj_err = model_output.get('majority_choice_model')
    if isinstance(sc_err, dict) and 'error' in sc_err:
        desc_lines.append("Note: The separate logistic models that included age x culture interactions failed (see social_choice_model error). We therefore rely on the MNLogit main-effects model; direct tests of age-by-culture moderation are not available from that model.")
    if isinstance(maj_err, dict) and 'error' in maj_err:
        desc_lines.append("Note: The majority_choice_model also failed; cannot extract interaction estimates from that model.")

    out["description"] = "\n".join(desc_lines)
    return out