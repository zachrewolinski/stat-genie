def extract_final_answer(model_output):
    """
    Extracts species-related statistics from a fitted statsmodels GLMResults-like object
    for the AMTL model and returns a summary useful for answering whether modern humans
    (Homo sapiens) show higher AMTL after adjustment.

    Returns a dict with keys:
      - "object": a dictionary with extracted numeric results (or None if not available)
      - "description": human-readable interpretation and notes about limitations / how to
                       directly test Homo vs non-human primates.

    The function handles the common case where the model is a statsmodels results object
    with attributes: params, bse, pvalues, conf_int, and uses dummy variable names that
    begin with the prefix "Species_".
    """
    import math
    import numpy as np
    import pandas as pd

    if model_output is None:
        return {
            "object": None,
            "description": (
                "No model output was provided (model_output is None). "
                "This means the GLM was not fitted (e.g., no data after filtering), "
                "so we cannot extract coefficients or answer whether Homo sapiens have "
                "higher AMTL."
            )
        }

    # Try to detect a statsmodels-like results object
    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = model_output.bse
        conf = model_output.conf_int()  # DataFrame with index = parameter names
    except Exception as e:
        return {
            "object": None,
            "description": (
                "The provided model_output does not look like a statsmodels results object "
                "with attributes params/pvalues/bse/conf_int. Error when accessing these: "
                f"{repr(e)}"
            )
        }

    # Convert params to a pandas Series for consistent indexing
    params = pd.Series(params)
    pvalues = pd.Series(pvalues)
    try:
        bse = pd.Series(bse)
    except Exception:
        # bse might be an array-like; convert if possible
        bse = pd.Series(np.asarray(bse), index=params.index)

    # Ensure conf is a DataFrame with indexed rows
    if isinstance(conf, pd.DataFrame):
        conf_df = conf
    else:
        # conf_int sometimes returns a numpy array; try to coerce
        try:
            conf_df = pd.DataFrame(conf, index=params.index)
        except Exception:
            conf_df = None

    exog_names = list(params.index)

    # Identify species dummy coefficients (those created as "Species_<level>")
    species_prefix = "Species_"
    species_terms = [nm for nm in exog_names if nm.startswith(species_prefix)]

    if len(species_terms) == 0:
        # No explicit Species_ terms found
        return {
            "object": None,
            "description": (
                "No model coefficients named with the prefix 'Species_' were found in the model. "
                "Either species was not included in the model, was encoded differently, or the "
                "species variable was the omitted reference level. Without species-term coefficients "
                "we cannot directly extract species comparisons from this fitted object. "
                "To directly test whether Homo sapiens differ from non-human primates, refit the model "
                "with a binary indicator (Homo_sapiens vs non-human) or create explicit contrasts."
            )
        }

    # Build a results dict for each species dummy
    species_effects = {}
    for term in species_terms:
        sp_name = term[len(species_prefix):]
        coef = float(params[term])
        se = float(bse[term]) if term in bse.index else None
        pval = float(pvalues[term]) if term in pvalues.index else None
        if conf_df is not None and term in conf_df.index:
            ci_low = float(conf_df.loc[term, 0])
            ci_high = float(conf_df.loc[term, 1])
        else:
            ci_low = ci_high = None

        # Compute odds ratio and CI when possible
        try:
            or_est = math.exp(coef)
            or_ci_low = math.exp(ci_low) if ci_low is not None else None
            or_ci_high = math.exp(ci_high) if ci_high is not None else None
        except Exception:
            or_est = or_ci_low = or_ci_high = None

        species_effects[sp_name] = {
            "term_name": term,
            "coef_log_odds": coef,
            "se": se,
            "p_value": pval,
            "conf_int_log_odds": (ci_low, ci_high),
            "odds_ratio": or_est,
            "conf_int_odds_ratio": (or_ci_low, or_ci_high),
            "interpretation": (
                "This coefficient is the log-odds difference in AMTL for this species "
                "compared to the omitted (reference) species in the model. "
                "A positive coefficient means higher odds of AMTL in this species "
                "relative to the reference; a negative coefficient means lower odds."
            )
        }

    # Try to detect whether Homo sapiens appears as a dummy or was the reference (omitted)
    homo_key = None
    for sp in species_effects.keys():
        # look for common ways Homo sapiens may appear in the original labels
        if ("Homo" in sp) or ("homo" in sp) or ("sapiens" in sp) or ("H. sapiens" in sp):
            homo_key = sp
            break

    # Prepare explanatory description
    if homo_key is not None:
        coef = species_effects[homo_key]["coef_log_odds"]
        pval = species_effects[homo_key]["p_value"]
        or_est = species_effects[homo_key]["odds_ratio"]
        desc = (
            f"Homo sapiens appears among the species dummy variables as '{homo_key}'.\n"
            f"Coefficient (log-odds) = {coef:.4g}; p-value = {pval:.4g}; odds ratio = "
            f"{or_est:.4g}.\n"
            "This coefficient tests Homo sapiens versus the omitted (reference) species in the model. "
            "If you want a single test of 'Homo sapiens vs all non-human primates grouped', "
            "you should either (1) refit the model with a binary predictor (Homo vs non-Homo) or "
            "(2) request explicit contrasts (e.g., pairwise linear combinations) comparing Homo to "
            "each non-human genus and combine / interpret those results. Negative coefficients for "
            "other species indicate that those species have lower AMTL than the omitted reference; "
            "if the omitted reference is a non-human species, negative coefficients for that species "
            "would imply Homo sapiens (if omitted) has higher AMTL."
        )
    else:
        # Homo likely the omitted/reference level; interpret other species relative to Homo
        desc = (
            "Homo sapiens does not appear among the model's 'Species_' dummy terms; this very likely "
            "means Homo sapiens was used as the omitted (reference) level when creating dummies. "
            "In that case, each reported 'Species_<X>' coefficient is the log-odds difference of species X "
            "relative to Homo sapiens. For those coefficients:\n"
            "  - A negative coefficient for Species_X means species X has LOWER AMTL than Homo sapiens "
            "    (i.e., Homo sapiens has HIGHER AMTL than species X),\n"
            "  - A positive coefficient means species X has HIGHER AMTL than Homo sapiens.\n\n"
            "The returned 'object' contains the per-species coefficients, p-values, confidence intervals, "
            "and odds ratios. To get a single yes/no answer to whether modern humans have higher AMTL "
            "than the non-human genera taken together, you should either (A) refit the model with a "
            "binary predictor 'IsHomo' (Homo sapiens vs all others) or (B) compute and test a contrast "
            "that averages/comparisons across the non-human genera. The current multi-level coding only "
            "gives comparisons of each genus versus the omitted reference."
        )

    result_object = {
        "species_effects": species_effects,
        "homo_dummy_present": homo_key is not None,
        "homo_dummy_name": homo_key
    }

    return {
        "object": result_object,
        "description": desc
    }