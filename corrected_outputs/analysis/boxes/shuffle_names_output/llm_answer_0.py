def extract_final_answer(model_output):
    """
    Extract statistics relevant to developmental change in reliance on the majority
    across cultures from a fitted statsmodels GLM (binomial) with formula:
      MajorityChoice ~ Age * C(Culture) + IsMale + MajorityDemoFirst

    Returns:
      {
        "object": {
          "reference_culture": <name>,
          "per_culture_slope": {
            "<culture>": {
               "log_odds_per_year": coef,
               "se": se,
               "z": z,
               "p": p,
               "odds_ratio_per_year": or_,
               "odds_ratio_95CI": [or_low, or_high]
            }, ...
          },
          "interaction_term_stats": {
            "Age:C(Culture)[T.<culture>]": {
               "coef": ...,
               "se": ...,
               "z": ...,
               "p": ...,
               "95CI_log_odds": [low, high]
            }, ...
          },
          "age_main_effect": { ... }  # stats for the Age main effect (reference culture slope)
        },
        "description": "Brief explanation..."
      }
    """
    import re
    import numpy as np
    import pandas as pd

    # for p-value calculation
    try:
        from scipy.stats import norm
    except Exception:
        # fallback using statsmodels if scipy not available
        import statsmodels.api as sm
        norm = sm.distributions.norm

    # Basic checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels model (missing .params).")

    params = model_output.params
    cov = model_output.cov_params()  # covariance matrix of parameter estimates
    conf = None
    try:
        conf = model_output.conf_int()
    except Exception:
        conf = None

    # Attempt to recover the original data to get list of cultures
    cultures = None
    try:
        df = model_output.model.data.frame
        if 'Culture' in df.columns:
            # preserve the order present in the data (if categorical with categories set, it will reflect that)
            cultures = list(pd.Categorical(df['Culture']).categories)
        else:
            cultures = None
    except Exception:
        df = None
        cultures = None

    # Identify interaction terms of the form Age:C(Culture)[T.<level>]
    interaction_pattern = re.compile(r"^Age:C\(Culture\)\[T\.?(.*)\]$")
    main_culture_pattern = re.compile(r"^C\(Culture\)\[T\.?(.*)\]$")

    interaction_terms = {}
    culture_dummy_terms = {}
    for name in params.index:
        m = interaction_pattern.match(name)
        if m:
            lvl = m.group(1)
            interaction_terms[lvl] = name
        m2 = main_culture_pattern.match(name)
        if m2:
            lvl2 = m2.group(1)
            culture_dummy_terms[lvl2] = name

    # If we have df and cultures list, determine reference culture as the one not present in dummy terms.
    ref_culture = None
    if cultures is not None:
        # cultures come as numpy array-like; convert to strings to compare with parsed names
        cultures_str = [str(c) for c in cultures]
        # find which culture is not present among dummy terms
        present_dummy_levels = set(culture_dummy_terms.keys())
        # Some levels in cultures_str may contain special characters; interaction parsing uses literal name.
        candidates = [c for c in cultures_str if c not in present_dummy_levels]
        if len(candidates) == 1:
            ref_culture = candidates[0]
        elif len(candidates) > 1:
            # ambiguous, pick first as fallback
            ref_culture = candidates[0]
        else:
            # none found -> fall back to deducing from params: choose the implicit reference as one not in dummy terms
            all_levels = cultures_str
            for c in all_levels:
                if c not in culture_dummy_terms:
                    ref_culture = c
                    break
    else:
        # No data frame available: attempt to deduce from parameter names vs known levels
        # If there are k interaction terms, assume reference is "reference" (unknown).
        ref_culture = None

    results_per_culture = {}

    # Get Age main effect stats (this is the slope for the reference culture)
    if 'Age' not in params.index:
        raise ValueError("Model does not contain an 'Age' main effect term named exactly 'Age'.")
    age_coef = float(params['Age'])
    # standard error for Age from covariance matrix
    age_se = float(np.sqrt(cov.loc['Age', 'Age']))
    age_z = age_coef / age_se if age_se > 0 else np.nan
    age_p = 2 * (1 - norm.cdf(abs(age_z)))
    # 95% CI on log-odds
    age_ci_low = age_coef - 1.96 * age_se
    age_ci_high = age_coef + 1.96 * age_se
    # odds ratio and CI
    age_or = float(np.exp(age_coef))
    age_or_ci = [float(np.exp(age_ci_low)), float(np.exp(age_ci_high))]

    # Store age main effect (interpreted as reference culture slope)
    age_main_effect = {
        "log_odds_per_year": age_coef,
        "se": age_se,
        "z": age_z,
        "p": age_p,
        "95CI_log_odds": [age_ci_low, age_ci_high],
        "odds_ratio_per_year": age_or,
        "odds_ratio_95CI": age_or_ci
    }

    # For each culture, compute slope = Age (reference) + Age:C(Culture)[T.<culture>] (if present)
    # If reference culture unknown, list reference as "reference (unspecified in params)"
    if ref_culture is None:
        # construct cultures from interaction terms keys + assume one omitted as reference; pick first present as reference placeholder
        inferred_levels = sorted(list(set(list(interaction_terms.keys()))))
        ref_culture = "(reference - not present in coefficient names)"
        # we'll compute slopes for listed interaction levels + a "reference" entry
        culture_list = [ref_culture] + inferred_levels
    else:
        # use cultures order inferred earlier if available, else combine ref + interaction keys
        if cultures is not None:
            culture_list = list(cultures_str)
        else:
            culture_list = [ref_culture] + sorted(list(interaction_terms.keys()))

    # covariance matrix available as DataFrame or ndarray; ensure indexing works
    cov_df = cov
    # ensure cov_df has index labels
    if not isinstance(cov_df, pd.DataFrame):
        cov_df = pd.DataFrame(cov_df, index=params.index, columns=params.index)

    # Build per-culture stats
    for c in culture_list:
        if c == ref_culture and ref_culture.startswith("("):
            # special placeholder reference; slope is age main effect
            slope = age_coef
            # se is se of age
            se = age_se
        elif c == ref_culture:
            slope = age_coef
            se = age_se
        else:
            # check if there is an interaction term for this culture
            inter_name = interaction_terms.get(c)
            if inter_name is None:
                # no explicit interaction term found: slope equals age_coef (no difference)
                slope = age_coef
                se = age_se
            else:
                # slope = Age + Age:C(Culture)[T.c]
                slope = age_coef + float(params.get(inter_name, 0.0))
                # variance = var(Age) + var(inter) + 2*cov(Age, inter)
                var_age = float(cov_df.loc['Age', 'Age'])
                var_inter = float(cov_df.loc[inter_name, inter_name])
                cov_ai = float(cov_df.loc['Age', inter_name])
                var_sum = var_age + var_inter + 2.0 * cov_ai
                se = float(np.sqrt(var_sum)) if var_sum >= 0 else np.nan

        z = slope / se if se and not np.isnan(se) else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low = slope - 1.96 * se if not np.isnan(se) else np.nan
        ci_high = slope + 1.96 * se if not np.isnan(se) else np.nan
        or_ = float(np.exp(slope)) if not np.isnan(slope) else np.nan
        or_ci = [float(np.exp(ci_low)), float(np.exp(ci_high))] if not np.isnan(ci_low) else [np.nan, np.nan]

        results_per_culture[c] = {
            "log_odds_per_year": float(slope),
            "se": float(se) if not np.isnan(se) else None,
            "z": float(z) if not np.isnan(z) else None,
            "p": float(p) if not np.isnan(p) else None,
            "95CI_log_odds": [float(ci_low) if not np.isnan(ci_low) else None,
                              float(ci_high) if not np.isnan(ci_high) else None],
            "odds_ratio_per_year": float(or_) if not np.isnan(or_) else None,
            "odds_ratio_95CI": or_ci
        }

    # Also provide stats for explicit interaction terms (their individual coefficients)
    interaction_term_stats = {}
    for lvl, name in interaction_terms.items():
        coef = float(params[name])
        se_ = float(np.sqrt(cov_df.loc[name, name]))
        z_ = coef / se_ if se_ > 0 else np.nan
        p_ = 2 * (1 - norm.cdf(abs(z_)))
        ci_low_, ci_high_ = (coef - 1.96 * se_, coef + 1.96 * se_)
        interaction_term_stats[name] = {
            "coef": coef,
            "se": se_,
            "z": z_,
            "p": p_,
            "95CI_log_odds": [ci_low_, ci_high_],
            "interpretation": ("This coefficient is the difference in the Age slope (log-odds per year) "
                               f"between culture '{lvl}' and the reference culture.")
        }

    # Package final object
    final_object = {
        "reference_culture": ref_culture,
        "age_main_effect": age_main_effect,
        "per_culture_slope": results_per_culture,
        "interaction_term_stats": interaction_term_stats,
        "notes": ("Slopes are expressed in log-odds per year. Odds ratios are multiplicative change in odds "
                  "of choosing the majority per additional year of age. p-values are two-sided using normal approximation.")
    }

    description = (
        "Extracted per-culture developmental slopes (change per year) for reliance on the majority, "
        "based on the fitted binomial GLM. The 'age_main_effect' is the slope for the model's reference culture. "
        "For each other culture, the slope equals (Age main effect) + (Age:C(Culture)[T.<culture>] interaction). "
        "Reported values: log-odds slope, standard error, z, two-sided p-value, and odds ratio with 95% CI. "
        "Significant positive slopes indicate increased reliance on the majority with age (higher probability of choosing majority), "
        "significant negative slopes indicate decreased reliance with age. Interaction term statistics show whether a culture's "
        "slope differs significantly from the reference culture's slope."
    )

    return {"object": final_object, "description": description}