def extract_final_answer(model_output):
    """
    Extracts age-related (per-year) effects on majority preference across cultures
    from the fitted models in `model_output`.

    Returns a dict with:
      - "object": structured results (per-culture slopes in log-odds, odds-ratios,
                  SEs, z, p, 95% CIs) for:
            * logit_majority (choosing majority vs not)
            * logit_demonstrated (choosing any demonstrated vs undemonstrated)
          plus joint (Wald) tests for the set of age-by-culture interaction terms.
      - "description": short interpretation of the contents and how to read them.
    """
    import numpy as np
    import pandas as pd
    import math
    from scipy import stats

    out = {}

    def summarize_interactions(binary_result):
        """
        Compute per-culture age slopes (log-odds per year), OR per year, SE, z, p,
        and 95% CIs for the slope, taking into account interaction terms.
        Also perform a joint Wald test for all age-by-culture interactions.
        Returns a dict with summary per culture and joint test result.
        """
        res = binary_result  # statsmodels BinaryResultsWrapper
        params = res.params
        cov = res.cov_params()
        # Dataframe used for extracting culture levels
        try:
            df = res.model.data.frame
        except Exception:
            # fallback: try .endog_names / .exog_names - but prefer raising informative error
            raise RuntimeError("Cannot access model data frame to get culture levels.")

        # Get unique culture levels as strings in the order encountered
        cultures = list(pd.unique(df['culture']))
        cultures_str = [str(c) for c in cultures]

        # Identify interaction parameter names: those that mention age_c and C(culture)
        interaction_names = [n for n in params.index if ('age_c' in n and 'C(culture)' in n)]
        # Identify which culture values appear in those interaction names (extract T.<level>)
        interaction_levels = []
        for n in interaction_names:
            # find substring T.<level>
            import re
            m = re.search(r'T\.([^\]\):]+)', n)
            if m:
                interaction_levels.append(m.group(1))
        # Determine reference level(s): those cultures not appearing in interaction_levels
        ref_levels = [s for s in cultures_str if s not in interaction_levels]
        reference = ref_levels[0] if len(ref_levels) > 0 else None

        summaries = {}
        # Base age coefficient name:
        if 'age_c' not in params.index:
            raise RuntimeError("Model does not contain 'age_c' main effect parameter.")
        beta_age = float(params['age_c'])
        var_age = float(cov.loc['age_c', 'age_c'])

        for cult, cult_str in zip(cultures, cultures_str):
            # Determine if culture has an interaction term
            # Find the matching interaction param name if any
            matched_inter = [n for n in interaction_names if f"T.{cult_str}" in n]
            if len(matched_inter) == 1:
                inter_name = matched_inter[0]
                beta_inter = float(params[inter_name])
                var_inter = float(cov.loc[inter_name, inter_name])
                cov_age_inter = float(cov.loc['age_c', inter_name])
                # combined slope:
                slope = beta_age + beta_inter
                var_slope = var_age + var_inter + 2.0 * cov_age_inter
            elif len(matched_inter) == 0:
                # reference culture (or culture without interaction param)
                slope = beta_age
                var_slope = var_age
            else:
                # unlikely to have >1 match, but if so combine them (rare)
                # sum all interactions matching (defensive)
                beta_inter = sum([float(params[n]) for n in matched_inter])
                var_inter = sum([float(cov.loc[n, n]) for n in matched_inter])
                cov_age_inter = sum([float(cov.loc['age_c', n]) for n in matched_inter])
                slope = beta_age + beta_inter
                var_slope = var_age + var_inter + 2.0 * cov_age_inter

            se_slope = math.sqrt(max(var_slope, 0.0))
            # Wald z and p
            if se_slope > 0:
                z = slope / se_slope
                p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            else:
                z = None
                p = None

            # 95% CI on log-odds and on OR scale
            ci_low = slope - 1.96 * se_slope
            ci_high = slope + 1.96 * se_slope
            or_per_year = math.exp(slope)
            or_ci_low = math.exp(ci_low)
            or_ci_high = math.exp(ci_high)

            summaries[cult_str] = {
                'slope_logodds_per_year': float(slope),
                'se_slope': float(se_slope),
                'z': float(z) if z is not None else None,
                'p_value': float(p) if p is not None else None,
                'ci_logodds_95': [float(ci_low), float(ci_high)],
                'OR_per_year': float(or_per_year),
                'OR_95_CI': [float(or_ci_low), float(or_ci_high)],
                'is_reference_level': (cult_str == reference)
            }

        # Joint Wald test of all interaction terms = 0 (i.e., no age-by-culture variation)
        if len(interaction_names) > 0:
            # Build restriction string e.g. "age_c:C(culture)[T.2] = 0, age_c:C(culture)[T.3] = 0"
            restriction = ', '.join([f"{name} = 0" for name in interaction_names])
            try:
                w = res.wald_test(restriction)
                # w.summary() exists; extract statistic and pvalue
                w_stat = float(w.statistic) if hasattr(w, 'statistic') else None
                w_p = float(w.pvalue) if hasattr(w, 'pvalue') else None
            except Exception:
                # fallback: set None
                w_stat = None
                w_p = None
        else:
            w_stat = None
            w_p = None

        return {
            'per_culture': summaries,
            'reference_level': reference,
            'interaction_param_names': interaction_names,
            'wald_test_all_interactions': {
                'statistic': w_stat,
                'p_value': w_p,
                'num_restrictions_tested': len(interaction_names)
            }
        }

    # Summarize majority preference model
    if 'logit_majority' not in model_output:
        raise KeyError("model_output must contain 'logit_majority' key.")
    maj_summary = summarize_interactions(model_output['logit_majority'])
    out['logit_majority_summary'] = maj_summary

    # Summarize demonstrated vs undemonstrated model
    if 'logit_demonstrated' not in model_output:
        raise KeyError("model_output must contain 'logit_demonstrated' key.")
    demo_summary = summarize_interactions(model_output['logit_demonstrated'])
    out['logit_demonstrated_summary'] = demo_summary

    # Short textual description
    desc_lines = [
        "This output provides age-related effects (change per year) on the log-odds of",
        "choosing the majority (logit_majority) and of choosing any demonstrated option",
        "(logit_demonstrated), evaluated separately for each cultural site present in the",
        "data. For each culture you get:",
        "  - slope_logodds_per_year: change in log-odds per additional year of age.",
        "  - OR_per_year: multiplicative change in odds per year (= exp(slope)).",
        "  - se, z, p_value: standard error, Wald z-statistic, and two-sided p-value for slope != 0.",
        "  - 95% CIs on log-odds and OR.",
        "",
        "The 'reference_level' is the culture treated as the baseline in the model (its",
        "age slope equals the model's main 'age_c' coefficient). For non-reference",
        "cultures the slope = main age coefficient + age-by-culture interaction coefficient.",
        "",
        "Also included is a joint Wald test result testing whether all age-by-culture",
        "interaction coefficients are simultaneously zero (i.e., no cultural variation in",
        "developmental slopes). A small p-value there indicates significant cross-cultural",
        "differences in how majority preference changes with age."
    ]
    description = "\n".join(desc_lines)

    return {"object": out, "description": description}