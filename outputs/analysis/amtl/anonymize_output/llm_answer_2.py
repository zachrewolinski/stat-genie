def extract_final_answer(model_output):
    """
    Extracts comparisons of Homo sapiens vs other genera from a fitted statsmodels GLM results object.
    Returns a dict with:
      - "object": dict of per-comparison numeric results (log-odds difference, OR, se, z, p, 95% CI)
      - "description": textual interpretation addressing whether Homo sapiens has higher AMTL than each genus

    If 'Homo sapiens' (or any 'Homo' level) is not present among the model's genus levels,
    the function will return an empty "object" and a descriptive message rather than raising an error.
    """
    import math
    import numpy as np
    import pandas as pd

    res = model_output

    # Get parameter names and parameter values
    params = pd.Series(res.params)
    param_names = list(params.index)

    # Try to recover the original dataframe and genus levels if available
    df = None
    try:
        df = res.model.data.frame
    except Exception:
        try:
            df = res.model.data.orig_endog  # fallback (likely not a df)
        except Exception:
            df = None

    # Determine genus levels from the dataframe if possible
    genus_levels = None
    if isinstance(df, pd.DataFrame) and 'Genus' in df.columns:
        # use categorical categories if present, else unique observed values
        if pd.api.types.is_categorical_dtype(df['Genus']):
            genus_levels = list(df['Genus'].cat.categories)
        else:
            # Use unique values sorted to be deterministic
            genus_levels = list(pd.Index(df['Genus']).unique())
    else:
        # Fallback: infer levels from parameter names by parsing strings
        inferred = set()
        for nm in param_names:
            if 'Genus' in nm:
                # Expect names like 'C(Genus)[T.Pan]' or 'Genus[T.Pan]'
                if 'T.' in nm:
                    part = nm.split('T.')[-1].rstrip(']')
                    inferred.add(part)
                else:
                    # try to extract after last dot or bracket
                    tokens = nm.replace(']', '').replace('[', '.').split('.')
                    if tokens:
                        inferred.add(tokens[-1])
        genus_levels = sorted(list(inferred)) if inferred else None

    if not genus_levels:
        description = ("Could not determine genus levels from model output or data. "
                       "Ensure the fitted model was created using 'C(Genus)' and the data frame contains a 'Genus' column.")
        return {"object": {}, "description": description}

    # Normalize Homo sapiens label variants to a single string present in genus_levels
    # Try to find matches for Homo sapiens (robust to case and small variants)
    def _contains_homo_sapiens(x):
        s = str(x).lower()
        return 'homo' in s and 'sapiens' in s

    def _contains_homo(x):
        s = str(x).lower()
        return 'homo' in s

    homo_labels = [g for g in genus_levels if _contains_homo_sapiens(g)]
    if len(homo_labels) == 0:
        # try any genus containing 'Homo'
        homo_labels = [g for g in genus_levels if _contains_homo(g)]
    # If still not found, do not raise; return informative message
    if len(homo_labels) == 0:
        description = ("'Homo sapiens' was not found among genus levels: {}. "
                       "No comparisons against Homo sapiens can be made.".format(genus_levels))
        return {"object": {}, "description": description}

    homo_label = homo_labels[0]

    # Build mapping from genus level -> parameter name (if exists). If missing, it's the reference (coef = 0).
    genus_to_param = {}
    for g in genus_levels:
        # expected parameter names (several possible patsy naming conventions)
        candidates = [
            f"C(Genus)[T.{g}]",
            f"Genus[T.{g}]",
            f"C(Genus)[{g}]",
            f"Genus[{g}]",
        ]
        found = None
        for c in candidates:
            if c in param_names:
                found = c
                break
        genus_to_param[g] = found  # None means reference (coef = 0)

    # Get covariance matrix for parameters (DataFrame or ndarray)
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    cov_df = None
    if isinstance(cov, pd.DataFrame):
        cov_df = cov
    else:
        # try to convert to DataFrame using param names
        try:
            cov_df = pd.DataFrame(cov, index=param_names, columns=param_names)
        except Exception:
            cov_df = None

    results = {}
    # Helper to get coef and variance for a parameter name (or zero if None)
    def get_coef_and_var(param_name):
        if param_name is None:
            return 0.0, 0.0
        coef = float(params[param_name])
        if cov_df is not None and param_name in cov_df.index:
            var = float(cov_df.loc[param_name, param_name])
        else:
            # fallback: try res.bse (may not be robust-covarianced)
            try:
                bse = res.bse[param_name]
                var = float(bse**2)
            except Exception:
                var = 0.0
        return coef, var

    # For each non-Homo genus, compute contrast: (Homo - genus)
    for g in genus_levels:
        if g == homo_label:
            continue
        p_homo = genus_to_param.get(homo_label)
        p_other = genus_to_param.get(g)

        beta_homo, var_homo = get_coef_and_var(p_homo)
        beta_other, var_other = get_coef_and_var(p_other)

        # covariance between the two params (0 if either is reference or cov not available)
        cov_homo_other = 0.0
        if cov_df is not None and p_homo is not None and p_other is not None:
            try:
                cov_homo_other = float(cov_df.loc[p_homo, p_other])
            except Exception:
                cov_homo_other = 0.0

        # contrast = Homo - other
        contrast = beta_homo - beta_other
        var_contrast = var_homo + var_other - 2.0 * cov_homo_other
        # numerical safety
        if var_contrast < 0 and var_contrast > -1e-12:
            var_contrast = 0.0
        se = math.sqrt(var_contrast) if var_contrast > 0 else 0.0

        # z and two-sided p-value using normal approximation
        if se > 0:
            z = contrast / se
            pvalue = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided: erfc(z/sqrt(2))
        else:
            z = float('inf') if contrast != 0 else 0.0
            pvalue = 0.0 if contrast != 0 else 1.0

        # Odds ratio and 95% CI on OR scale
        or_point = float(np.exp(contrast))
        ci_low = float(np.exp(contrast - 1.96 * se)) if se > 0 else or_point
        ci_high = float(np.exp(contrast + 1.96 * se)) if se > 0 else or_point

        results[str(g)] = {
            "contrast_logodds_Homo_minus_{}".format(g): float(contrast),
            "se_logodds": float(se),
            "z": float(z),
            "p_value": float(pvalue),
            "OR_Homo_vs_{}".format(g): float(or_point),
            "OR_95CI": (ci_low, ci_high),
            "Homo_param_name": p_homo,
            "Other_param_name": p_other
        }

    # Build human-readable description
    lines = []
    lines.append(f"Reference genus/levels detected: {genus_levels}")
    lines.append(f"Comparisons are computed as log-odds(Homo sapiens) - log-odds(other genus).")
    lines.append("Therefore OR > 1 means Homo sapiens has higher odds of AMTL than the other genus; OR < 1 means lower odds.")
    summary_flags = []
    for g, stats in results.items():
        orv = stats.get(f"OR_Homo_vs_{g}")
        p = stats.get("p_value")
        ci = stats.get("OR_95CI")
        # Guard defaults
        if orv is None:
            continue
        direction = "higher" if orv > 1 else ("lower" if orv < 1 else "equal")
        signif = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
        lines.append(f"- Homo sapiens vs {g}: OR = {orv:.3f}, 95% CI = ({ci[0]:.3f}, {ci[1]:.3f}), p = {p:.4f} -> Homo {direction} AMTL than {g}; {signif}.")
        summary_flags.append((orv > 1 and p < 0.05))

    # Final answer interpretation: require Homo to be significantly higher than each non-human genus
    if len(summary_flags) == 0:
        overall = "No non-human genera found to compare (or Homo sapiens had no valid contrasts)."
    elif all(summary_flags):
        overall = "Yes — Homo sapiens has higher AMTL than all compared non-human genera (all pairwise ORs > 1 and p < 0.05)."
    else:
        # Report which comparisons show higher/significant
        pos_sig = [g for g, stats in results.items() if (stats.get(f"OR_Homo_vs_{g}", 0) > 1 and stats.get("p_value", 1) < 0.05)]
        pos_nonsig = [g for g, stats in results.items() if (stats.get(f"OR_Homo_vs_{g}", 0) > 1 and stats.get("p_value", 1) >= 0.05)]
        lower_sig = [g for g, stats in results.items() if (stats.get(f"OR_Homo_vs_{g}", 0) < 1 and stats.get("p_value", 1) < 0.05)]
        if pos_sig:
            overall = ("Homo sapiens shows significantly higher AMTL vs: " + ", ".join(pos_sig) + ". "
                       "However, this is not true for all genera.")
        else:
            overall = ("Homo sapiens does not show consistent significantly higher AMTL across the non-human genera. "
                       "See per-genus results above for details.")
    lines.append("")
    lines.append("Overall conclusion: " + overall)

    description = "\n".join(lines)

    return {"object": results, "description": description}