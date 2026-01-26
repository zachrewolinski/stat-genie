def extract_final_answer(model_output):
    """
    Extracts statistics comparing Homo sapiens to non-human genera (Pan, Pongo, Papio)
    from a binomial (logit) GLM stored in model_output.

    Returns a dict with keys:
      - "object": dict of comparisons (Homo vs Pan/Pongo/Papio) with:
          estimate_log_odds, se_log_odds, z, p_value, ci_log_odds, odds_ratio, ci_odds_ratio,
          param_names_used (the parameter names in the model)
      - "description": human-readable interpretation of each comparison (direction and
                       statistical significance at alpha=0.05), and notes about model
                       used (clustered vs conventional SE).
    """
    import re
    import math
    import numpy as np
    try:
        from scipy.stats import norm
    except Exception:
        # If scipy is unavailable, approximate using math.erfc for tail-probabilities
        def _norm_sf(x):
            # survival function using approximation
            return 0.5 * math.erfc(x / math.sqrt(2))
        class _NormApprox:
            @staticmethod
            def sf(x): return _norm_sf(x)
            @staticmethod
            def cdf(x): return 1.0 - _norm_sf(x)
        norm = _NormApprox()

    # Choose clustered results if available, otherwise fall back to conventional GLM result
    result = None
    covmat = None
    used_cov_type = None
    if model_output is None or not isinstance(model_output, dict):
        raise ValueError("model_output must be the dict returned by the modeling function.")
    if model_output.get('glm_result_clustered') is not None:
        result = model_output['glm_result_clustered']
        try:
            covmat = result.cov_params()
            used_cov_type = 'clustered'
        except Exception:
            # fallback
            result = model_output.get('glm_result')
            covmat = result.cov_params()
            used_cov_type = 'conventional (cluster requested but failed)'
    else:
        result = model_output.get('glm_result')
        if result is None:
            raise ValueError("No glm_result found in model_output.")
        covmat = result.cov_params()
        used_cov_type = 'conventional'

    params = result.params  # pandas Series
    param_index = list(params.index)

    # Build mapping from genus level -> parameter name (for parameters of form C(genus)[T.<LEVEL>])
    genus_param_map = {}
    genus_param_pattern = re.compile(r"C\(genus\)\[T\.(.+)\]")  # captures the level text
    for pname in param_index:
        m = genus_param_pattern.match(pname)
        if m:
            level = m.group(1)
            genus_param_map[level] = pname

    # Determine whether Homo level is present (may be 'Homo sapiens' or similar)
    homo_level = None
    for level in genus_param_map:
        if re.search(r'homo|sapiens', level, flags=re.I):
            homo_level = level
            break

    # If Homo level is not present among parameters, homo may be the reference level
    homo_is_reference = False
    if homo_level is None:
        # If no explicit Homo param, assume Homo is reference (i.e., its effect is absorbed in intercept)
        homo_is_reference = True

    # Prepare list of non-human genera to compare
    compare_genera = ['Pan', 'Pongo', 'Papio']

    comparisons = {}
    for gen in compare_genera:
        # find best matching param name for this genus (case-insensitive match)
        gen_param = None
        for level, pname in genus_param_map.items():
            if level.strip().lower() == gen.strip().lower():
                gen_param = pname
                matched_level = level
                break
        if gen_param is None:
            # try partial match (e.g., underscores, spacing differences)
            for level, pname in genus_param_map.items():
                if gen.strip().lower() in level.strip().lower():
                    gen_param = pname
                    matched_level = level
                    break

        # Determine beta_homo and beta_gen (on log-odds scale)
        if homo_is_reference:
            beta_homo = 0.0
            homo_param_name = None
            var_homo = 0.0
        else:
            homo_param_name = genus_param_map[homo_level]
            beta_homo = float(params[homo_param_name])
            var_homo = float(covmat.loc[homo_param_name, homo_param_name])

        if gen_param is None:
            # genus is likely the reference level (no parameter)
            beta_gen = 0.0
            gen_param_name = None
            var_gen = 0.0
        else:
            gen_param_name = gen_param
            beta_gen = float(params[gen_param_name])
            var_gen = float(covmat.loc[gen_param_name, gen_param_name])

        # Estimate difference Homo - Genus on log-odds scale
        est = beta_homo - beta_gen

        # Compute variance of the contrast:
        if (homo_param_name is not None) and (gen_param_name is not None):
            cov_hg = float(covmat.loc[homo_param_name, gen_param_name])
            var_diff = var_homo + var_gen - 2.0 * cov_hg
        elif (homo_param_name is None) and (gen_param_name is not None):
            # Homo reference, var(homo)=0, cov=0
            var_diff = var_gen
        elif (homo_param_name is not None) and (gen_param_name is None):
            # Genus reference, var(genus)=0, cov=0
            var_diff = var_homo
        else:
            # both are reference? unlikely (would mean only one genus in data)
            var_diff = 0.0

        # numerical safety
        se = math.sqrt(max(var_diff, 0.0))

        # z and two-sided p-value
        z = est / se if se > 0 else float('nan')
        if se > 0:
            p_value = 2.0 * norm.sf(abs(z))
        else:
            p_value = float('nan')

        # 95% CI on log-odds
        z_crit = 1.96
        ci_low = est - z_crit * se
        ci_high = est + z_crit * se

        # convert to odds ratio scale
        or_est = math.exp(est)
        or_ci_low = math.exp(ci_low)
        or_ci_high = math.exp(ci_high)

        comparisons[f"Homo_vs_{gen}"] = {
            "log_odds_difference": est,
            "se_log_odds_difference": se,
            "z": z,
            "p_value": p_value,
            "ci_log_odds": (ci_low, ci_high),
            "odds_ratio": or_est,
            "ci_odds_ratio": (or_ci_low, or_ci_high),
            "param_homo": homo_param_name,
            "param_genus": gen_param_name,
            "genus_level_matched": matched_level if 'matched_level' in locals() else None
        }
        # clear matched_level for next loop
        if 'matched_level' in locals():
            del matched_level

    # Build description: interpret direction and significance
    desc_lines = []
    desc_lines.append(f"Using GLM results with {used_cov_type} standard errors.")
    for key, stats in comparisons.items():
        gen = key.replace("Homo_vs_", "")
        est = stats["log_odds_difference"]
        p = stats["p_value"]
        or_est = stats["odds_ratio"]
        ci_or = stats["ci_odds_ratio"]

        if math.isnan(est):
            interpretation = f"Comparison Homo vs {gen}: could not compute (missing parameters)."
        else:
            direction = "higher" if est > 0 else ("lower" if est < 0 else "no difference")
            signif = ""
            if not math.isnan(p):
                signif = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
            interpretation = (f"Homo sapiens has {direction} AMTL odds compared to {gen} "
                              f"(OR = {or_est:.3f}, 95% CI = [{ci_or[0]:.3f}, {ci_or[1]:.3f}], p = {p:.3g}). "
                              f"This is {signif}.")
        desc_lines.append(interpretation)

    description = " ".join(desc_lines)

    return {
        "object": {
            "used_cov_type": used_cov_type,
            "comparisons": comparisons,
            "raw_params": params.to_dict()
        },
        "description": description
    }