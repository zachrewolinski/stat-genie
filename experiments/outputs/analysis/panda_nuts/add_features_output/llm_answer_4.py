def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and key marginal effects
    (at mean-centered age_z = 0) for the predictors relevant to the question:
      - age_z
      - Sex_M
      - HelpReceived
      - Sex_M:HelpReceived
      - age_z:HelpReceived

    Returns a dictionary with keys:
      - "object": dict of extracted numeric results (parameter table + marginal effects)
      - "description": brief interpretation of those results in context
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy.stats import norm
    except Exception:
        # If scipy not available, define normal cdf using math.erf
        import math
        def _norm_cdf(x):
            return 0.5 * (1 + math.erf(x / sqrt(2)))
        class _Norm:
            @staticmethod
            def sf(x): return 1 - _norm_cdf(x)
            @staticmethod
            def cdf(x): return _norm_cdf(x)
        norm = _Norm()

    # Helper to get parameter names robustly (try alternatives)
    def find_param_name(params_index, target):
        # try exact
        if target in params_index:
            return target
        # try with spaces around ':' or reversed order
        alt1 = target.replace(":", ": ")
        if alt1 in params_index:
            return alt1
        alt2 = target.replace(":", " : ")
        if alt2 in params_index:
            return alt2
        # reversed order
        if ":" in target:
            a, b = target.split(":")
            rev = f"{b}:{a}"
            if rev in params_index:
                return rev
            rev2 = rev.replace(":", ": ")
            if rev2 in params_index:
                return rev2
        # as last resort, try to find any index containing both variable names
        parts = target.split(":")
        if len(parts) == 2:
            a, b = parts
            for name in params_index:
                if a in name and b in name:
                    return name
        return None

    # Extract params, cov matrix, conf_int if available
    try:
        params = model_output.params.copy()
    except Exception as e:
        raise ValueError("Could not extract .params from model_output: " + str(e))

    try:
        cov = model_output.cov_params()
    except Exception:
        # try method name variation
        try:
            cov = model_output.cov_params_default()
        except Exception as e:
            raise ValueError("Could not extract covariance matrix from model_output: " + str(e))

    # Try to get bse and pvalues and conf_int; compute pvalues if needed
    bse = None
    pvalues = None
    conf_int = None
    if hasattr(model_output, "bse"):
        try:
            bse = model_output.bse.copy()
        except Exception:
            bse = None
    if hasattr(model_output, "pvalues"):
        try:
            pvalues = model_output.pvalues.copy()
        except Exception:
            pvalues = None
    if hasattr(model_output, "conf_int"):
        try:
            conf_int = model_output.conf_int()
        except Exception:
            conf_int = None

    # If bse not available, compute from cov diagonal
    if bse is None:
        try:
            bse = np.sqrt(np.diag(cov))
            # make into a pandas Series if params is Series
            try:
                import pandas as _pd
                bse = _pd.Series(bse, index=params.index)
            except Exception:
                pass
        except Exception as e:
            raise ValueError("Could not obtain standard errors: " + str(e))

    # If pvalues not available, compute two-sided normal-approx p-values
    if pvalues is None:
        with_norm = True
        zscores = params / bse
        try:
            pvals = 2 * (1 - norm.cdf(np.abs(zscores)))
        except Exception:
            # fallback if norm.cdf not vectorized
            pvals = []
            for z in np.abs(zscores):
                pvals.append(2 * (1 - norm.cdf(z)))
            pvals = np.array(pvals)
        try:
            import pandas as _pd
            pvalues = _pd.Series(pvals, index=params.index)
        except Exception:
            pvalues = pvals

    # If conf_int not available, compute 95% CI using normal approx
    if conf_int is None:
        try:
            z = norm.ppf(0.975)
        except Exception:
            # if norm has no ppf (fallback), approximate 1.96
            z = 1.96
        lower = params - z * bse
        upper = params + z * bse
        try:
            import pandas as _pd
            conf_int = _pd.DataFrame({"2.5%": lower, "97.5%": upper})
        except Exception:
            conf_int = np.vstack([lower, upper]).T

    # Variables of interest
    interest = ['age_z', 'Sex_M', 'HelpReceived', 'Sex_M:HelpReceived', 'age_z:HelpReceived']
    resolved_names = {}
    for var in interest:
        name = find_param_name(params.index, var)
        resolved_names[var] = name  # may be None if not present

    # Build parameter table for available variables
    param_table = {}
    for var, name in resolved_names.items():
        if name is None:
            param_table[var] = None
        else:
            coef = float(params[name])
            se = float(bse[name]) if hasattr(bse, '__getitem__') else float(bse[list(params.index).index(name)])
            p = float(pvalues[name]) if hasattr(pvalues, '__getitem__') else float(pvalues[list(params.index).index(name)])
            try:
                ci_low = float(conf_int.loc[name].iloc[0]) if hasattr(conf_int, "loc") else float(conf_int[list(params.index).index(name)][0])
                ci_high = float(conf_int.loc[name].iloc[1]) if hasattr(conf_int, "loc") else float(conf_int[list(params.index).index(name)][1])
            except Exception:
                # conf_int may be shaped differently
                try:
                    ci_low = float(conf_int[name][0])
                    ci_high = float(conf_int[name][1])
                except Exception:
                    ci_low, ci_high = None, None
            param_table[var] = {"param_name": name, "coef": coef, "se": se, "p": p, "95%_CI": [ci_low, ci_high]}

    # Function to compute SE and p for linear combination L'*beta
    def lincomb_stats(L_dict):
        # L_dict: mapping param_name -> multiplier
        # build vector L aligned with params.index
        import pandas as _pd
        idx = list(params.index)
        L = np.zeros(len(idx), dtype=float)
        for pname, mult in L_dict.items():
            if pname not in idx:
                raise KeyError(f"Parameter '{pname}' not found in model params when computing linear combination.")
            L[idx.index(pname)] = float(mult)
        estimate = float(np.dot(L, params.values))
        var_est = float(np.dot(L, np.dot(cov.values, L)))
        se_est = float(np.sqrt(var_est)) if var_est >= 0 else float(np.nan)
        try:
            z = estimate / se_est if se_est != 0 else np.nan
            pval = float(2 * (1 - norm.cdf(abs(z))))
        except Exception:
            pval = None
        # 95% CI
        try:
            zcrit = norm.ppf(0.975)
        except Exception:
            zcrit = 1.96
        ci_low = estimate - zcrit * se_est
        ci_high = estimate + zcrit * se_est
        return {"estimate": estimate, "se": se_est, "p": pval, "95%_CI": [ci_low, ci_high]}

    # Compute marginal effects at mean age (age_z = 0). Because age_z is centered, 0 = mean age.
    marginal_effects = {}
    # Effect of HelpReceived for females (Sex_M=0) at mean age => just coef of HelpReceived
    h_name = resolved_names['HelpReceived']
    s_h_name = resolved_names['Sex_M:HelpReceived']
    age_name = resolved_names['age_z']
    age_h_name = resolved_names['age_z:HelpReceived']
    sex_name = resolved_names['Sex_M']

    if h_name is not None:
        try:
            # female (Sex_M=0): effect = beta_H
            L = {h_name: 1.0}
            marginal_effects['HelpEffect_female_at_mean_age'] = lincomb_stats(L)
        except Exception as e:
            marginal_effects['HelpEffect_female_at_mean_age'] = {"error": str(e)}
    else:
        marginal_effects['HelpEffect_female_at_mean_age'] = None

    if h_name is not None and s_h_name is not None:
        try:
            # male (Sex_M=1): effect = beta_H + beta_SxH
            L = {h_name: 1.0, s_h_name: 1.0}
            marginal_effects['HelpEffect_male_at_mean_age'] = lincomb_stats(L)
        except Exception as e:
            marginal_effects['HelpEffect_male_at_mean_age'] = {"error": str(e)}
    else:
        marginal_effects['HelpEffect_male_at_mean_age'] = None

    # Effect of age (slope) when no help and when help
    if age_name is not None:
        try:
            # when HelpReceived = 0: slope = beta_age_z
            L = {age_name: 1.0}
            marginal_effects['AgeSlope_nohelp'] = lincomb_stats(L)
        except Exception as e:
            marginal_effects['AgeSlope_nohelp'] = {"error": str(e)}
    else:
        marginal_effects['AgeSlope_nohelp'] = None

    if age_name is not None and age_h_name is not None:
        try:
            # when HelpReceived = 1: slope = beta_age_z + beta_agez:HelpReceived
            L = {age_name: 1.0, age_h_name: 1.0}
            marginal_effects['AgeSlope_withhelp'] = lincomb_stats(L)
        except Exception as e:
            marginal_effects['AgeSlope_withhelp'] = {"error": str(e)}
    else:
        marginal_effects['AgeSlope_withhelp'] = None

    # Sex differences at mean age, without and with help
    if sex_name is not None:
        try:
            marginal_effects['SexEffect_nohelp_at_mean_age'] = lincomb_stats({sex_name: 1.0})
        except Exception as e:
            marginal_effects['SexEffect_nohelp_at_mean_age'] = {"error": str(e)}
    else:
        marginal_effects['SexEffect_nohelp_at_mean_age'] = None

    if sex_name is not None and s_h_name is not None:
        try:
            marginal_effects['SexEffect_withhelp_at_mean_age'] = lincomb_stats({sex_name: 1.0, s_h_name: 1.0})
        except Exception as e:
            marginal_effects['SexEffect_withhelp_at_mean_age'] = {"error": str(e)}
    else:
        marginal_effects['SexEffect_withhelp_at_mean_age'] = None

    # Package results
    results_object = {
        "parameter_table": param_table,
        "marginal_effects_at_mean_age": marginal_effects,
        "model_params_index": list(params.index)
    }

    # Build concise description
    # We'll summarize which effects appear (based on p < 0.05) if p-values available
    def sig_label(p):
        try:
            if p is None:
                return "p: NA"
            p = float(p)
            if p < 0.001:
                return "p < 0.001"
            else:
                return f"p = {p:.3f}"
        except Exception:
            return "p: NA"

    desc_lines = []
    desc_lines.append("Extracted parameter estimates and marginal effects (at mean-centered age_z = 0).")
    # summarize each main predictor
    for var in ['age_z', 'Sex_M', 'HelpReceived', 'Sex_M:HelpReceived', 'age_z:HelpReceived']:
        entry = param_table.get(var)
        if entry is None:
            desc_lines.append(f"- {var}: not estimated / not found in model output.")
        else:
            desc_lines.append(f"- {var}: coef = {entry['coef']:.3f}, se = {entry['se']:.3f}, {sig_label(entry['p'])}, 95%CI = [{entry['95%_CI'][0]:.3f}, {entry['95%_CI'][1]:.3f}]")

    # Summarize key marginal contrasts concisely
    desc_lines.append("Key marginal effects at mean age (age_z = 0):")
    me = marginal_effects
    def append_me(key, label):
        val = me.get(key)
        if val is None:
            desc_lines.append(f"- {label}: NA")
        elif "error" in (val or {}):
            desc_lines.append(f"- {label}: error computing ({val.get('error')})")
        else:
            desc_lines.append(f"- {label}: est = {val['estimate']:.3f}, se = {val['se']:.3f}, {sig_label(val['p'])}, 95%CI = [{val['95%_CI'][0]:.3f}, {val['95%_CI'][1]:.3f}]")
    append_me('HelpEffect_female_at_mean_age', "Effect of receiving help (females)")
    append_me('HelpEffect_male_at_mean_age', "Effect of receiving help (males)")
    append_me('AgeSlope_nohelp', "Age slope when no help")
    append_me('AgeSlope_withhelp', "Age slope when help received")
    append_me('SexEffect_nohelp_at_mean_age', "Sex effect (male vs female) when no help")
    append_me('SexEffect_withhelp_at_mean_age', "Sex effect (male vs female) when help received")

    description = "\n".join(desc_lines)

    return {"object": results_object, "description": description}