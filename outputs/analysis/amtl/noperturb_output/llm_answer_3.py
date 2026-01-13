def extract_final_answer(model_output):
    """
    Extracts genus comparison statistics from the model output and returns a concise,
    interpretable summary about whether Homo sapiens has higher AMTL than the non-human genera.

    Returns:
      {
        "object": {
          "<Genus>": {
              "coef": float,             # log-odds (genus - Homo sapiens)
              "se": float,               # clustered (or model) standard error
              "z": float,                # z-statistic
              "p": float,                # two-sided p-value
              "ci_95": [lower, upper],   # 95% CI on log-odds
              "odds_ratio": float,       # exp(coef)
              "odds_ratio_ci_95": [lower_or, upper_or], # CI for OR
              "interpretation": str      # brief interpretation vs Homo sapiens
          },
          ...
        },
        "description": str   # short textual summary of comparisons and overall conclusion
      }
    """
    import re
    import numpy as np
    import math

    # Helpers for normal p-value if scipy is not available
    def two_sided_p_from_z(z):
        try:
            # prefer scipy if available
            from scipy.stats import norm
            return 2 * (1 - norm.cdf(abs(z)))
        except Exception:
            # fallback using math.erf: Phi(z) = 0.5*(1 + erf(z/sqrt(2)))
            Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            return 2 * (1.0 - Phi) if z >= 0 else 2 * Phi

    # Get the results object (prefer clustered_result)
    clustered = model_output.get('clustered_result', None)
    glmres = model_output.get('glm_result', None)

    if clustered is None and glmres is None:
        raise ValueError("model_output must contain at least one of 'clustered_result' or 'glm_result'.")

    # Function to extract named arrays from a result-like object
    def _get_arrays(res):
        """
        Return params (pd.Series or dict-like), bse (pd.Series), pvalues (pd.Series or None),
        and conf_int_func (callable alpha -> DataFrame with lower/upper) where possible.
        """
        params = None
        bse = None
        pvalues = None
        conf_int_func = None

        # clustered_result wrapper path
        if res is not None and hasattr(res, 'params') and hasattr(res, 'bse'):
            params = res.params
            bse = res.bse
            pvalues = getattr(res, 'pvalues', None)
            # conf_int may be a method
            if hasattr(res, 'conf_int') and callable(getattr(res, 'conf_int')):
                conf_int_func = res.conf_int
            else:
                conf_int_func = None
            return params, bse, pvalues, conf_int_func

        # fallback to glmres (statsmodels results)
        if glmres is not None:
            res = glmres
            try:
                params = res.params
            except Exception:
                params = None
            try:
                bse = res.bse
            except Exception:
                bse = None
            try:
                pvalues = res.pvalues
            except Exception:
                pvalues = None
            try:
                conf_int_func = res.conf_int  # method
            except Exception:
                conf_int_func = None
            return params, bse, pvalues, conf_int_func

        return None, None, None, None

    params, bse, pvalues, conf_int_func = _get_arrays(clustered if clustered is not None else glmres)

    if params is None:
        raise RuntimeError("Could not extract parameter estimates from model output.")

    # Ensure we have indexable mappings (convert numpy arrays to pandas-like mapping with names if necessary)
    # params and bse might be pandas Series; if numpy arrays, attempt to get names from glmres.params.index
    try:
        param_index = list(params.index)
    except Exception:
        # params might be a numpy array; try to get names from glmres if present
        if glmres is not None and hasattr(glmres, 'params') and hasattr(glmres.params, 'index'):
            param_index = list(glmres.params.index)
            # map params, bse to Series for consistent handling
            import pandas as pd
            params = pd.Series(np.asarray(params), index=param_index)
            if bse is not None:
                bse = pd.Series(np.asarray(bse), index=param_index)
        else:
            raise RuntimeError("Parameter names not available in model output.")

    # Regex to find genus coefficient names produced by Patsy/statsmodels with Treatment coding:
    # e.g. C(genus, Treatment(reference="Homo sapiens"))[T.Pan]
    genus_pat = re.compile(r'C\(genus.*\)\[T\.([^]]+)\]')

    results = {}
    significant_higher_in_humans = []  # list of genera where Homo sapiens > genus (significant)
    significant_lower_in_humans = []   # list where Homo sapiens < genus (significant)
    nonsig = []

    for name in param_index:
        m = genus_pat.search(name)
        if not m:
            continue
        genus = m.group(1)
        est = float(params.loc[name]) if hasattr(params, 'loc') else float(params[name])
        # get standard error
        if bse is not None:
            try:
                se = float(bse.loc[name])
            except Exception:
                se = float(bse[name])
        else:
            # if no SE available, try to compute from covariance if present
            se = None
            cov = None
            if clustered is not None and hasattr(clustered, 'cov_params'):
                cov = clustered.cov_params
            elif glmres is not None:
                try:
                    cov = glmres.cov_params()
                except Exception:
                    cov = None
            if cov is not None:
                try:
                    # cov may be DataFrame
                    if hasattr(cov, 'loc'):
                        se = float(np.sqrt(cov.loc[name, name]))
                    else:
                        # assume numpy array and matching index order
                        idx = param_index.index(name)
                        se = float(np.sqrt(cov[idx, idx]))
                except Exception:
                    se = None

        # compute z and p
        if se is None or se == 0 or np.isnan(se):
            z = None
            p = None
        else:
            z = est / se
            p = two_sided_p_from_z(z)

        # confidence interval on log-odds
        if conf_int_func is not None:
            try:
                ci_df = conf_int_func()
                # conf_int might return DataFrame with rows ordered by params index or named
                if hasattr(ci_df, 'loc') and name in ci_df.index:
                    lower = float(ci_df.loc[name, 0])
                    upper = float(ci_df.loc[name, 1])
                else:
                    # try to index by position
                    if hasattr(ci_df, 'iloc'):
                        idx = param_index.index(name)
                        lower = float(ci_df.iloc[idx, 0])
                        upper = float(ci_df.iloc[idx, 1])
                    else:
                        lower = est - 1.96 * se if se is not None else None
                        upper = est + 1.96 * se if se is not None else None
            except Exception:
                lower = est - 1.96 * se if se is not None else None
                upper = est + 1.96 * se if se is not None else None
        else:
            lower = est - 1.96 * se if se is not None else None
            upper = est + 1.96 * se if se is not None else None

        # odds ratio and CI
        try:
            or_point = float(np.exp(est))
            or_lower = float(np.exp(lower)) if lower is not None else None
            or_upper = float(np.exp(upper)) if upper is not None else None
        except Exception:
            or_point = None
            or_lower = None
            or_upper = None

        # Interpretation: recall coefficients are (genus - Homo sapiens).
        # If est < 0 => genus has lower log-odds than Homo sapiens => Homo sapiens higher AMTL.
        if p is not None and p < 0.05:
            if est < 0:
                interp = f"Homo sapiens has significantly higher AMTL than {genus} (p={p:.3g})."
                significant_higher_in_humans.append(genus)
            else:
                interp = f"Homo sapiens has significantly lower AMTL than {genus} (p={p:.3g})."
                significant_lower_in_humans.append(genus)
        else:
            interp = f"No statistically significant difference in AMTL between Homo sapiens and {genus} (p={p:.3g} if available)."
            nonsig.append(genus)

        results[genus] = {
            "coef_log_odds (genus - Homo sapiens)": est,
            "se": se,
            "z": z,
            "p": p,
            "ci_95_log_odds": [lower, upper],
            "odds_ratio (genus / Homo sapiens)": or_point,
            "odds_ratio_95ci": [or_lower, or_upper],
            "interpretation": interp
        }

    # Build a short textual summary/description
    summary_parts = []
    if significant_higher_in_humans:
        summary_parts.append(
            "Homo sapiens has significantly higher AMTL than: " + ", ".join(significant_higher_in_humans) + "."
        )
    if significant_lower_in_humans:
        summary_parts.append(
            "Homo sapiens has significantly lower AMTL than: " + ", ".join(significant_lower_in_humans) + "."
        )
    if nonsig:
        summary_parts.append(
            "No significant difference with: " + ", ".join(nonsig) + "."
        )
    if not summary_parts:
        summary = "No genus-level comparisons found in model output."
    else:
        summary = " ".join(summary_parts)

    # Final overall statement answering the task question (qualify by significance)
    if significant_higher_in_humans and not significant_lower_in_humans and len(nonsig) == 0:
        overall = "Overall: Yes — modern humans (Homo sapiens) show higher AMTL than all compared non-human genera (all differences statistically significant)."
    else:
        # more nuanced result
        parts = []
        if significant_higher_in_humans:
            parts.append("Homo sapiens has higher AMTL than " + ", ".join(significant_higher_in_humans))
        if significant_lower_in_humans:
            parts.append("Homo sapiens has lower AMTL than " + ", ".join(significant_lower_in_humans))
        if nonsig:
            parts.append("no significant difference with " + ", ".join(nonsig))
        overall = "Overall: " + "; ".join(parts) + ". Differences referred to are statistical comparisons (alpha=0.05) vs the Homo sapiens reference."

    description = summary + " " + overall

    return {
        "object": results,
        "description": description
    }