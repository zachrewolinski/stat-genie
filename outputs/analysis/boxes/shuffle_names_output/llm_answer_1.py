def extract_final_answer(model_output):
    """
    Extracts age-related effects from a fitted statsmodels GLMResultsWrapper (with or without
    robust cov) for the formula: MajorityChosen ~ Age * C(Site) + IsGirl + MajorityFirstShown.

    Returns a dictionary with:
      - "object": a dict containing:
          - "age_main": coefficient, se, z, p, conf_int for the Age main effect
                       (this is the age slope in the reference site),
          - "slopes_by_site": mapping from each Site level -> estimated age slope (coef),
                              se, z, p, 95% CI (these are Age slopes within each site),
          - "interaction_test_p": p-value for the joint test that all Age:Site interaction
                                  coefficients are zero (i.e., age effect does NOT vary by site).
      - "description": brief interpretation of the results in the study context.
    """
    import numpy as np
    import re

    # Defensive checks
    if model_output is None:
        raise ValueError("model_output is None")

    # Extract parameter names and params
    try:
        params = model_output.params
        param_names = list(params.index)
    except Exception as e:
        raise ValueError(f"Unable to access params from model_output: {e}")

    # Try to get original data frame to find all Site levels
    try:
        df = model_output.model.data.frame
        site_levels = np.unique(df['Site'])
    except Exception:
        # Fallback: infer site levels from parameter names of C(Site)[T.x]
        site_levels = []
        for nm in param_names:
            m = re.search(r"C\(Site\)\[T\.?(.+?)\]", nm)
            if m:
                site_levels.append(m.group(1))
        # If none found, leave as empty list
        site_levels = np.array(site_levels, dtype=object)

    # Identify the Age main parameter name (should be 'Age')
    if 'Age' in param_names:
        age_param = 'Age'
    else:
        # Fallback: find a parameter equal to 'Age' or that endswith ':Age' or startswith 'Age'
        candidates = [n for n in param_names if n == 'Age' or n.startswith('Age') or n.endswith(':Age') or ':Age:' in n]
        if len(candidates) >= 1:
            age_param = candidates[0]
        else:
            raise ValueError("Could not find an 'Age' parameter name in the model parameters.")

    # Find interaction parameter names of form 'Age:C(Site)[T.x]'
    interaction_pattern = re.compile(r"Age:C\(Site\)\[T\.?(.+?)\]")
    interaction_params = {}
    for nm in param_names:
        m = interaction_pattern.match(nm)
        if m:
            lvl = m.group(1)
            interaction_params[str(lvl)] = nm

    # Helper to build contrast vector for a given linear combination of parameters
    def make_contrast_vector(param_weights):
        """
        param_weights: dict param_name -> weight
        returns: numpy array of length k (number of params) with weights in param order
        """
        k = len(param_names)
        L = np.zeros((1, k))
        name_to_idx = {n: i for i, n in enumerate(param_names)}
        for pname, w in param_weights.items():
            if pname not in name_to_idx:
                raise KeyError(f"Parameter name '{pname}' not found among model parameters.")
            L[0, name_to_idx[pname]] = w
        return L

    # Function to get contrast results (coef, se, z, p, conf_int)
    def contrast_results(L):
        # L should be (1, k)
        try:
            res = model_output.t_test(L)
        except Exception:
            # Try wald_test as fallback
            res = model_output.wald_test(L)
        # res may be a ContrastResults or WaldTestResults; handle common attributes
        effect = float(res.effect.flatten())
        # standard error
        try:
            se = float(res.sd.flatten())
        except Exception:
            # fallback compute from covariance: sqrt(L * cov * L')
            cov = model_output.cov_params()
            se = float(np.sqrt(L.dot(cov).dot(L.T))[0, 0])
        # z/t value and pvalue
        try:
            tstat = float(res.tvalue.flatten())
            pval = float(res.pvalue)
        except Exception:
            # compute z and p manually (assume normal)
            import math
            if se == 0:
                tstat = float('nan')
                pval = float('nan')
            else:
                tstat = effect / se
                from math import erf, sqrt
                # two-sided p-value using normal approx
                import scipy.stats as st
                pval = float(2 * (1 - st.norm.cdf(abs(tstat))))
        # 95% CI
        try:
            ci = res.conf_int()
            ci_low, ci_high = float(ci[0, 0]), float(ci[0, 1])
        except Exception:
            # compute using normal approx
            import scipy.stats as st
            zcrit = st.norm.ppf(0.975)
            ci_low = effect - zcrit * se
            ci_high = effect + zcrit * se
        return {"coef": effect, "se": se, "z": tstat, "p": pval, "95% CI": (ci_low, ci_high)}

    # Extract Age main effect (reference site slope)
    L_age = make_contrast_vector({age_param: 1.0})
    age_main_res = contrast_results(L_age)

    # For each site level, compute site-specific slope = Age + Age:C(Site)[T.<level>] (if interaction exists)
    slopes_by_site = {}
    # If we managed to get site_levels from df, they may be ints; convert to strings for matching interaction keys
    site_levels_list = list(site_levels)
    if site_levels_list == []:
        # If site levels unknown, try to infer from interaction params + reference
        # Infer reference by checking which C(Site)[T.x] levels appear: we'll collect levels from interaction params and C(Site) params
        levels_in_params = set()
        for nm in param_names:
            m = re.search(r"C\(Site\)\[T\.?(.+?)\]", nm)
            if m:
                levels_in_params.add(m.group(1))
        # If numeric range 1..8 expected, try that
        try:
            candidate_levels = [str(i) for i in range(1, 9)]
            site_levels_list = candidate_levels
        except Exception:
            site_levels_list = sorted(list(levels_in_params))
    # Attempt to get the reference site: the one for which there is no 'C(Site)[T.x]' param
    # Identify all levels present in data if possible
    ref_site = None
    try:
        if 'Site' in model_output.model.data.frame.columns:
            all_levels = list(np.unique(model_output.model.data.frame['Site']))
            # Determine which level lacks a corresponding C(Site)[T.x] or Age:C(Site)[T.x] param:
            present_levels = set()
            for nm in param_names:
                m = re.search(r"C\(Site\)\[T\.?(.+?)\]", nm)
                if m:
                    present_levels.add(m.group(1))
            # convert levels to strings for comparison
            for lev in all_levels:
                if str(lev) not in present_levels:
                    ref_site = lev
                    break
            if ref_site is None:
                # if none missing, assume the smallest level is reference
                ref_site = min(all_levels)
    except Exception:
        ref_site = None

    # If we still don't have site_levels_list in a good form, fall back to interaction param keys + likely reference
    # Convert everything to strings for consistent keys
    # Build a set of candidate site identifiers to report
    candidate_sites = []
    try:
        # Prefer numeric levels from data if available
        if 'Site' in model_output.model.data.frame.columns:
            candidate_sites = sorted(list(np.unique(model_output.model.data.frame['Site'])), key=lambda x: (float(x) if str(x).replace('.','',1).isdigit() else str(x)))
        else:
            # use interaction param extracted levels + a guessed reference
            candidate_sites = []
            for nm in param_names:
                m = re.search(r"C\(Site\)\[T\.?(.+?)\]", nm)
                if m:
                    candidate_sites.append(m.group(1))
            candidate_sites = sorted(list(set(candidate_sites)))
            # guess reference as '1' if not present
            if ref_site is None and '1' not in candidate_sites:
                candidate_sites = ['1'] + candidate_sites
    except Exception:
        candidate_sites = []

    # Compute slopes for each candidate site
    for site in candidate_sites:
        site_str = str(site)
        # If interaction param exists for this site, include it; else only Age main
        if site_str in interaction_params:
            inter_param_name = interaction_params[site_str]
            L = make_contrast_vector({age_param: 1.0, inter_param_name: 1.0})
        else:
            # No interaction param -> this is the reference site
            L = make_contrast_vector({age_param: 1.0})
        try:
            slopes_by_site[site_str] = contrast_results(L)
        except KeyError:
            # If a required param name not found, skip this site
            slopes_by_site[site_str] = {"error": f"Could not construct contrast for site {site_str}."}

    # Joint test: are all Age:C(Site) interaction coefficients simultaneously zero?
    # Build R matrix with one row per interaction param and 1 at that param column.
    inter_param_names = [interaction_params[k] for k in sorted(interaction_params.keys(), key=lambda x: (float(x) if str(x).replace('.','',1).isdigit() else x))]
    interaction_test_p = None
    try:
        if len(inter_param_names) == 0:
            interaction_test_p = 1.0  # no interactions in model (unlikely), so trivially no interaction
        else:
            k = len(param_names)
            R = np.zeros((len(inter_param_names), k))
            name_to_idx = {n: i for i, n in enumerate(param_names)}
            for i, pname in enumerate(inter_param_names):
                R[i, name_to_idx[pname]] = 1.0
            # Use wald_test for joint zero hypothesis R * beta = 0
            wt = model_output.wald_test(R)
            # wt has attribute pvalue
            try:
                interaction_test_p = float(wt.pvalue)
            except Exception:
                # fallback compute from statistic and df
                try:
                    f_val = float(wt.statistic)
                    df_denom = model_output.df_resid
                    from scipy.stats import f
                    # number of restrictions
                    q = R.shape[0]
                    pval = 1 - f.cdf(f_val, q, df_denom)
                    interaction_test_p = float(pval)
                except Exception:
                    interaction_test_p = None
    except Exception:
        interaction_test_p = None

    result_object = {
        "age_main": age_main_res,
        "slopes_by_site": slopes_by_site,
        "interaction_test_p": interaction_test_p,
        "param_names": param_names
    }

    # Compose human-readable description
    if interaction_test_p is None:
        p_text = "could not be computed"
    else:
        p_text = f"{interaction_test_p:.4g}"
    description = (
        "Extracted the age slope (effect of 1 year increase in Age) for the reference site "
        f"and site-specific age slopes (Age + Age:Site interaction) for each observed Site. "
        f"A joint Wald test for whether the Age:Site interactions are all zero has p = {p_text}. "
        "If that p-value is small (e.g., < .05), it indicates that the age-related change in reliance "
        "on the majority differs across cultural sites. The returned 'object' contains the numeric "
        "estimates (coef), standard errors, z-statistics, two-sided p-values, and 95% CIs."
    )

    return {"object": result_object, "description": description}