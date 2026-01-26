def extract_final_answer(model_output):
    """
    Extract relevant statistics about age effects and age-by-culture interactions
    from the model output returned by the modeling function.

    Returns a dict with:
      - "object": nested dict containing coefficients, cluster-robust SEs, z-stats,
                  two-sided p-values, and 95% CIs for:
                    * the main 'age' coefficient (baseline slope)
                    * culture-specific age slopes (baseline + interaction for each culture)
                  for both the 'social_model' and 'majority_model' (if present).
      - "description": short explanation of what was extracted and how to interpret it.

    The code is robust to the ResultWrapper objects used in the model code: it will
    obtain params from result.params and the cluster-robust covariance via result.cov_params().
    It computes standard errors and inference for linear combinations (e.g., baseline age
    and baseline+interaction for culture-specific slopes) using the covariance matrix.
    """
    import numpy as np
    import pandas as pd
    import re
    from scipy.stats import norm

    def _get_params_and_cov(result):
        # params as pandas Series
        try:
            params = result.params
        except Exception:
            # fallback: try attribute 'orig.params' if wrapper
            params = getattr(result, 'orig').params

        # covariance: try result.cov_params() (could be DataFrame or ndarray)
        cov = None
        try:
            cov = result.cov_params()
        except Exception:
            # fallback to orig
            cov = getattr(result, 'orig').cov_params()

        if isinstance(cov, pd.DataFrame):
            cov_df = cov
            cov_mat = cov_df.values
            cov_index = list(cov_df.index)
        else:
            cov_mat = np.asarray(cov)
            cov_index = list(params.index)

        params = pd.Series(params)  # ensure Series
        return params, cov_mat, cov_index

    def _linear_combo_stats(weights, params, cov_mat, cov_index):
        """
        weights: dict param_name -> weight (float)
        params: pandas Series indexed by param names
        cov_mat: numpy ndarray covariance matrix
        cov_index: list of param names in order for cov_mat
        Returns: dict with coef, se, z, p_two_sided, ci_lower, ci_upper
        """
        # build weight vector aligned with cov_index
        w = np.zeros(len(cov_index))
        for i, pname in enumerate(cov_index):
            if pname in weights:
                w[i] = weights[pname]
        # coefficient
        # params may not include every cov_index key? ensure alignment
        pvec = np.array([params.get(n, 0.0) for n in cov_index], dtype=float)
        coef = float(np.dot(w, pvec))
        var = float(np.dot(w, cov_mat.dot(w)))
        se = np.sqrt(var) if var >= 0 else np.nan
        z = coef / se if se and not np.isnan(se) else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low = coef - 1.96 * se if not np.isnan(se) else np.nan
        ci_high = coef + 1.96 * se if not np.isnan(se) else np.nan
        return {
            'coef': coef,
            'se': se,
            'z': z,
            'p_two_sided': p,
            'ci_95_lower': ci_low,
            'ci_95_upper': ci_high
        }

    def _extract_model_info(result):
        """
        Extract:
          - main age coefficient stats
          - culture-specific age slopes (baseline + interaction) for each culture found in the model data
        Returns a dict.
        """
        if result is None:
            return None

        params, cov_mat, cov_index = _get_params_and_cov(result)

        # Attempt to get the data frame to read culture levels
        cultures = None
        try:
            df = result.model.data.frame
            cultures = pd.unique(df['culture'])
            # convert to strings for matching with param suffixes
            cultures_str = [str(c) for c in cultures]
        except Exception:
            cultures = None
            cultures_str = None

        # identify interaction parameter names and the culture labels they correspond to
        interaction_map = {}  # label_str -> param_name
        for pname in params.index:
            if ('C(culture)' in pname) and ('age' in pname) and (':' in pname):
                # extract the label inside C(culture)[T.xxx]
                m = re.search(r'C\(culture\)\[T\.?([^\]]+)\]', pname)
                if m:
                    lab = m.group(1)
                    interaction_map[str(lab)] = pname
                else:
                    # if no match, as fallback use full param name as label
                    interaction_map[pname] = pname

        # main age parameter must exist
        if 'age' not in list(params.index):
            # cannot proceed reliably
            baseline_stats = None
        else:
            baseline_stats = _linear_combo_stats({'age': 1.0}, params, cov_mat, cov_index)

        # Determine culture labels to report slopes for:
        culture_labels_to_report = []
        if cultures_str is not None:
            # include all observed culture levels (as strings)
            culture_labels_to_report = cultures_str
        else:
            # fallback: infer cultures from interaction_map keys and indicate a 'reference' level
            inferred = sorted(interaction_map.keys())
            # we can report 'reference' plus those inferred
            culture_labels_to_report = ['reference'] + inferred

        # Find reference label (the level that has no interaction param)
        reference_label = None
        if cultures_str is not None:
            # reference = those observed that are not in interaction_map keys
            ref_candidates = [c for c in cultures_str if c not in interaction_map]
            if len(ref_candidates) >= 1:
                # choose the first as reference
                reference_label = str(ref_candidates[0])
            else:
                # if none, set to first observed (will have to use baseline age only)
                reference_label = cultures_str[0] if cultures_str else 'reference'
        else:
            # fallback
            if 'reference' in culture_labels_to_report:
                reference_label = 'reference'
            else:
                reference_label = culture_labels_to_report[0]

        # compute slope stats per culture: slope_c = age + interaction_for_c (if exists)
        slopes = {}
        for lab in culture_labels_to_report:
            lab_str = str(lab)
            if lab_str == reference_label:
                # slope is baseline age
                if baseline_stats is not None:
                    slopes[lab_str] = dict(baseline_stats)  # copy
                    slopes[lab_str]['note'] = 'reference (baseline) culture slope for age'
                else:
                    slopes[lab_str] = None
            else:
                # find interaction param for this label
                inter_param = interaction_map.get(lab_str, None)
                if inter_param is None:
                    # no interaction estimated for this label -> slope equals baseline
                    slopes[lab_str] = dict(baseline_stats) if baseline_stats is not None else None
                    if slopes[lab_str] is not None:
                        slopes[lab_str]['note'] = 'no estimated interaction; slope equals baseline'
                else:
                    # slope = age + interaction_param
                    weights = {'age': 1.0, inter_param: 1.0}
                    stats = _linear_combo_stats(weights, params, cov_mat, cov_index)
                    stats['note'] = f"slope for culture {lab_str} = baseline age + interaction ({inter_param})"
                    slopes[lab_str] = stats

        # Additionally extract the raw interaction coefficients (for interpretation)
        interactions_raw = {}
        for lab, pname in interaction_map.items():
            inter_stats = _linear_combo_stats({pname: 1.0}, params, cov_mat, cov_index)
            interactions_raw[lab] = {'param_name': pname, **inter_stats}

        # Main outputs: baseline age, interactions, slopes
        out = {
            'baseline_age_param': {'param_name': 'age', **(baseline_stats if baseline_stats is not None else {})},
            'interactions': interactions_raw,
            'age_slopes_by_culture': slopes,
            'all_params': params.to_dict()  # include all coefficients for context (log-odds scale)
        }
        return out

    result = {}

    # Expecting model_output to be a dict with keys 'social_model' and 'majority_model'
    social_res = model_output.get('social_model') if isinstance(model_output, dict) else None
    majority_res = model_output.get('majority_model') if isinstance(model_output, dict) else None

    result['social_model'] = _extract_model_info(social_res)
    result['majority_model'] = _extract_model_info(majority_res) if majority_res is not None else None

    description_lines = [
        "Extracted statistics (coef, cluster-robust SE, z, two-sided p-value, 95% CI) for:",
        "- the baseline 'age' coefficient (this is the reference culture slope of age on the log-odds of the outcome),",
        "- each culture-specific age slope (baseline age + the age:C(culture) interaction when present),",
        "- the raw interaction coefficients age:C(culture)[T.x] (so you can see how each culture's slope differs from baseline).",
        "",
        "Interpretation notes:",
        "- Coefficients are on the log-odds scale (logit). A positive age slope means the log-odds of the outcome",
        "  (relying on social information, or preferring the majority among social learners) increase with age.",
        "- Culture-specific slopes show whether the developmental trajectory (effect of age) differs by site.",
        "- Significance is given by the two-sided p-value; typically p < 0.05 indicates a statistically significant effect.",
        "- Covariance and standard errors come from the cluster-robust covariance matrix provided/attached to the model results,",
        "  so the inference accounts for within-site clustering as implemented in the modeling code."
    ]

    return {
        "object": result,
        "description": "\n".join(description_lines)
    }