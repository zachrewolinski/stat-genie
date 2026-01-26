def extract_final_answer(model_output):
    """
    Extracts the age-related developmental slopes for choosing the majority option
    across cultural sites from a fitted statsmodels MNLogit results object.

    Returns a dictionary with:
      - "object": pandas.DataFrame with rows per culture and columns:
            ['culture', 'estimate', 'se', 'z', 'p', 'ci_lower', 'ci_upper', 'scale']
          where:
            * estimate = per-unit-age effect (interpreted below in 'scale')
            * scale = 'log-odds (vs base)' if the model parameterization directly
                      contains the majority outcome equation; otherwise
                      'probability' if numerical marginal effect on probability was used.
      - "description": human-readable explanation of what the numbers mean.

    The function handles two cases:
      1) The multinomial model has a parameter vector for the majority outcome
         (i.e., log-odds(majority vs base) parameters are present). In that case
         the returned 'estimate' is the change in log-odds of choosing the majority
         (compared to the model's baseline category) per unit increase in age for
         each culture. Standard errors are computed by combining relevant parameters
         using the model covariance matrix.
      2) If the majority outcome is the model's baseline (no direct parameter column),
         the function computes a numerical average marginal effect of age on the
         predicted probability of choosing the majority option (via finite differences),
         and estimates its standard error by numerically constructing the Jacobian
         w.r.t. the model parameters and using the parameter covariance matrix.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output  # statsmodels MNLogitResultsWrapper

    # Basic info
    try:
        # original outcome labels (unique sorted)
        endog = np.asarray(res.model.endog)
        categories = np.unique(endog)
    except Exception:
        categories = None

    # We expect majority label coded as 2 per problem statement
    majority_label = 2

    # Parameter table (DataFrame) with rows = exog names, columns = outcome labels (non-base)
    params = res.params  # DataFrame
    # Covariance matrix for parameters (DataFrame or ndarray)
    cov = res.cov_params()

    # Helper: flatten params into a Series indexed by (exog, outcome)
    try:
        params_stack = params.stack()  # MultiIndex (exog, outcome)
        params_vec = params_stack.copy()
        params_index = params_vec.index  # tuples
    except Exception:
        # If params is 1-D (unlikely), handle gracefully
        params_vec = pd.Series(params.values.flatten())
        params_index = params_vec.index

    # Determine whether the model has explicit parameters for the majority outcome
    has_majority_params = False
    try:
        # params.columns may be dtype numeric or string; compare using equality
        if majority_label in params.columns:
            has_majority_params = True
        else:
            # Try string form
            if str(majority_label) in params.columns.astype(str).tolist():
                # map to actual column label
                majority_col = [c for c in params.columns if str(c) == str(majority_label)][0]
                has_majority_params = True
                majority_label = majority_col
    except Exception:
        has_majority_params = False

    results_rows = []

    # Determine cultures present from exog names (those starting with 'culture_' and also reference)
    exog_names = res.model.exog_names
    # Identify culture dummy names (e.g., 'culture_2', 'culture_3', ...)
    culture_dummies = [n for n in exog_names if n.startswith('culture_')]
    # Derive culture labels: reference culture is the one omitted (culture_1)
    # We'll create a list of culture identifiers including reference 'culture_1'
    cultures = []
    # parse numbers from the dummy names
    parsed = []
    for cd in culture_dummies:
        # assume format 'culture_X'
        try:
            num = int(cd.split('_')[-1])
            parsed.append(num)
        except Exception:
            # keep name verbatim as fallback
            parsed.append(cd)
    # include reference culture 1 as well
    unique_cultures = sorted([p for p in parsed])
    # Map to labels like 'culture_1', 'culture_2', ...
    culture_labels = []
    for p in unique_cultures:
        if isinstance(p, int):
            culture_labels.append(f'culture_{p}')
        else:
            culture_labels.append(str(p))
    # Ensure culture_1 (reference) present
    if 'culture_1' not in culture_labels:
        culture_labels = ['culture_1'] + [c for c in culture_labels if c != 'culture_1']

    # Function to safely get param value given exog name and outcome label
    def get_param(exog_name, outcome_label):
        try:
            return params.loc[exog_name, outcome_label]
        except Exception:
            # maybe params index/columns are reversed or strings differ; attempt stack lookup
            try:
                return params_stack.loc[exog_name, outcome_label]
            except Exception:
                # fallback NaN
                return np.nan

    # Covariance matrix alignment helper:
    # We'll try to produce cov matrix C and vector of parameter names in same order as params_vec
    # If cov is a DataFrame with matching MultiIndex, use it directly. Otherwise fallback to numpy array.
    try:
        cov_df = cov
        # If cov_df has MultiIndex equal to params_vec.index, good
        if hasattr(cov_df.index, 'equals') and cov_df.index.equals(params_vec.index):
            cov_matrix = cov_df.values
            cov_index = params_vec.index
        else:
            # Attempt to see if cov index is strings matching "exog, outcome" when joined
            cov_index = cov_df.index
            # create string keys for params_vec
            def mkkey(t):
                return f"{t[0]}|{t[1]}"
            params_keys = [mkkey(t) for t in params_vec.index]
            cov_keys = [str(i) for i in cov_index]
            if set(params_keys).issubset(set(cov_keys)):
                # reorder cov to params order
                order = [cov_keys.index(k) for k in params_keys]
                cov_matrix = cov_df.values[np.ix_(order, order)]
            else:
                # Last fallback: treat cov as numpy with same order as params_vec
                cov_matrix = np.asarray(cov_df)
    except Exception:
        # Final fallback: numeric covariance array
        cov_matrix = np.asarray(cov)

    # If we can treat params_vec as numpy in a consistent order:
    params_vector = np.asarray(params_vec.values).reshape(-1)

    # Helper to compute linear-combination variance when L is vector aligned with params_vec
    def lincomb_var_and_se(L):
        L = np.asarray(L).reshape(-1)
        var = float(L.dot(cov_matrix).dot(L))
        se = np.sqrt(max(var, 0.0))
        return var, se

    # Case A: model directly contains majority outcome parameters
    if has_majority_params:
        # Identify the exact column label used for majority in params.columns
        # majority_label already set to matching column if needed
        maj_col = majority_label

        for cult in culture_labels:
            # For reference culture 'culture_1', culture dummy is all zeros and interactions absent
            # The marginal log-odds slope of majority vs base = beta_age + beta_age:culture_k (if culture_k present)
            # Build linear combination L that picks out those parameters from params_vec
            # Initialize L of zeros
            L = np.zeros_like(params_vector, dtype=float)
            # We need to find index in params_vec for (exog='age_c', outcome=maj_col)
            try:
                age_idx = list(params_vec.index).index(('age_c', maj_col))
                L[age_idx] = 1.0
            except ValueError:
                # if not found, try string forms
                matched = None
                for i, idx in enumerate(params_vec.index):
                    if str(idx[0]) == 'age_c' and str(idx[1]) == str(maj_col):
                        matched = i
                        break
                if matched is None:
                    # cannot find age param for majority; skip
                    estimate = np.nan
                    se = np.nan
                    z = np.nan
                    p = np.nan
                    ci_low = np.nan
                    ci_high = np.nan
                    results_rows.append({
                        'culture': cult,
                        'estimate': estimate,
                        'se': se,
                        'z': z,
                        'p': p,
                        'ci_lower': ci_low,
                        'ci_upper': ci_high,
                        'scale': 'log-odds (vs base)'
                    })
                    continue
                else:
                    age_idx = matched
                    L[age_idx] = 1.0

            # If not reference culture, add interaction term if present
            if cult != 'culture_1':
                inter_name = f'age_c:{cult}'
                # try both 'age_c:culture_X' and 'age_c:culture_X' patterns if model used different separator
                # but per problem it's 'age_c:culture_X'
                try:
                    inter_idx = list(params_vec.index).index((inter_name, maj_col))
                    L[inter_idx] = 1.0
                except ValueError:
                    # try alternative ordering exog name might be 'age_c:culture_X' literally
                    try:
                        inter_idx = list(params_vec.index).index((f'age_c:{cult}', maj_col))
                        L[inter_idx] = 1.0
                    except Exception:
                        # If interaction not present in params (e.g., no data for that culture),
                        # then treat interaction coefficient as 0 (no additional effect)
                        pass

            # Compute estimate and SE
            estimate = float(params_vector.dot(L))
            var, se = lincomb_var_and_se(L)
            z = estimate / se if se > 0 else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            ci_low = estimate - 1.96 * se
            ci_high = estimate + 1.96 * se

            results_rows.append({
                'culture': cult,
                'estimate': estimate,
                'se': se,
                'z': z,
                'p': p,
                'ci_lower': ci_low,
                'ci_upper': ci_high,
                'scale': 'log-odds (vs base)'
            })

    else:
        # Case B: majority is the model baseline and no direct parameter column exists.
        # We'll compute numerical marginal effect of age on the predicted probability of choosing majority
        # for each culture at a central covariate configuration (age_c = 0, is_boy=0, majority_first=0).
        # We'll also compute a SE via numerical Jacobian and parameter covariance.

        # Build a template exog vector using the model's exog_names; set defaults:
        exog_names = res.model.exog_names
        # create a dict with default zeros
        base_exog = {n: 0.0 for n in exog_names}
        # set intercept if present
        if 'const' in exog_names:
            base_exog['const'] = 1.0
        # age_c baseline at 0 (centered)
        base_exog['age_c'] = 0.0
        # is_boy and majority_first baseline 0
        if 'is_boy' in exog_names:
            base_exog['is_boy'] = 0.0
        if 'majority_first' in exog_names:
            base_exog['majority_first'] = 0.0

        # Function to build exog row for a given culture and age_c
        def make_exog_row(culture_label, age_val):
            row = base_exog.copy()
            row['age_c'] = age_val
            # set culture dummies according to culture_label
            for cd in culture_dummies:
                row[cd] = 0.0
            if culture_label != 'culture_1':
                # turn on the matching dummy if present
                if culture_label in exog_names:
                    row[culture_label] = 1.0
                else:
                    # sometimes culture dummy naming could differ; attempt to find matching name
                    matches = [n for n in exog_names if n.endswith(culture_label.split('_')[-1])]
                    if matches:
                        row[matches[0]] = 1.0
            # set interactions: if exog includes explicit interaction columns like 'age_c:culture_2',
            # set them equal to age_val * culture_dummy (which will be age_val for the active culture)
            for name in exog_names:
                if name.startswith('age_c:') or name.startswith('age_c:culture_'):
                    # expected format 'age_c:culture_X'
                    parts = name.split(':')
                    if len(parts) >= 2:
                        cult_part = parts[-1]
                        if culture_label == cult_part:
                            row[name] = age_val
                        else:
                            row[name] = 0.0
            # Return as 2D array in the correct column order
            exog_row = np.array([row[n] for n in exog_names], dtype=float).reshape(1, -1)
            return exog_row

        # small epsilon for finite differences
        h = 1e-4

        # Flattened parameter vector and covariance already available: params_vector, cov_matrix
        # Create helper to compute predicted probability of majority given params vector b and exog row
        def predict_prob_with_params(b_vec, exog_row):
            # We need to reconstruct parameter matrix shape = (n_exog, n_nonbase_outcomes)
            # params originally had shape (n_exog, n_outcomes_nonbase). We can attempt to reshape accordingly.
            n_exog = len(exog_names)
            # determine number of non-base outcomes from params shape
            try:
                n_nonbase = params.shape[1]
            except Exception:
                # fallback: infer from unique second index in params_vec.index
                n_nonbase = len(np.unique([t[1] for t in params_vec.index]))
            # reshape b_vec into (n_exog, n_nonbase) in the same order as params_vec
            try:
                b_mat = b_vec.reshape((n_exog, n_nonbase), order='F')
            except Exception:
                # fallback: try row-major
                b_mat = b_vec.reshape((n_exog, n_nonbase), order='C')
            # compute linear predictors X @ B for non-base outcomes -> shape (1, n_nonbase)
            linpred = exog_row.dot(b_mat)  # (1, n_nonbase)
            # softmax over full set of outcomes including base: we need to insert zero for baseline outcome
            # Determine the ordering of outcomes that the model uses in predict: statsmodels predict returns
            # probabilities in ascending order of category labels (most likely), but safer to use model._ynames maybe.
            # We'll use model.predict which returns array with shape (1, n_outcomes). But here,
            # because we are manipulating parameters directly, reconstruct full probability vector:
            # For non-base outcomes, use exp(linpred); for baseline outcome, value = 1 (exp(0))
            exps = np.exp(linpred).ravel()
            all_exps = np.concatenate([exps, np.array([1.0])])  # append baseline as last
            probs = all_exps / all_exps.sum()
            # Need to map which column corresponds to majority_label. We assume baseline corresponds to the
            # omitted category and was placed last; earlier we determined majority was baseline, so majority
            # probability = last element of probs
            # If majority is not the baseline in this branch, this function will not be reached.
            return probs

        # For each culture compute numerical derivative and its SE
        for cult in culture_labels:
            # exog for age = +h and -h
            exog_plus = make_exog_row(cult, h)
            exog_minus = make_exog_row(cult, -h)

            # Predict probabilities using model.predict (which uses fitted params)
            try:
                p_plus = res.predict(exog_plus)[0]  # returns array with probs for all outcomes
                p_minus = res.predict(exog_minus)[0]
            except Exception:
                # If predict requires a DataFrame with column names:
                exog_df_plus = pd.DataFrame(exog_plus, columns=exog_names)
                exog_df_minus = pd.DataFrame(exog_minus, columns=exog_names)
                p_plus = res.predict(exog_df_plus).iloc[0].values
                p_minus = res.predict(exog_df_minus).iloc[0].values

            # Map which index in p arrays corresponds to majority label
            maj_idx = None
            if categories is not None:
                # If predict returns probabilities in order of categories, find index
                try:
                    # categories sorted; find index of majority_label
                    maj_idx = list(categories).index(majority_label)
                except Exception:
                    # fallback: assume baseline is last
                    maj_idx = -1
            else:
                maj_idx = -1

            # Numerical derivative of majority probability wrt age
            p_maj_plus = p_plus[maj_idx]
            p_maj_minus = p_minus[maj_idx]
            estimate = (p_maj_plus - p_maj_minus) / (2 * h)

            # Now compute SE via numerical Jacobian wrt parameters:
            # For each parameter j (in params_vector), compute dp/db_j at base exog (age=0)
            # We'll use central difference on parameter vector: for parameter j add small eps to param j.
            eps = 1e-6
            n_params = len(params_vector)
            # Precompute base probs at current fitted params (we'll use res.predict at age 0)
            exog_base = make_exog_row(cult, 0.0)
            try:
                p_base = res.predict(exog_base)[0]
            except Exception:
                p_base = res.predict(pd.DataFrame(exog_base, columns=exog_names)).iloc[0].values
            # We need dp_db vector for the majority probability only
            dp_db = np.zeros(n_params, dtype=float)
            # To avoid a very large loop for huge models, keep eps small; typical models ok.
            for j in range(n_params):
                # perturb parameter j by +eps and -eps and compute predicted majority probability
                b_plus = params_vector.copy()
                b_minus = params_vector.copy()
                b_plus[j] += eps
                b_minus[j] -= eps
                p_plus_par = predict_prob_with_params(b_plus, exog_base)
                p_minus_par = predict_prob_with_params(b_minus, exog_base)
                dp_db[j] = (p_plus_par[maj_idx] - p_minus_par[maj_idx]) / (2 * eps)
            # variance = dp_db' * cov * dp_db
            var = float(dp_db.dot(cov_matrix).dot(dp_db))
            se = np.sqrt(max(var, 0.0))
            z = estimate / se if se > 0 else np.nan
            pval = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            ci_low = estimate - 1.96 * se
            ci_high = estimate + 1.96 * se

            results_rows.append({
                'culture': cult,
                'estimate': estimate,
                'se': se,
                'z': z,
                'p': pval,
                'ci_lower': ci_low,
                'ci_upper': ci_high,
                'scale': 'probability'
            })

    # Assemble results DataFrame
    results_df = pd.DataFrame(results_rows)

    description = (
        "For each culture, 'estimate' is the per-unit-age effect on choosing the majority option.\n"
        "- If 'scale' == 'log-odds (vs base)': estimate = change in log-odds of choosing majority "
        "(vs model baseline category) per 1 unit increase in centered age. Positive = older children more likely.\n"
        "- If 'scale' == 'probability': estimate = approximate change in predicted probability of choosing majority "
        "per 1 unit increase in centered age (estimated by finite differences). Positive = probability increases with age.\n\n"
        "Columns: culture (culture dummy name), estimate, se (standard error), z, p (two-sided p-value), "
        "ci_lower and ci_upper (95% CI), scale (interpretation)."
    )

    return {"object": results_df, "description": description}