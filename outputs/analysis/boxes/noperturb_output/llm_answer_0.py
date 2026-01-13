import numpy as np
import math
from scipy import stats as st


def extract_final_answer(model_output):
    """
    Extracts statistics relevant to how reliance on the majority option changes with age
    across cultural sites from a fitted statsmodels GLMResultsWrapper.

    Returns a dictionary:
      {
        "object": {
           "age_c": {coef, se, z, p, ci_lower, ci_upper, OR, OR_ci_lower, OR_ci_upper},
           "age2": { ... }  # if present
           "site_slopes": {
               "culture_1": {coef_logit, se, t, p, ci, OR, OR_CI},
               "culture_2": {...}, ...
           },
           "joint_interaction_p": <p-value testing all age_x_culture coefficients = 0 (if any interactions)>,
           "notes": "slope = change in log-odds per centered year; OR = multiplicative change in odds per year"
        },
        "description": "Interpretation..."
      }

    model_output: statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
    """
    res = model_output

    # Basic parameter tables
    params = res.params  # pandas Series
    bse = res.bse
    pvalues = res.pvalues
    conf = res.conf_int()  # DataFrame with columns [0,1]
    cov = res.cov_params()  # covariance matrix DataFrame or ndarray
    param_names = list(params.index)

    out = {}
    # Helper to safe-get param stats
    def _param_stats(name):
        if name not in params.index:
            return None
        coef = float(params[name])
        se = float(bse[name])
        # For GLM (logit) params, statsmodels reports z-values for large-sample; compute z = coef / se
        z = coef / se if se != 0 else np.nan
        p = float(pvalues[name])
        ci_lower = float(conf.loc[name, 0])
        ci_upper = float(conf.loc[name, 1])
        or_ = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower))
        or_ci_upper = float(np.exp(ci_upper))
        return {
            "coef_logit": coef,
            "se": se,
            "z_or_t": z,
            "p": p,
            "ci_logit": [ci_lower, ci_upper],
            "OR": or_,
            "OR_CI": [or_ci_lower, or_ci_upper],
        }

    # Extract age_c main effect
    age_stats = _param_stats("age_c")
    if age_stats is None:
        raise ValueError("Model does not contain 'age_c' parameter; cannot extract age effect.")

    out["age_c"] = age_stats

    # Extract age2 if present
    age2_stats = _param_stats("age2")
    if age2_stats is not None:
        out["age2"] = age2_stats

    # Identify culture dummy params and interaction params present in the model
    culture_params = [n for n in param_names if n.startswith("culture_")]
    # Infer available site ids from culture params (e.g., 'culture_2' -> 2)
    site_ids = set()
    for cname in culture_params:
        try:
            site_ids.add(int(cname.split("_", 1)[1]))
        except Exception:
            pass
    # Always include culture_1 as the reference site
    site_ids.add(1)
    site_ids = sorted(site_ids)

    # Identify interaction parameters: age_c_x_culture_{k}
    inter_params = [n for n in param_names if n.startswith("age_c_x_culture_")]
    inter_site_ids = []
    for iname in inter_params:
        try:
            sid = int(iname.split("age_c_x_culture_")[1])
            inter_site_ids.append(sid)
        except Exception:
            pass

    # Prepare slopes per site: slope = age_c + age_c_x_culture_k (if present), otherwise age_c alone (reference)
    site_slopes = {}
    # prepare index lookup
    k = len(param_names)

    # Prepare covariance matrix as numpy array aligned to param_names ordering
    if hasattr(cov, "values"):
        cov_mat = np.asarray(cov.values)
    else:
        cov_mat = np.asarray(cov)

    param_values = np.asarray(params.values)

    for sid in site_ids:
        # Build R vector for the linear combination (1 * age_c + 1 * age_c_x_culture_{sid} if present)
        r = np.zeros((k,))
        # index of age_c
        try:
            idx_age = param_names.index("age_c")
        except ValueError:
            raise ValueError("'age_c' not found among model parameters.")
        r[idx_age] = 1.0
        inter_name = f"age_c_x_culture_{sid}"
        used_inter = False
        if sid != 1 and inter_name in param_names:
            idx_inter = param_names.index(inter_name)
            r[idx_inter] = 1.0
            used_inter = True

        # Compute effect and its variance using the covariance matrix
        effect = float(np.dot(r, param_values))
        # variance: r' * cov * r
        try:
            var = float(r.dot(cov_mat).dot(r))
        except Exception:
            var = float(np.dot(np.dot(r, cov_mat), r))
        sd = math.sqrt(var) if var >= 0 else float("nan")
        # Use normal approximation (z) for GLM large-sample inference
        zval = effect / sd if sd != 0 and not math.isnan(sd) else float("nan")
        pval = float(2.0 * st.norm.sf(abs(zval))) if not math.isnan(zval) else float("nan")
        ci_lower = effect - 1.96 * sd if not math.isnan(sd) else float("nan")
        ci_upper = effect + 1.96 * sd if not math.isnan(sd) else float("nan")

        site_slopes[f"culture_{sid}"] = {
            "slope_logit": effect,
            "se": sd,
            "t": zval,
            "p": pval,
            "ci_logit": [ci_lower, ci_upper],
            "OR_per_year": float(np.exp(effect)),
            "OR_CI": [float(np.exp(ci_lower)), float(np.exp(ci_upper))] if not math.isnan(ci_lower) else [None, None],
            "used_interaction": used_inter,
        }

    out["site_slopes"] = site_slopes

    # Joint test: are all age_c_x_culture_* coefficients = 0? (i.e., do slopes differ across sites?)
    if len(inter_params) > 0:
        # Build R matrix with one row per interaction, picking that parameter
        R_joint = np.zeros((len(inter_params), k))
        for i, name in enumerate(inter_params):
            idx = param_names.index(name)
            R_joint[i, idx] = 1.0
        # Wald test for joint hypothesis that all listed interactions == 0
        try:
            wtest = res.wald_test(R_joint)
            try:
                joint_p = float(wtest.pvalue)
            except Exception:
                # some versions expose 'pvalue' as attribute, others as method
                joint_p = float(getattr(wtest, "pvalue", None) or getattr(wtest, "pvals", None) or float("nan"))
        except Exception:
            # As a fallback, compute Wald statistic manually: W = b' (R (cov) R')^{-1} b
            # where b = R * params_vector (for the interaction rows)
            try:
                R_mat = R_joint
                b_vec = R_mat.dot(param_values)
                cov_sub = R_mat.dot(cov_mat).dot(R_mat.T)
                # invert cov_sub
                inv_cov_sub = np.linalg.pinv(cov_sub)
                W = float(b_vec.dot(inv_cov_sub).dot(b_vec))
                # degrees of freedom = number of restrictions = rows of R_mat
                df = R_mat.shape[0]
                # p-value from chi-square
                joint_p = float(1.0 - st.chi2.cdf(W, df))
            except Exception:
                joint_p = None
        out["joint_interaction_p"] = joint_p
    else:
        out["joint_interaction_p"] = None

    # Include a short notes field explaining interpretation
    notes = (
        "All 'slope' values are change in log-odds of choosing the majority option per centered year of age.\n"
        "OR_per_year is exp(slope): multiplicative change in odds of choosing majority per additional year.\n"
        "site_slopes['culture_1'] is the reference site's age slope (no interaction). "
        "For culture_k, slope = age_c + age_c_x_culture_k (if interaction present).\n"
        "joint_interaction_p tests whether the age slope differs across sites (null: all age_x_culture = 0)."
    )
    out["notes"] = notes

    description = (
        "This output provides the estimated age effect (age_c) and quadratic term (age2 if present), "
        "per-site combined age slopes (age_c plus any site-specific interaction), their standard errors, "
        "test statistics and p-values, 95% confidence intervals on the log-odds scale, and the corresponding "
        "odds ratios (OR) with CIs. The key test for whether developmental trajectories differ across cultural sites "
        "is 'joint_interaction_p': a small p-value indicates that age-related slopes for choosing the majority "
        "option differ across sites."
    )

    return {"object": out, "description": description}