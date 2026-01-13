def extract_final_answer(model_output):
    """
    Extracts the effect of 'Children' on reported affairs from a fitted
    statsmodels ZeroInflatedNegativeBinomialResults (wrapper) object.

    Returns a dictionary:
      - "object": a dict with numeric summaries for the count-model effect
                  of Children for females and males (coef, se, z, p, 95% CI,
                  incidence-rate-ratio (IRR) and its 95% CI), plus the
                  inflation-model coefficient for Children (logit), its p-value
                  and interpretation.
      - "description": brief plain-language interpretation about what the
                       extracted statistics mean for whether children decrease
                       extramarital affairs.

    The function is robust to parameter naming conventions used by statsmodels:
    it searches for parameter names for the count model and the inflation model.
    """
    import numpy as np
    import math

    res = model_output  # alias

    # Get parameter series and covariance; handle different container types
    try:
        params = res.params  # pandas Series if available
    except Exception:
        params = np.asarray(res.params)

    # Try to access indexable names for parameters (pandas Index) if present
    param_names = None
    if hasattr(params, "index"):
        param_names = list(params.index)
    else:
        # fallback: if model stores names on the model object
        try:
            param_names = list(res.model.param_names)
        except Exception:
            param_names = None

    # Helper to find a parameter name for the count model (prefer exact matches)
    def find_count_param(base_name):
        if param_names is None:
            return None
        # Prefer exact match
        if base_name in param_names:
            return base_name
        # Prefer names that exactly match and are not inflation-prefixed
        for nm in param_names:
            if nm.endswith(base_name) and not nm.startswith("inflate"):
                return nm
        # As last resort, return any name that equals base_name
        for nm in param_names:
            if nm == base_name:
                return nm
        return None

    # Helper to find inflation parameter name (likely prefixed with 'inflate_' or 'inflate.')
    def find_infl_param(base_name):
        if param_names is None:
            return None
        keys = []
        for nm in param_names:
            if ("inflate" in nm and base_name in nm) or nm == f"inflate_{base_name}" or nm.endswith("." + base_name):
                keys.append(nm)
        # If multiple, prefer exact inflate_{base_name}
        for nm in keys:
            if nm == f"inflate_{base_name}":
                return nm
        if keys:
            return keys[0]
        # If nothing found, also check for plain base_name that might correspond to inflation (edge cases)
        if base_name in param_names:
            return base_name
        return None

    # Identify parameter names
    child_name = find_count_param("Children")
    interaction_name = find_count_param("Children_x_Male") or find_count_param("Children:GenderMale") or find_count_param("Children_x_Male")  # try a few variants
    # If interaction not found, try pattern search
    if interaction_name is None and param_names is not None:
        for nm in param_names:
            if ("Children" in nm and ("Male" in nm or "Gender" in nm) and not nm.startswith("inflate")):
                interaction_name = nm
                break

    infl_child_name = find_infl_param("Children")

    # Extract coefficient values (handle when params is Series or numpy array)
    def get_param_value(name):
        if name is None:
            return None
        try:
            return float(params[name])
        except Exception:
            # if params is ndarray and param_names exist, map by index
            if param_names is not None and name in param_names:
                idx = param_names.index(name)
                return float(np.asarray(params)[idx])
        return None

    # Get covariance matrix (DataFrame or ndarray)
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    def get_cov(a, b):
        # return covariance between params a and b
        if cov is None or param_names is None:
            return None
        # cov may be DataFrame or ndarray
        if hasattr(cov, "loc"):
            return float(cov.loc[a, b])
        else:
            # ndarray: map names to indices
            i = param_names.index(a)
            j = param_names.index(b)
            return float(np.asarray(cov)[i, j])

    # Basic availability checks
    if child_name is None:
        raise KeyError("Could not find a parameter name for 'Children' in the model output parameters.")
    # Interaction may be missing; treat as zero if absent
    child_coef = get_param_value(child_name)
    child_se = None
    child_p = None
    child_ci = (None, None)
    try:
        # try to fetch standard error and conf int for child_name
        if hasattr(res, "bse") and child_name in list(res.bse.index):
            child_se = float(res.bse[child_name])
            # p-value from res.pvalues if available
            if hasattr(res, "pvalues"):
                child_p = float(res.pvalues[child_name])
            # conf int
            if hasattr(res, "conf_int"):
                ci_df = res.conf_int()
                if child_name in list(ci_df.index):
                    child_ci = (float(ci_df.loc[child_name, 0]), float(ci_df.loc[child_name, 1]))
    except Exception:
        child_se = None

    # Get interaction
    inter_coef = get_param_value(interaction_name) if interaction_name is not None else 0.0

    # Compute female effect (GenderMale=0): just child_coef
    female_coef = child_coef
    # Standard error for female is just se(child)
    if child_se is None and cov is not None:
        # try to get from covariance diag
        try:
            var = get_cov(child_name, child_name)
            child_se = math.sqrt(var)
        except Exception:
            child_se = None

    female_se = child_se
    female_z = None
    female_p = None
    female_ci = (None, None)
    if female_se is not None and female_se > 0:
        female_z = female_coef / female_se
        female_p = math.erfc(abs(female_z) / math.sqrt(2))  # two-sided p from normal approx
        female_ci = (female_coef - 1.96 * female_se, female_coef + 1.96 * female_se)

    # Compute male effect (GenderMale=1): child_coef + interaction_coef
    if inter_coef is None:
        inter_coef = 0.0
    male_coef = female_coef + inter_coef

    # SE for linear combination: var(child) + var(interaction) + 2 cov(child,interaction)
    male_se = None
    if cov is not None and interaction_name is not None:
        try:
            v1 = get_cov(child_name, child_name)
            v2 = get_cov(interaction_name, interaction_name)
            cov12 = get_cov(child_name, interaction_name)
            male_var = v1 + v2 + 2.0 * cov12
            if male_var >= 0:
                male_se = math.sqrt(male_var)
        except Exception:
            male_se = None
    else:
        # If interaction absent (treated as 0), male_se = female_se
        if interaction_name is None:
            male_se = female_se

    male_z = None
    male_p = None
    male_ci = (None, None)
    if male_se is not None and male_se > 0:
        male_z = male_coef / male_se
        male_p = math.erfc(abs(male_z) / math.sqrt(2))
        male_ci = (male_coef - 1.96 * male_se, male_coef + 1.96 * male_se)

    # Convert log-coef to incidence-rate-ratio (IRR) and CIs
    def irr_and_ci(logcoef, lo_ci, hi_ci):
        if logcoef is None:
            return (None, (None, None))
        try:
            irr = math.exp(logcoef)
            if lo_ci is None or hi_ci is None:
                return (irr, (None, None))
            return (irr, (math.exp(lo_ci), math.exp(hi_ci)))
        except Exception:
            return (None, (None, None))

    female_irr, female_irr_ci = irr_and_ci(female_coef, female_ci[0], female_ci[1])
    male_irr, male_irr_ci = irr_and_ci(male_coef, male_ci[0], male_ci[1])

    # Inflation model: coefficient for Children (on logit scale), interpretation:
    infl_coef = get_param_value(infl_child_name)
    infl_se = None
    infl_p = None
    infl_ci = (None, None)
    if infl_child_name is not None and hasattr(res, "bse") and infl_child_name in list(res.bse.index):
        try:
            infl_se = float(res.bse[infl_child_name])
            if hasattr(res, "pvalues"):
                infl_p = float(res.pvalues[infl_child_name])
            if hasattr(res, "conf_int"):
                ci_df = res.conf_int()
                if infl_child_name in list(ci_df.index):
                    infl_ci = (float(ci_df.loc[infl_child_name, 0]), float(ci_df.loc[infl_child_name, 1]))
        except Exception:
            infl_se = None
    else:
        # try to get from covariance diag
        try:
            if cov is not None and infl_child_name is not None:
                v = get_cov(infl_child_name, infl_child_name)
                infl_se = math.sqrt(v)
                if infl_se is not None and infl_se > 0:
                    z = infl_coef / infl_se
                    infl_p = math.erfc(abs(z) / math.sqrt(2))
                    infl_ci = (infl_coef - 1.96 * infl_se, infl_coef + 1.96 * infl_se)
        except Exception:
            infl_se = None

    # Interpretation for inflation coef:
    if infl_coef is None:
        infl_interpretation = "No inflation-model Children parameter found."
    else:
        # positive infl_coef => higher log-odds of being an 'excess zero' (structural zero), i.e., more likely to be in always-zero group => reduces probability of any affairs
        sign_text = "increases" if infl_coef > 0 else "decreases"
        infl_interpretation = (
            f"In the inflation (logit) part of the model, the Children coefficient is {infl_coef:.4g} "
            f"(p={infl_p:.4g} if available). A positive value would {sign_text} the odds of being in the "
            "structural-zero group (i.e., being 'certain' to have zero affairs); thus a positive sign implies "
            "children are associated with a higher probability of reporting zero affairs (reducing any-affair probability)."
        )

    # Build final structured object
    results_dict = {
        "param_names_used": {
            "children_count_param": child_name,
            "children_interaction_param": interaction_name,
            "children_inflation_param": infl_child_name,
        },
        "children_count_effect_female": {
            "log_coef": female_coef,
            "se": female_se,
            "z": female_z,
            "p_value": female_p,
            "95CI_log": female_ci,
            "IRR": female_irr,
            "95CI_IRR": female_irr_ci,
            "interpretation": (
                "For females (GenderMale=0): the count-model coefficient is on the log scale. "
                "IRR < 1 means children are associated with fewer expected affairs; IRR > 1 means more."
            ),
        },
        "children_count_effect_male": {
            "log_coef": male_coef,
            "se": male_se,
            "z": male_z,
            "p_value": male_p,
            "95CI_log": male_ci,
            "IRR": male_irr,
            "95CI_IRR": male_irr_ci,
            "interpretation": (
                "For males (GenderMale=1): this is the sum of the Children main effect and the Children×Male interaction. "
                "Same IRR interpretation applies."
            ),
        },
        "children_inflation_effect": {
            "logit_coef": infl_coef,
            "se": infl_se,
            "p_value": infl_p,
            "95CI_logit": infl_ci,
            "interpretation": infl_interpretation,
        },
    }

    # High-level description (concise)
    # We avoid making a categorical yes/no here because significance and direction depend on the estimates:
    desc_lines = []
    desc_lines.append("Summary of extracted effects for 'Children' on extramarital affairs (from the ZINB model):")
    desc_lines.append("- Count model (expected number of affairs):")
    desc_lines.append(
        "  * Females: log-coef = {:+.4g}, IRR = {:.4g}, 95% CI for IRR = ({:.4g}, {:.4g}), p = {}".format(
            female_coef if female_coef is not None else float("nan"),
            female_irr if female_irr is not None else float("nan"),
            female_irr_ci[0] if female_irr_ci[0] is not None else float("nan"),
            female_irr_ci[1] if female_irr_ci[1] is not None else float("nan"),
            ("{:.4g}".format(female_p) if female_p is not None else "NA"),
        )
    )
    desc_lines.append(
        "  * Males:   log-coef = {:+.4g}, IRR = {:.4g}, 95% CI for IRR = ({:.4g}, {:.4g}), p = {}".format(
            male_coef if male_coef is not None else float("nan"),
            male_irr if male_irr is not None else float("nan"),
            male_irr_ci[0] if male_irr_ci[0] is not None else float("nan"),
            male_irr_ci[1] if male_irr_ci[1] is not None else float("nan"),
            ("{:.4g}".format(male_p) if male_p is not None else "NA"),
        )
    )
    desc_lines.append("- Inflation model (probability of being a structural zero):")
    desc_lines.append(
        "  * Children inflation logit-coef = {:+.4g}, p = {}".format(
            infl_coef if infl_coef is not None else float("nan"),
            ("{:.4g}".format(infl_p) if infl_p is not None else "NA"),
        )
    )
    desc_lines.append("")
    desc_lines.append(
        "Interpretation guidance: In the count model, a negative log-coefficient (IRR < 1) indicates that having children "
        "is associated with fewer expected extramarital incidents. In the inflation model, a positive coefficient means "
        "children increase the odds of being in the 'certain zero' group (i.e., more likely to report zero affairs). "
        "To decide whether 'having children decreases engagement in extramarital affairs', examine both: a statistically "
        "significant IRR < 1 in the count model and/or a positive, significant inflation coefficient both support the claim."
    )

    description = "\n".join(desc_lines)

    return {"object": results_dict, "description": description}