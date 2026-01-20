def extract_final_answer(model_output):
    """
    Extract age-related (linear and quadratic) coefficients and their interactions with culture
    from a fitted statsmodels Logit model, and compute combined coefficients, standard errors,
    z-values, p-values, 95% CIs, and odds ratios (with CIs) for each culture level.
    
    Returns:
      {
        "object": {
            "<culture_level>": {
                "reference": True/False,
                "age_linear_coef": float,
                "age_linear_se": float,
                "age_linear_z": float,
                "age_linear_p": float,
                "age_linear_ci95": [low, high],
                "age_linear_or": float,
                "age_linear_or_ci95": [low, high],
                "age_quad_coef": float,
                "age_quad_se": float,
                "age_quad_z": float,
                "age_quad_p": float,
                "age_quad_ci95": [low, high],
                "age_quad_or": float,
                "age_quad_or_ci95": [low, high]
            },
            ...
        },
        "description": "Brief interpretation of output and how to read it."
      }
    """
    import re
    import numpy as np
    from scipy import stats

    res = model_output  # statsmodels BinaryResultsWrapper
    params = res.params  # pandas Series
    cov = res.cov_params()  # DataFrame
    # Ensure required main terms exist
    if 'age_c' not in params.index or 'age_c2' not in params.index:
        raise ValueError("Model does not contain 'age_c' and 'age_c2' main effects as expected.")

    # Obtain culture levels from the original data if available
    try:
        df = res.model.data.frame
        if 'culture' in df.columns:
            # preserve the order as seen in the data
            levels = list(pd.unique(df['culture']))
        else:
            # fallback to scanning param names
            levels = None
    except Exception:
        levels = None

    # If we couldn't get levels from data, build levels from param names
    if levels is None:
        levels = []
        for name in params.index:
            m = re.search(r"C\(culture\)\[T\.([^\]]+)\]", name)
            if m:
                lvl = m.group(1)
                if lvl not in levels:
                    levels.append(lvl)
    # Determine the reference (omitted) level: the one present in data levels but not as C(culture)[T.<lvl>] in params
    # If levels were gathered only from params, we can't know reference; assume the first in data (if data used)
    ref_level = None
    # Build set of levels appearing as explicit dummy params
    dummy_levels = set()
    for name in params.index:
        m = re.search(r"C\(culture\)\[T\.([^\]]+)\]", name)
        if m:
            dummy_levels.add(m.group(1))
    if levels:
        # find reference as level in levels that is not in dummy_levels
        candidates = [lvl for lvl in levels if lvl not in dummy_levels]
        if len(candidates) == 1:
            ref_level = candidates[0]
        elif len(candidates) > 1:
            # unlikely but choose first
            ref_level = candidates[0]
        else:
            # fallback: if no candidate, take first level (may be risky)
            ref_level = levels[0]
    else:
        raise ValueError("Could not determine culture levels from model or data.")

    # Build maps from level -> interaction parameter name (if exists)
    lin_inter_map = {}   # level -> param name for age_c:C(culture)[T.level]
    quad_inter_map = {}  # level -> param name for age_c2:C(culture)[T.level]
    for name in params.index:
        # find level if present
        m = re.search(r"C\(culture\)\[T\.([^\]]+)\]", name)
        if not m:
            continue
        lvl = m.group(1)
        # check if name also contains age_c or age_c2
        if 'age_c2' in name and 'age_c' not in name:
            quad_inter_map[lvl] = name
        elif 'age_c' in name:
            # could be linear interaction; careful not to match main age_c (which doesn't contain C(culture))
            lin_inter_map[lvl] = name

    results = {}
    # helper to safely get cov entries
    def cov_entry(a, b):
        try:
            return float(cov.loc[a, b])
        except Exception:
            # if either name missing, return 0 (used when interaction absent)
            return 0.0

    for lvl in levels:
        # for reference level, interactions are absent by definition
        inter_lin_name = lin_inter_map.get(lvl, None)
        inter_quad_name = quad_inter_map.get(lvl, None)

        # Combined coefficient = main + interaction (if present). For reference, interactions absent.
        base_lin = float(params['age_c'])
        base_quad = float(params['age_c2'])
        inter_lin_val = float(params[inter_lin_name]) if inter_lin_name in params.index else 0.0
        inter_quad_val = float(params[inter_quad_name]) if inter_quad_name in params.index else 0.0

        comb_lin = base_lin + inter_lin_val
        comb_quad = base_quad + inter_quad_val

        # Variance of sum: Var(a+b) = Var(a) + Var(b) + 2Cov(a,b)
        var_lin = float(cov_entry('age_c', 'age_c'))
        if inter_lin_name:
            var_lin += float(cov_entry(inter_lin_name, inter_lin_name)) + 2.0 * float(cov_entry('age_c', inter_lin_name))
        se_lin = np.sqrt(max(var_lin, 0.0))

        var_quad = float(cov_entry('age_c2', 'age_c2'))
        if inter_quad_name:
            var_quad += float(cov_entry(inter_quad_name, inter_quad_name)) + 2.0 * float(cov_entry('age_c2', inter_quad_name))
        se_quad = np.sqrt(max(var_quad, 0.0))

        # z and p-values
        z_lin = comb_lin / se_lin if se_lin > 0 else np.nan
        p_lin = 2.0 * (1.0 - stats.norm.cdf(abs(z_lin))) if se_lin > 0 else np.nan
        ci_lin_low = comb_lin - 1.96 * se_lin
        ci_lin_high = comb_lin + 1.96 * se_lin

        z_quad = comb_quad / se_quad if se_quad > 0 else np.nan
        p_quad = 2.0 * (1.0 - stats.norm.cdf(abs(z_quad))) if se_quad > 0 else np.nan
        ci_quad_low = comb_quad - 1.96 * se_quad
        ci_quad_high = comb_quad + 1.96 * se_quad

        # Odds ratios and their CIs (exp of coefficients)
        or_lin = np.exp(comb_lin)
        or_lin_ci = [np.exp(ci_lin_low), np.exp(ci_lin_high)]
        or_quad = np.exp(comb_quad)
        or_quad_ci = [np.exp(ci_quad_low), np.exp(ci_quad_high)]

        results[lvl] = {
            "reference": (lvl == ref_level),
            "age_linear_coef": float(comb_lin),
            "age_linear_se": float(se_lin),
            "age_linear_z": float(z_lin) if not np.isnan(z_lin) else None,
            "age_linear_p": float(p_lin) if not np.isnan(p_lin) else None,
            "age_linear_ci95": [float(ci_lin_low), float(ci_lin_high)],
            "age_linear_or": float(or_lin),
            "age_linear_or_ci95": [float(or_lin_ci[0]), float(or_lin_ci[1])],
            "age_quad_coef": float(comb_quad),
            "age_quad_se": float(se_quad),
            "age_quad_z": float(z_quad) if not np.isnan(z_quad) else None,
            "age_quad_p": float(p_quad) if not np.isnan(p_quad) else None,
            "age_quad_ci95": [float(ci_quad_low), float(ci_quad_high)],
            "age_quad_or": float(or_quad),
            "age_quad_or_ci95": [float(or_quad_ci[0]), float(or_quad_ci[1])]
        }

    description = (
        "For each culture level, the object provides the combined linear (age_c) and quadratic (age_c2) "
        "log-odds coefficients (i.e., main effect + any culture-specific interaction), their standard errors, "
        "z-statistics, two-sided p-values, 95% confidence intervals, and the corresponding odds ratios with 95% CIs. "
        "Positive linear coefficients indicate that as centered age increases, the log-odds (and hence odds) of choosing the majority option increase; "
        "negative coefficients indicate a decrease. The quadratic term captures curvature in age-related change (negative quadratic -> concave down; positive -> concave up). "
        "Use the p-values (and CIs) to judge whether age effects differ from zero within each culture. The 'reference' flag marks which culture was the omitted "
        "reference level in the model (its culture-specific interactions are absent by definition)."
    )

    return {"object": results, "description": description}