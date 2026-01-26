def extract_final_answer(model_output):
    """
    Extracts statistics describing how age (Age_z) predicts choosing the majority option,
    using the fitted binary logistic model 'majority_logit_result' from the provided
    model_output dict. Computes overall Age_z effect and culture-specific simple slopes
    (Age_z main effect + Age_z:C(culture) interaction when present), with standard errors,
    z-scores, two-sided p-values, and 95% Wald confidence intervals.

    Returns:
      {
        "object": pandas.DataFrame with rows for each culture (including the omitted/reference),
        "description": str describing what the table contains and how to interpret it
      }
    """
    import re
    import math
    import pandas as pd

    # Get the binary logistic result that modeled MajorityChoice (majority vs. not)
    if 'majority_logit_result' not in model_output:
        raise ValueError("model_output must contain 'majority_logit_result' key")
    res = model_output['majority_logit_result']

    # Parameters and covariance (DataFrame)
    params = res.params.copy()
    cov = res.cov_params().copy()

    # Find the exact name for the Age_z main effect (robust to slight naming variations)
    age_name = None
    for n in params.index:
        if n == 'Age_z' or n.endswith('.Age_z') or n == 'Age_z' :
            age_name = n
            break
    if age_name is None:
        # fallback: any name that contains Age_z
        age_candidates = [n for n in params.index if 'Age_z' in n]
        if not age_candidates:
            raise KeyError("Could not find an 'Age_z' parameter in model params")
        age_name = age_candidates[0]

    beta_age = float(params[age_name])
    var_age = float(cov.loc[age_name, age_name])
    se_age = math.sqrt(var_age)
    z_age = beta_age / se_age if se_age > 0 else float('nan')
    # two-sided p-value using erf for normal cdf
    cdf = 0.5 * (1 + math.erf(abs(z_age) / math.sqrt(2)))
    p_age = 2 * (1 - cdf)
    ci_lower = beta_age - 1.96 * se_age
    ci_upper = beta_age + 1.96 * se_age

    # Identify interaction terms that involve Age_z and culture
    # typical names: 'Age_z:C(culture)[T.SITE]' or 'C(culture)[T.SITE]:Age_z'
    inter_names = [n for n in params.index if ('Age_z' in n) and ('C(culture)' in n or 'C(culture)' in n)]
    # If not found by that pattern, search for any param names that include both 'Age_z' and 'culture' (robust)
    if not inter_names:
        inter_names = [n for n in params.index if ('Age_z' in n) and ('culture' in n)]
    # Parse culture levels from interaction names
    culture_levels = []
    inter_map = {}  # map level -> (name, coef)
    for n in inter_names:
        # try extracting with regex pattern for Patsy naming: C(culture)[T.<level>]
        m = re.search(r"C\(culture\)\[T\.?([^\]]+)\]", n)
        if m:
            lvl = m.group(1)
        else:
            # fallback: take the part after last ':' or '.' as level descriptor
            parts = re.split(r'[:\.]', n)
            lvl = parts[-1]
        culture_levels.append(lvl)
        inter_map[lvl] = (n, float(params[n]))

    # Determine the "reference" (omitted) culture: it's the one with no interaction term.
    # We cannot recover its original label from the model output (no data provided),
    # so we label it 'reference (omitted)' and report its slope = Age_z main effect.
    rows = []
    rows.append({
        'culture': 'reference (omitted)',
        'slope_logodds': beta_age,
        'se': se_age,
        'z': z_age,
        'p': p_age,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'coef_parts': f"Age_z"
    })

    # For each explicit culture level with an interaction, compute simple slope and its se/p/ci
    for lvl in sorted(culture_levels):
        inter_name, beta_int = inter_map[lvl]
        slope = beta_age + beta_int
        # variance = Var(age) + Var(int) + 2 Cov(age,int)
        if (age_name in cov.index) and (inter_name in cov.index):
            cov_ai = float(cov.loc[age_name, inter_name])
            var_int = float(cov.loc[inter_name, inter_name])
            var_slope = var_age + var_int + 2.0 * cov_ai
            se_slope = math.sqrt(var_slope) if var_slope > 0 else float('nan')
            z_slope = slope / se_slope if se_slope > 0 else float('nan')
            cdf_s = 0.5 * (1 + math.erf(abs(z_slope) / math.sqrt(2)))
            p_slope = 2 * (1 - cdf_s)
            ci_l = slope - 1.96 * se_slope
            ci_u = slope + 1.96 * se_slope
        else:
            # if covariance entries missing for some reason, set NaNs
            se_slope = float('nan')
            z_slope = float('nan')
            p_slope = float('nan')
            ci_l = float('nan')
            ci_u = float('nan')

        rows.append({
            'culture': lvl,
            'slope_logodds': slope,
            'se': se_slope,
            'z': z_slope,
            'p': p_slope,
            'ci_lower': ci_l,
            'ci_upper': ci_u,
            'coef_parts': f"Age_z + {inter_name}"
        })

    result_df = pd.DataFrame(rows).set_index('culture')

    description_lines = [
        "This table reports how the standardized age variable (Age_z) predicts the log-odds",
        "of choosing the majority option (MajorityChoice model: majority vs not).",
        "- 'reference (omitted)' is the baseline culture (the C(culture) reference level omitted by the encoding).",
        "- For the omitted reference culture the slope is the Age_z main effect; for each listed culture",
        "  the simple slope = Age_z main effect + Age_z:C(culture)[T.<level>] interaction coefficient.",
        "- Columns: slope_logodds (log-odds change per 1 SD increase in age), se, z, two-sided p-value, and 95% Wald CI.",
        "Interpretation: positive slope_logodds => older children are more likely to choose the majority (vs not) in that culture;",
        "negative => older children are less likely. Use the p-value/CI to judge statistical support."
    ]
    description = " ".join(description_lines)

    return {"object": result_df, "description": description}