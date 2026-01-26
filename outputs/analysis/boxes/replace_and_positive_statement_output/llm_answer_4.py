def extract_final_answer(model_output):
    """
    Extract statistics relevant to age effects (age_c) and Age x Culture interactions
    from the fitted models returned by the model() function.

    Input:
      model_output: dict with keys
        - 'model_social_reliance' : statsmodels BinaryResultsWrapper (logit) or None
        - 'model_majority_preference': statsmodels BinaryResultsWrapper (logit) or None
        - 'model_multinomial_choice': statsmodels MNLogitResultsWrapper or None

    Returns:
      dict with keys:
        - "object": a nested dict containing extracted coefficients, standard errors,
                    p-values, 95% CIs, and odds ratios for age_c and any age_c:C(culture) terms
                    for each model (and for each multinomial outcome).
        - "description": textual explanation of the extracted numbers and how to interpret them.
    """
    import numpy as np
    import pandas as pd

    out = {}
    z = 1.96  # approx for 95% CI

    def extract_from_binary(res):
        """Extract age_c and age_c:C(culture) terms from a binary (Logit) result."""
        if res is None:
            return None
        res_dict = {}
        # params, bse, pvalues are usually pandas Series indexed by param names
        params = getattr(res, 'params', None)
        bse = getattr(res, 'bse', None)
        pvals = getattr(res, 'pvalues', None)

        if params is None:
            return None

        # find all parameter names that include 'age_c' (main or interactions)
        matched = [name for name in params.index if 'age_c' in name]
        for name in matched:
            coef = float(params.loc[name])
            se = float(bse.loc[name]) if bse is not None else None
            p = float(pvals.loc[name]) if pvals is not None else None
            if se is not None:
                ci_low = coef - z * se
                ci_high = coef + z * se
            else:
                ci_low = ci_high = None
            res_dict[name] = {
                'coef': coef,
                'se': se,
                'pvalue': p,
                '95ci': (ci_low, ci_high),
                'odds_ratio': float(np.exp(coef)),
                'odds_ratio_95ci': (float(np.exp(ci_low)) if ci_low is not None else None,
                                    float(np.exp(ci_high)) if ci_high is not None else None)
            }
        return res_dict

    def extract_from_multinomial(res):
        """Extract age_c and age_c:C(culture) terms from MNLogit result.
           For MNLogit, res.params is typically a DataFrame with columns per outcome
           (each column corresponds to log-odds for that outcome vs base)."""
        if res is None:
            return None
        params = getattr(res, 'params', None)
        bse = getattr(res, 'bse', None)
        pvals = getattr(res, 'pvalues', None)

        if params is None:
            return None

        res_dict = {}
        # If params is a Series (unlikely), convert to DataFrame with single column
        if isinstance(params, pd.Series):
            params = params.to_frame(name='outcome_0')

        # Expect params to be DataFrame with index=param names and columns=outcome labels (0..k-1)
        # For each outcome (comparison) extract rows where param name contains 'age_c'
        for outcome in params.columns:
            out_series = params[outcome]
            res_dict[outcome] = {}
            for param in out_series.index:
                if 'age_c' in param:
                    coef = float(out_series.loc[param])
                    # get se and pval from corresponding DataFrames (same layout)
                    se = float(bse.loc[param, outcome]) if (bse is not None and param in bse.index) else None
                    p = (float(pvals.loc[param, outcome])
                         if (pvals is not None and param in pvals.index) else None)
                    if se is not None:
                        ci_low = coef - z * se
                        ci_high = coef + z * se
                    else:
                        ci_low = ci_high = None
                    res_dict[outcome][param] = {
                        'coef': coef,
                        'se': se,
                        'pvalue': p,
                        '95ci': (ci_low, ci_high),
                        'odds_ratio': float(np.exp(coef)),
                        'odds_ratio_95ci': (float(np.exp(ci_low)) if ci_low is not None else None,
                                            float(np.exp(ci_high)) if ci_high is not None else None)
                    }
        return res_dict

    # Extract for model 1: social_choice ~ age_c * C(culture) + ...
    out['social_reliance'] = extract_from_binary(model_output.get('model_social_reliance'))

    # Extract for model 2: majority_choice among demonstrated choices
    out['majority_preference_among_demonstrated'] = extract_from_binary(model_output.get('model_majority_preference'))

    # Extract for model 3: multinomial over y=1,2,3 (endog was y-1)
    out['multinomial_choice'] = extract_from_multinomial(model_output.get('model_multinomial_choice'))

    # Provide a human-readable description that explains how to interpret the entries
    description_lines = [
        "Returned object summary:",
        " - For each model we extract coefficients, standard errors, p-values, 95% CIs, and odds ratios",
        "   for any parameter whose name contains 'age_c' (this includes the main age_c term and any",
        "   Age x Culture interaction terms which are typically named like 'age_c:C(culture)[T.X]')",
        "",
        "How to interpret entries:",
        " - For the binary logit models (social_reliance; majority_preference_among_demonstrated):",
        "     * The coefficient 'age_c' is the estimated log-odds change per unit (mean-centered) age in the reference culture.",
        "     * An interaction term 'age_c:C(culture)[T.X]' is the additional log-odds change per unit age for culture X.",
        "     * To get the age effect in culture X, add the main 'age_c' coefficient to the interaction coefficient for culture X.",
        "     * The odds_ratio is exp(coef). For example, odds_ratio > 1 for age_c means increasing age increases odds of the outcome.",
        "",
        " - For the multinomial model (multinomial_choice):",
        "     * The MNLogit reports coefficients for each non-reference outcome versus the reference outcome.",
        "     * In the original modeling, endog = y - 1, so:",
        "         - outcome column '0' (if present) corresponds to original y=2 (majority) vs y=1 (undemonstrated),",
        "         - outcome column '1' corresponds to original y=3 (minority) vs y=1 (undemonstrated).",
        "     * For each outcome we extract age_c and age_c:C(culture)[T.X] coefficients with the same interpretation",
        "       (add main + interaction to get culture-specific age effect).",
        "",
        "Notes and next steps you may want to run with these extracted numbers:",
        " - To obtain the age effect in each culture explicitly, compute (age_c coef) + (age_c:C(culture)[T.X] coef) for each culture X.",
        " - You can similarly combine the standard errors (using the covariance matrix) to get correct SEs for the sums if you need hypothesis tests",
        "   on the per-culture age slopes. This function does not perform covariance-based combination; it reports per-parameter SEs.",
        " - If you want per-culture predicted probabilities across ages, use the model's predict() method and vary age_c while setting culture dummies accordingly.",
        "",
        "Returned structure:",
        " - object: dict with keys 'social_reliance', 'majority_preference_among_demonstrated', 'multinomial_choice'.",
        "   Each contains extracted parameter info as described above (or None if that model wasn't fit).",
        ""
    ]

    description = "\n".join(description_lines)

    return {"object": out, "description": description}