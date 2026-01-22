def extract_final_answer(model_output):
    """
    Extract and interpret genus contrasts from the model output produced by the provided modeling function.

    Expects model_output to be a dict containing a pandas DataFrame under key 'contrasts' with columns at least:
      ['contrast', 'log_odds_ratio', 'OR', 'OR_95CI_low', 'OR_95CI_high', 'z', 'p_two_tailed'].

    Returns a dictionary with keys:
      - "object": a serializable dict summarizing each contrast (OR, 95% CI, p-value, significance, direction)
                  and an overall boolean 'humans_higher_vs_all_nonhuman'.
      - "description": a brief interpretation of what those statistics mean in the context of the task.
    """
    import pandas as pd
    import math

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the model() function.")

    if 'contrasts' not in model_output:
        raise ValueError("model_output does not contain 'contrasts'. Expected a DataFrame under model_output['contrasts'].")

    contrasts_df = model_output['contrasts']

    # If contrasts come as a non-DataFrame (e.g., list of dicts), coerce to DataFrame
    if not isinstance(contrasts_df, pd.DataFrame):
        contrasts_df = pd.DataFrame(contrasts_df)

    required_cols = {'contrast', 'log_odds_ratio', 'OR', 'OR_95CI_low', 'OR_95CI_high', 'z', 'p_two_tailed'}
    if not required_cols.issubset(set(contrasts_df.columns)):
        raise ValueError(f"contrasts DataFrame missing required columns. Found columns: {list(contrasts_df.columns)}")

    summary_list = []
    all_higher_and_significant = True

    for _, row in contrasts_df.iterrows():
        # Safely convert to Python floats
        try:
            or_est = float(row['OR'])
        except Exception:
            or_est = None
        try:
            ci_lo = float(row['OR_95CI_low'])
            ci_hi = float(row['OR_95CI_high'])
        except Exception:
            ci_lo = ci_hi = None
        try:
            pval = float(row['p_two_tailed'])
        except Exception:
            pval = None

        # Determine direction and significance (alpha=0.05)
        if or_est is None:
            direction = 'unknown'
        else:
            if math.isclose(or_est, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                direction = 'no_difference'
            elif or_est > 1.0:
                direction = 'higher_in_Homo_sapiens'
            else:
                direction = 'lower_in_Homo_sapiens'

        significant = (pval is not None) and (pval < 0.05)

        if not (direction == 'higher_in_Homo_sapiens' and significant):
            all_higher_and_significant = False

        summary_list.append({
            'contrast': str(row['contrast']),
            'log_odds_ratio': float(row['log_odds_ratio']) if pd.notna(row['log_odds_ratio']) else None,
            'OR': or_est,
            'OR_95CI_low': ci_lo,
            'OR_95CI_high': ci_hi,
            'z': float(row['z']) if pd.notna(row['z']) else None,
            'p_two_tailed': pval,
            'significant_at_0.05': bool(significant),
            'direction': direction
        })

    # Build final object and textual interpretation
    result_object = {
        'contrasts': summary_list,
        'humans_higher_vs_all_nonhuman': bool(all_higher_and_significant)
    }

    if all_higher_and_significant:
        description = (
            "After adjusting for age, sex (prob_male), and tooth class in a binomial GEE (clustered by specimen), "
            "Homo sapiens shows significantly higher odds of antemortem tooth loss (AMTL) compared to each non-human "
            "genus present in the model. All pairwise contrasts (Homo sapiens vs. each non-human genus) have OR > 1 "
            "and p < 0.05. See 'object[\"contrasts\"]' for ORs, 95% CIs, z-statistics, and p-values."
        )
    else:
        description = (
            "The contrasts comparing Homo sapiens to non-human genera are provided. The flag "
            "'humans_higher_vs_all_nonhuman' is False, meaning not all contrasts show statistically significant "
            "higher AMTL in Homo sapiens (OR > 1 and p < 0.05). Inspect 'object[\"contrasts\"]' for per-contrast "
            "ORs, 95% CIs, z-statistics, and p-values to see which comparisons are significant."
        )

    return {"object": result_object, "description": description}