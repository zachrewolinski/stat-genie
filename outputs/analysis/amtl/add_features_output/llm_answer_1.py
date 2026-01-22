def extract_final_answer(model_output):
    """
    Extracts statistics comparing Homo sapiens (reference) to non-human genera (Pan, Papio, Pongo)
    from the provided model_output (expected to contain an 'odds_ratios' DataFrame or a fitted model).
    
    Returns:
      {
        "object": {
          "per_genus": {
             "Pan": {"coef": ..., "OR": ..., "CI": [low, high], "pvalue": ..., "significant": True/False},
             "Papio": {...},
             "Pongo": {...}
          },
          "humans_higher_amtl": True/False,   # True if all non-human genera have OR < 1 and p < 0.05
          "notes": "reference = Homo sapiens; model adjusted for age_c, prob_male, tooth_class"
        },
        "description": "<text explanation>"
      }
    """
    import numpy as np
    import pandas as pd

    # Terms we expect in the odds_ratios table (Homo sapiens is the reference level)
    genus_terms = {
        "Pan": "C(genus)[T.Pan]",
        "Papio": "C(genus)[T.Papio]",
        "Pongo": "C(genus)[T.Pongo]"
    }

    # Try to get the precomputed odds_ratios table if present
    or_table = None
    if isinstance(model_output, dict) and 'odds_ratios' in model_output:
        or_table = model_output['odds_ratios']
    elif isinstance(model_output, dict) and 'model_fit' in model_output:
        # Build a similar table from the fitted model if necessary
        res = model_output['model_fit']
        params = res.params
        conf = res.conf_int()
        pvals = res.pvalues
        or_table = pd.DataFrame({
            'coef': params,
            'OR': np.exp(params),
            'CI_lower': np.exp(conf[0]),
            'CI_upper': np.exp(conf[1]),
            'pvalue': pvals
        })
    else:
        raise ValueError("model_output must be a dict containing 'odds_ratios' or 'model_fit'.")

    # Ensure index is accessible (if term column exists convert to index)
    if 'term' in or_table.columns:
        or_table = or_table.set_index('term')

    results_per_genus = {}
    all_lower_and_sig = True

    for common_name, term in genus_terms.items():
        if term in or_table.index:
            row = or_table.loc[term]
            coef = float(row['coef'])
            OR = float(row['OR'])
            ci_low = float(row['CI_lower'])
            ci_high = float(row['CI_upper'])
            p = float(row['pvalue'])
            significant = (p < 0.05)
            results_per_genus[common_name] = {
                'term': term,
                'coef': coef,
                'OR': OR,
                'CI': [ci_low, ci_high],
                'pvalue': p,
                'significant': significant
            }
            if not (OR < 1 and significant):
                all_lower_and_sig = False
        else:
            # Term missing: record as NaNs and mark overall as inconclusive
            results_per_genus[common_name] = {
                'term': term,
                'coef': np.nan,
                'OR': np.nan,
                'CI': [np.nan, np.nan],
                'pvalue': np.nan,
                'significant': False
            }
            all_lower_and_sig = False

    # Build overall conclusion
    if all_lower_and_sig:
        conclusion_text = (
            "Yes — after adjusting for age, sex (prob_male), and tooth class, Homo sapiens (the model reference) "
            "have significantly higher AMTL odds than each non-human genus listed. "
            "Each non-human genus shows OR < 1 (lower odds of AMTL relative to Homo) with p < 0.05."
        )
    else:
        conclusion_text = (
            "No / Inconclusive — not all comparisons show significantly lower odds in non-human genera, "
            "so we cannot conclude that Homo sapiens have higher AMTL in all comparisons after adjustment."
        )

    output_object = {
        'per_genus': results_per_genus,
        'humans_higher_amtl': bool(all_lower_and_sig),
        'notes': "Reference category is Homo sapiens; model controls: age_c, prob_male, C(tooth_class)."
    }

    description_lines = [
        "Extracted coefficients, odds ratios (OR), 95% CIs, and p-values for comparisons of Pan, Papio, and Pongo",
        "to the reference genus (Homo sapiens).",
        f"Conclusion: {conclusion_text}",
        "Detailed per-genus results are provided in the 'object' field under 'per_genus'."
    ]
    description = " ".join(description_lines)

    return {"object": output_object, "description": description}