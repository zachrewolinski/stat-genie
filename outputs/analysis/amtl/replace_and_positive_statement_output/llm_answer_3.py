def extract_final_answer(model_output):
    """
    Extract statistics for genus comparisons vs. the reference level (Homo sapiens)
    from the provided model_output and give a brief interpretation.

    Returns a dictionary with:
      - "object": dict mapping genus -> {coef, OR, CI_lower, CI_upper, pvalue, significant, direction}
      - "description": short conclusion about whether Homo sapiens show higher AMTL
    """
    import numpy as np
    import pandas as pd

    # Attempt to get a DataFrame with the genus rows
    genus_df = None

    # Preferred: model_output contains 'genus_or' (a DataFrame of genus rows)
    if isinstance(model_output, dict) and 'genus_or' in model_output:
        genus_df = model_output['genus_or']
    # Fallback: use 'or_table' and filter rows that start with 'C(genus)'
    elif isinstance(model_output, dict) and 'or_table' in model_output:
        df = model_output['or_table']
        genus_rows = [r for r in df.index if str(r).startswith('C(genus)')]
        genus_df = df.loc[genus_rows].copy()
    # Final fallback: try to construct from a results object (statsmodels results)
    elif isinstance(model_output, dict) and 'results' in model_output:
        res = model_output['results']
        params = res.params
        conf = res.conf_int()
        pvals = res.pvalues
        genus_rows = [r for r in params.index if str(r).startswith('C(genus)')]
        if genus_rows:
            records = []
            for r in genus_rows:
                coef = float(params[r])
                ci_low = float(conf.loc[r, 0])
                ci_high = float(conf.loc[r, 1])
                p = float(pvals[r])
                records.append((r, coef, np.exp(coef), np.exp(ci_low), np.exp(ci_high), p))
            genus_df = pd.DataFrame.from_records(records,
                                                 columns=['row', 'coef', 'OR', 'CI_lower', 'CI_upper', 'pvalue']).set_index('row')

    if genus_df is None:
        return {
            "object": None,
            "description": "Could not find genus-level coefficients in model_output. Expected keys like 'genus_or', 'or_table', or 'results'."
        }

    # Ensure columns exist and are numeric
    # Rename columns if they are named differently (some outputs used 'coef','OR','CI_lower','CI_upper','pvalue')
    expected_cols = ['coef', 'OR', 'CI_lower', 'CI_upper', 'pvalue']
    # If genus_df has numeric columns in different names, try to coerce
    df = genus_df.copy()
    # If the DataFrame is a plain Series (one row), convert to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    # If column names match expected, use them; otherwise try to infer from position
    if not all(c in df.columns for c in expected_cols):
        # Try positional mapping
        cols = list(df.columns)
        # We assume order: coef, OR, CI_lower, CI_upper, pvalue (as in the example)
        if len(cols) >= 5:
            df = df.rename(columns={cols[0]: 'coef', cols[1]: 'OR', cols[2]: 'CI_lower', cols[3]: 'CI_upper', cols[4]: 'pvalue'})
        else:
            # As a last resort, compute OR and CIs from coef and conf_int if available in model_output['results']
            pass

    # Build result dict per genus
    genus_stats = {}
    for row_label, row in df.iterrows():
        # Extract genus name from label like "C(genus)[T.Pan]" -> "Pan"
        lab = str(row_label)
        if '[' in lab and lab.endswith(']'):
            inside = lab.split('[', 1)[1].rstrip(']')
            # inside e.g., "T.Pan" or "Pan"
            if inside.startswith('T.'):
                genus_name = inside[2:]
            else:
                genus_name = inside
        else:
            # fallback: take last dot-separated token
            genus_name = lab.split('.')[-1]

        try:
            coef = float(row['coef'])
            OR = float(row['OR'])
            CI_lower = float(row['CI_lower'])
            CI_upper = float(row['CI_upper'])
            pval = float(row['pvalue'])
        except Exception:
            # If conversion fails, skip this row
            continue

        significant = (pval < 0.05)
        # Direction relative to Homo sapiens (reference):
        # - If coef < 0 and significant -> genus has lower odds than Homo (so Homo higher)
        # - If coef > 0 and significant -> genus has higher odds than Homo (so Homo lower)
        if significant and coef < 0:
            direction = 'Homo sapiens higher (genus lower; significant)'
        elif significant and coef > 0:
            direction = 'Homo sapiens lower (genus higher; significant)'
        else:
            # Not significant
            # Note: coef sign may indicate a non-significant trend
            if coef < 0:
                direction = 'no significant difference (trend: Homo sapiens higher)'
            elif coef > 0:
                direction = 'no significant difference (trend: Homo sapiens lower)'
            else:
                direction = 'no significant difference (no effect)'

        genus_stats[genus_name] = {
            'coef': coef,
            'OR': OR,
            'CI_lower': CI_lower,
            'CI_upper': CI_upper,
            'pvalue': pval,
            'significant': bool(significant),
            'direction': direction
        }

    # Formulate concise conclusion about whether Homo sapiens have higher AMTL
    # For Homo to be higher than all listed genera we'd need significant negative coef for each genus.
    all_higher = all((v['significant'] and v['coef'] < 0) for v in genus_stats.values())
    any_significant_diff = any(v['significant'] for v in genus_stats.values())

    if all_higher:
        conclusion_text = ("Yes: after adjusting for age, sex, and tooth class, Homo sapiens show "
                           "statistically significantly higher AMTL than the compared genera "
                           "(all genus coefficients are significantly negative).")
    elif not any_significant_diff:
        # Check CIs include 1
        cis_include_one = all((v['CI_lower'] <= 1.0 <= v['CI_upper']) for v in genus_stats.values())
        conclusion_text = ("No: there is no evidence that Homo sapiens have higher AMTL compared to "
                           "Pan, Pongo, or Papio after adjusting for age, sex, and tooth class. "
                           "Genus odds ratios are near 1 and none of the genus comparisons are statistically significant "
                           "(p > 0.05); their 95% CIs include 1.)")
        if not cis_include_one:
            conclusion_text += " Note: at least one CI does not include 1 despite p>0.05 (check model output)."
    else:
        # Mixed significant differences (some genera higher/lower)
        parts = []
        for g, v in genus_stats.items():
            if v['significant']:
                parts.append(f"{g}: {'lower' if v['coef'] < 0 else 'higher'} than Homo (p={v['pvalue']:.3g}, OR={v['OR']:.3g})")
        conclusion_text = ("Mixed results: some non-human genera differ significantly from Homo sapiens: "
                           + "; ".join(parts))

    return {
        "object": genus_stats,
        "description": conclusion_text
    }