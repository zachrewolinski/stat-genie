def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of being a modern human (IsHuman) on AMTL
    from the fitted GLM results returned by the modeling function.

    Returns:
      {
        "object": { ... numeric results and objects ... },
        "description": "text explanation of results & conclusion"
      }

    The function expects model_output to be a dict with keys:
      - 'model_is_human': statsmodels GLMResultsWrapper (binary IsHuman model)
      - 'model_by_genus':  statsmodels GLMResultsWrapper (genus-level model)
    """
    import numpy as np
    import pandas as pd

    # Unpack models
    res1 = model_output.get('model_is_human', None)
    res2 = model_output.get('model_by_genus', None)
    if res1 is None:
        raise ValueError("model_output must contain 'model_is_human'")
    if res2 is None:
        raise ValueError("model_output must contain 'model_by_genus'")

    out = {}

    # --- Extract IsHuman stats from model_is_human ---
    params1 = res1.params
    pvals1 = res1.pvalues
    ci1 = res1.conf_int()

    if 'IsHuman' not in params1.index:
        raise ValueError("The fitted model 'model_is_human' does not contain an 'IsHuman' coefficient.")

    coef = float(params1['IsHuman'])
    pval = float(pvals1['IsHuman'])
    ci_low, ci_high = map(float, ci1.loc['IsHuman'])
    # Coefficient is on log-odds scale (GLM Binomial uses logit link by default).
    or_val = float(np.exp(coef))
    or_ci_low, or_ci_high = map(float, np.exp([ci_low, ci_high]))

    # Predicted probabilities at mean covariate values for IsHuman = 0 and 1
    exog_names = list(res1.model.exog_names)
    # compute column means from model.exog (columns aligned with exog_names)
    exog_means = pd.Series(res1.model.exog.mean(axis=0), index=exog_names)

    # Build two exog rows: IsHuman = 0 and IsHuman = 1; keep other covariates at their means.
    exog_base = exog_means.copy()
    exog_zero = exog_base.copy()
    exog_zero['IsHuman'] = 0.0
    exog_one = exog_base.copy()
    exog_one['IsHuman'] = 1.0

    # Predict probabilities
    # statsmodels accepts arrays/dataframes with columns matching exog_names
    exog_df = pd.DataFrame([exog_zero.values, exog_one.values], columns=exog_names)
    probs = res1.predict(exog_df)  # returns probabilities because model is binomial-logit
    prob_nonhuman = float(probs.iloc[0])
    prob_human = float(probs.iloc[1])
    prob_diff = prob_human - prob_nonhuman

    # Decision rule: positive coefficient, statistically significant (p < 0.05),
    # and predicted probability for humans > non-humans -> conclude "higher".
    significant = (pval < 0.05)
    direction_positive = (coef > 0)
    if direction_positive and significant and (prob_human > prob_nonhuman):
        conclusion = "Yes: modern humans have higher AMTL rates (statistically significant)."
    elif direction_positive and (not significant):
        conclusion = "No strong evidence: coefficient positive but not statistically significant."
    elif (not direction_positive) and significant:
        conclusion = "No: modern humans have lower AMTL rates (statistically significant)."
    else:
        conclusion = "No strong evidence for higher AMTL in modern humans (no statistically significant positive effect)."

    # Prepare object summary for IsHuman model
    ishuman_summary = {
        'coef_log_odds': coef,
        'p_value': pval,
        'conf_int_95_log_odds': [ci_low, ci_high],
        'odds_ratio': or_val,
        'odds_ratio_conf_int_95': [or_ci_low, or_ci_high],
        'predicted_prob_nonhuman_at_means': prob_nonhuman,
        'predicted_prob_human_at_means': prob_human,
        'predicted_prob_difference (human - nonhuman)': prob_diff,
        'significant_at_0.05': bool(significant),
        'conclusion_basic': conclusion
    }

    out['IsHuman_model'] = ishuman_summary

    # --- Extract genus-level results from model_by_genus ---
    # We'll compute predicted probability (at mean of other covariates) for each genus
    # using the genus dummy columns in res2.
    exog_names2 = list(res2.model.exog_names)
    exog_means2 = pd.Series(res2.model.exog.mean(axis=0), index=exog_names2)

    genus_cols = [c for c in exog_names2 if c.startswith('Genus_')]
    genus_names = [c[len('Genus_'):] for c in genus_cols]

    genus_results = {}
    # If no genus dummies exist (e.g., only one genus present), handle gracefully.
    if len(genus_cols) == 0:
        genus_results['note'] = "No genus dummy columns found in genus-level model (maybe only one genus present)."
    else:
        # Identify reference genus: it's the genus not represented among dummies.
        # We can't always recover the reference name directly from model, but we can infer:
        # All genera present = dummies + (reference). We attempt to get the set of genus levels
        # by inspecting parameter index for any names like 'Genus_<name>'; reference is the missing one.
        # If 'Genus_Homo sapiens' present, Homo sapiens is not reference. If absent, Homo sapiens may be reference.
        # We'll compute predicted probs for each genus we can represent (those in dummy columns)
        # and also for the reference genus by setting all genus dummies to 0.
        # Build base exog with means
        base = exog_means2.copy()

        # Probability for the reference genus (all genus dummies = 0)
        ref_exog = base.copy()
        for c in genus_cols:
            ref_exog[c] = 0.0
        ref_df = pd.DataFrame([ref_exog.values], columns=exog_names2)
        prob_ref = float(res2.predict(ref_df).iloc[0])
        genus_results['reference_genus_prob_at_means'] = prob_ref

        # Record predicted probability for each dummy-coded genus
        for col, gname in zip(genus_cols, genus_names):
            ex = base.copy()
            # set this genus dummy to 1, others to 0
            for c in genus_cols:
                ex[c] = 1.0 if c == col else 0.0
            ex_df = pd.DataFrame([ex.values], columns=exog_names2)
            prob_g = float(res2.predict(ex_df).iloc[0])
            # Extract coefficient and p-value for this genus dummy (if present)
            coef_g = float(res2.params[col]) if col in res2.params.index else None
            p_g = float(res2.pvalues[col]) if col in res2.pvalues.index else None
            ci_g = None
            if col in res2.params.index:
                ci_row = res2.conf_int().loc[col]
                ci_g = [float(ci_row[0]), float(ci_row[1])]
            genus_results[gname] = {
                'coef_log_odds_vs_reference': coef_g,
                'p_value': p_g,
                'conf_int_95_log_odds': ci_g,
                'predicted_prob_at_means': prob_g
            }

        # Try to determine if Homo sapiens (or 'Homo sapiens' variant) is present among genus dummies
        # We check for substring 'Homo' or exact 'Homo sapiens'
        homo_keys = [g for g in genus_results.keys() if isinstance(g, str) and ('Homo' in g or 'sapiens' in g)]
        if len(homo_keys) == 0:
            # Homo may be the reference (i.e., not in dummy columns)
            homo_status = "Homo sapiens may be the reference category (no 'Genus_Homo...' dummy present). Predicted probability for reference is reported under 'reference_genus_prob_at_means'."
        else:
            # Report the Homo entry(s)
            homo_status = f"Homo entries found among genus dummies: {homo_keys}. Their predicted probs and coefficients are included."
        genus_results['homo_presence_note'] = homo_status

    out['Genus_model'] = genus_results

    # Build a concise description string
    desc_lines = []
    desc_lines.append("Primary test (IsHuman): coefficient, p-value, odds ratio, predicted probabilities at mean covariates.")
    desc_lines.append(f"IsHuman coef (log-odds) = {coef:.4f}, p = {pval:.4g}, 95% CI (log-odds) = [{ci_low:.4f}, {ci_high:.4f}].")
    desc_lines.append(f"Odds ratio = {or_val:.3f}, 95% CI = [{or_ci_low:.3f}, {or_ci_high:.3f}].")
    desc_lines.append(f"Predicted AMTL probability at mean covariates: non-human = {prob_nonhuman:.3f}, human = {prob_human:.3f}, difference = {prob_diff:.3f}.")
    desc_lines.append(f"Interpretation: {conclusion}")
    desc_lines.append("")  # blank line
    desc_lines.append("Genus-level model: predicted probabilities at mean covariates are provided for each genus dummy and for the reference genus (all genus dummies = 0). See 'object' -> 'Genus_model' for numbers and coefficients.")
    description = "\n".join(desc_lines)

    return {"object": out, "description": description}