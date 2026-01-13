def extract_final_answer(model_output):
    """
    Extracts and interprets the Reader View effects (main effect and interaction
    with dyslexia) from a fitted statsmodels OLS results object.

    Returns a dict with:
      - "object": a dict of numeric results (coefficients, SEs, t, p, 95% CIs,
                  percent change on the original speed scale) for:
            * non-dyslexic readers (main effect of reader_view)
            * dyslexic readers (combined effect: reader_view + interaction)
            * the interaction term itself
            * a copy of the model summary text for reference
      - "description": a concise plain-English interpretation and a final
                       yes/no/inconclusive statement on whether Reader View
                       improves reading speed for individuals with dyslexia.
    """
    import numpy as np

    results = model_output  # statsmodels RegressionResultsWrapper

    # Parameter names
    params = results.params
    param_names = list(params.index)

    # Find names for main effect and interaction robustly
    # main effect should be exactly 'reader_view'
    if 'reader_view' not in param_names:
        raise KeyError("Expected parameter name 'reader_view' not found in model parameters. Found: {}".format(param_names))

    # interaction might be 'reader_view:is_dyslexic' or 'is_dyslexic:reader_view'
    interaction_name = None
    for n in param_names:
        if ':' in n and 'reader_view' in n and 'is_dyslexic' in n:
            interaction_name = n
            break
    if interaction_name is None:
        raise KeyError("Interaction term between reader_view and is_dyslexic not found in model parameters. Found params: {}".format(param_names))

    # Indices
    idx_main = param_names.index('reader_view')
    idx_inter = param_names.index(interaction_name)

    # Helper: use results.t_test with contrast vector to get combined estimates, SE, t, p, CI
    k = len(param_names)
    # Main effect contrast (reader_view alone)
    contrast_main = np.zeros((k,))
    contrast_main[idx_main] = 1.0
    test_main = results.t_test(contrast_main)

    est_main = float(np.asarray(test_main.effect).reshape(-1)[0])
    se_main = float(np.asarray(test_main.sd).reshape(-1)[0])
    t_main = float(np.asarray(test_main.tvalue).reshape(-1)[0])
    p_main = float(np.asarray(test_main.pvalue).reshape(-1)[0])
    ci_main_raw = test_main.conf_int(alpha=0.05)
    ci_main_flat = np.asarray(ci_main_raw).reshape(-1)
    ci_main = (float(ci_main_flat[0]), float(ci_main_flat[1]))

    # Combined effect for dyslexic readers: reader_view + interaction
    contrast_dys = np.zeros((k,))
    contrast_dys[idx_main] = 1.0
    contrast_dys[idx_inter] = 1.0
    test_dys = results.t_test(contrast_dys)

    est_dys = float(np.asarray(test_dys.effect).reshape(-1)[0])
    se_dys = float(np.asarray(test_dys.sd).reshape(-1)[0])
    t_dys = float(np.asarray(test_dys.tvalue).reshape(-1)[0])
    p_dys = float(np.asarray(test_dys.pvalue).reshape(-1)[0])
    ci_dys_raw = test_dys.conf_int(alpha=0.05)
    ci_dys_flat = np.asarray(ci_dys_raw).reshape(-1)
    ci_dys = (float(ci_dys_flat[0]), float(ci_dys_flat[1]))

    # Interaction term alone (reader_view:is_dyslexic)
    est_inter = float(params[interaction_name])
    # Standard error from covariance matrix
    cov = results.cov_params()
    se_inter = float(np.sqrt(cov.loc[interaction_name, interaction_name]))
    t_inter = float(results.tvalues[interaction_name])
    p_inter = float(results.pvalues[interaction_name])
    # 95% CI for interaction
    ci_inter_low = est_inter - 1.96 * se_inter
    ci_inter_high = est_inter + 1.96 * se_inter
    ci_inter = (ci_inter_low, ci_inter_high)

    # Convert log-scale coefficients to multiplicative percent-change in speed:
    pct_main = (np.exp(est_main) - 1.0) * 100.0
    pct_main_ci = ((np.exp(ci_main[0]) - 1.0) * 100.0, (np.exp(ci_main[1]) - 1.0) * 100.0)

    pct_dys = (np.exp(est_dys) - 1.0) * 100.0
    pct_dys_ci = ((np.exp(ci_dys[0]) - 1.0) * 100.0, (np.exp(ci_dys[1]) - 1.0) * 100.0)

    # Simple decision rule for the task question:
    # We consider "improves reading speed for individuals with dyslexia" to be:
    #   a positive estimated combined effect (est_dys > 0) AND statistically significant (p_dys < 0.05).
    if (est_dys > 0) and (p_dys < 0.05):
        conclusion = "Yes"
        conclusion_text = ("Reader View appears to improve reading speed for readers with dyslexia: estimated change = "
                           f"{pct_dys:.2f}% (95% CI {pct_dys_ci[0]:.2f}% to {pct_dys_ci[1]:.2f}%), p = {p_dys:.3g}.")
    elif (est_dys < 0) and (p_dys < 0.05):
        conclusion = "Yes (but negative)"
        conclusion_text = ("Reader View appears to significantly decrease reading speed for readers with dyslexia: estimated change = "
                           f"{pct_dys:.2f}% (95% CI {pct_dys_ci[0]:.2f}% to {pct_dys_ci[1]:.2f}%), p = {p_dys:.3g}.")
    else:
        conclusion = "Inconclusive"
        conclusion_text = ("No statistically significant evidence that Reader View changes reading speed for readers with dyslexia "
                           f"(estimated change = {pct_dys:.2f}%, 95% CI {pct_dys_ci[0]:.2f}% to {pct_dys_ci[1]:.2f}%, p = {p_dys:.3g}).")

    # Build output object dictionary
    output_object = {
        "coef_non_dyslexic_log": est_main,
        "se_non_dyslexic_log": se_main,
        "t_non_dyslexic": t_main,
        "p_non_dyslexic": p_main,
        "ci95_non_dyslexic_log": ci_main,
        "pct_change_non_dyslexic": pct_main,
        "pct_change_non_dyslexic_ci95": pct_main_ci,
        "coef_dyslexic_log": est_dys,
        "se_dyslexic_log": se_dys,
        "t_dyslexic": t_dys,
        "p_dyslexic": p_dys,
        "ci95_dyslexic_log": ci_dys,
        "pct_change_dyslexic": pct_dys,
        "pct_change_dyslexic_ci95": pct_dys_ci,
        "interaction_name": interaction_name,
        "interaction_coef_log": est_inter,
        "interaction_se_log": se_inter,
        "interaction_t": t_inter,
        "interaction_p": p_inter,
        "interaction_ci95_log": ci_inter,
        "model_summary_text": results.summary().as_text(),
        "conclusion_label": conclusion
    }

    # Compose a concise description
    description_lines = []
    description_lines.append("Interpretation of key estimates (dependent var = log(speed)):")
    description_lines.append(f"- Non-dyslexic readers (Reader View ON vs OFF): log-coef = {est_main:.4f}, "
                              f"SE = {se_main:.4f}, t = {t_main:.3f}, p = {p_main:.3g}; "
                              f"approx. {pct_main:.2f}% change in speed (95% CI {pct_main_ci[0]:.2f}% to {pct_main_ci[1]:.2f}%).")
    description_lines.append(f"- Readers with dyslexia (combined effect): log-coef = {est_dys:.4f}, "
                              f"SE = {se_dys:.4f}, t = {t_dys:.3f}, p = {p_dys:.3g}; "
                              f"approx. {pct_dys:.2f}% change in speed (95% CI {pct_dys_ci[0]:.2f}% to {pct_dys_ci[1]:.2f}%).")
    description_lines.append(f"- Interaction term ({interaction_name}): log-coef = {est_inter:.4f}, SE = {se_inter:.4f}, "
                              f"t = {t_inter:.3f}, p = {p_inter:.3g}; 95% CI = ({ci_inter[0]:.4f}, {ci_inter[1]:.4f}).")
    description_lines.append("")
    description_lines.append("Conclusion regarding the task question:")
    description_lines.append(conclusion_text)
    description = "\n".join(description_lines)

    return {"object": output_object, "description": description}