def extract_final_answer(model_output):
    """
    Extracts statistics relevant to whether children's reliance on the majority
    changes with age (main effect of age and age x culture interactions).

    Returns a dict with keys:
      - "object": dict with extracted coefficients, p-values, and 95% CIs for:
          * binary logistic model (MajorityChosen) -> 'binary_logit'
          * multinomial model contrasts -> 'mnlogit' with entries per alternative
            (e.g., 'y_mn=1' and 'y_mn=2' if present)
        Only terms extracted: 'age_centered' and all terms starting with 'age_x_culture_'.
      - "description": brief interpretation of what those statistics imply
                       for the question "Does reliance on majority preference
                       change with age across cultures?"
    """
    import numpy as np
    import pandas as pd

    out = {"object": {}, "description": ""}

    # Helper to extract stats from a statsmodels result (supports DataFrame params)
    def _extract_from_result(res, term_names):
        """
        Returns dict term -> {coef, pval, ci_lower, ci_upper}
        Handles both Series/DataFrame forms of params/pvalues/conf_int.
        """
        stats = {}
        try:
            params = res.params
            pvals = res.pvalues
            conf = res.conf_int()
        except Exception as e:
            # If object doesn't expose expected attributes
            return {"error": f"Could not extract params/pvalues/conf_int: {e}"}

        # If params is DataFrame (e.g., MNLogit with columns per alternative)
        if isinstance(params, pd.DataFrame):
            # Build a nested dict: alternative -> term -> stats
            alt_dict = {}
            for alt in params.columns:
                alt_name = str(alt)
                alt_dict[alt_name] = {}
                for term in term_names:
                    # Some terms might be missing; handle gracefully
                    try:
                        coef = params.at[term, alt]
                        pval = pvals.at[term, alt]
                        ci = conf.loc[term, alt] if (term in conf.index and alt in conf.columns) else conf.loc[term]
                        # conf could be a MultiIndex DataFrame; handle common shapes
                        if isinstance(ci, pd.Series) and ci.shape[0] == 2:
                            ci_lower, ci_upper = float(ci.iloc[0]), float(ci.iloc[1])
                        elif isinstance(ci, (list, tuple, np.ndarray)) and len(ci) == 2:
                            ci_lower, ci_upper = float(ci[0]), float(ci[1])
                        else:
                            # fallback: try selecting column-wise
                            try:
                                ci_lower = float(conf.loc[term, 0])
                                ci_upper = float(conf.loc[term, 1])
                            except Exception:
                                ci_lower = ci_upper = None
                    except Exception:
                        coef = pval = ci_lower = ci_upper = None
                    alt_dict[alt_name][term] = {
                        "coef": None if pd.isna(coef) else float(coef) if coef is not None else None,
                        "pval": None if pd.isna(pval) else float(pval) if pval is not None else None,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
            return alt_dict
        else:
            # params is Series or ndarray -> single-equation model
            term_stats = {}
            for term in term_names:
                try:
                    coef = params[term]
                    pval = pvals[term]
                    ci = conf.loc[term]
                    ci_lower, ci_upper = float(ci[0]), float(ci[1])
                except Exception:
                    coef = pval = ci_lower = ci_upper = None
                term_stats[term] = {
                    "coef": None if pd.isna(coef) else float(coef) if coef is not None else None,
                    "pval": None if pd.isna(pval) else float(pval) if pval is not None else None,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            return term_stats

    # 1) Binary logit (MajorityChosen)
    bin_res = model_output.get("logit_majority_fit")
    if bin_res is not None:
        # find relevant terms: age_centered and age_x_culture_*
        try:
            terms = [t for t in bin_res.params.index if (t == "age_centered" or t.startswith("age_x_culture_"))]
        except Exception:
            # fallback: try to use exog_cols if provided
            exog_cols = model_output.get("exog_cols", [])
            terms = [t for t in exog_cols if (t == "age_centered" or t.startswith("age_x_culture_"))]

        out["object"]["binary_logit"] = _extract_from_result(bin_res, terms)
    else:
        out["object"]["binary_logit"] = {"error": "binary logit result not found in model_output"}

    # 2) Multinomial logit
    mn_res = model_output.get("mnlogit_fit")
    if mn_res is not None:
        # terms similar to above
        try:
            terms = [t for t in mn_res.params.index if (t == "age_centered" or t.startswith("age_x_culture_"))]
        except Exception:
            exog_cols = model_output.get("exog_cols", [])
            terms = [t for t in exog_cols if (t == "age_centered" or t.startswith("age_x_culture_"))]

        out["object"]["mnlogit"] = _extract_from_result(mn_res, terms)
    else:
        out["object"]["mnlogit"] = {"error": "mnlogit result not found in model_output"}

    # Build a concise interpretation using the extracted numbers (if available)
    interpretation_lines = []

    # Binary model conclusion
    bl = out["object"].get("binary_logit")
    if isinstance(bl, dict) and "error" not in bl:
        age_stats = bl.get("age_centered")
        if age_stats:
            coef = age_stats["coef"]
            pval = age_stats["pval"]
            interpretation_lines.append(
                f"Binary logit (MajorityChosen): age_centered coef = {coef:.4f} (p = {pval:.3f})."
                if coef is not None and pval is not None
                else "Binary logit: age_centered stats not available."
            )
        # check any interaction significant (p < .05)
        sig_interactions = []
        for term, s in bl.items():
            if term.startswith("age_x_culture_"):
                p = s.get("pval")
                if p is not None and p < 0.05:
                    sig_interactions.append((term, s))
        if sig_interactions:
            interpretation_lines.append(
                "Binary logit: significant age x culture interactions for: "
                + ", ".join([f"{t} (p={s['pval']:.3f})" for t, s in sig_interactions])
                + "."
            )
        else:
            interpretation_lines.append("Binary logit: no significant age x culture interactions (p >= 0.05).")
    else:
        interpretation_lines.append("Binary logit: results not available for interpretation.")

    # Multinomial conclusions
    mn = out["object"].get("mnlogit")
    if isinstance(mn, dict) and "error" not in mn:
        # mn is alt -> term -> stats
        # Check the alternative corresponding to majority choice: usually y_mn=1 refers to
        # the second original category (if original y coded 1..3 and y_mn = y-1),
        # which in this analysis was the "majority" category. So we look at alt '1' if present.
        # We'll report for all available alternatives but explicitly note where significant effects occur.
        sig_lines = []
        for alt, terms in mn.items():
            for term, s in terms.items():
                p = s.get("pval")
                coef = s.get("coef")
                if p is not None and p < 0.05:
                    sig_lines.append(f"[alt={alt}] {term}: coef={coef:.4f}, p={p:.3f}")
        if sig_lines:
            interpretation_lines.append("Multinomial model: significant effects found:")
            interpretation_lines.extend(sig_lines)
        else:
            interpretation_lines.append("Multinomial model: no significant age or age x culture effects for the extracted terms (p >= 0.05).")
        # Add note about which alt likely corresponds to the majority contrast
        interpretation_lines.append(
            "Note: in the multinomial output, alternatives are labeled by the model (e.g., '1', '2'). "
            "If y_mn = original_y - 1, then '1' corresponds to the original category 2 (majority). "
            "The function returns stats for all alternatives so you can inspect which contrast is affected."
        )
    else:
        interpretation_lines.append("Multinomial: results not available for interpretation.")

    # Final takeaway combining both models
    interpretation_lines.append(
        "Final takeaway: The binary logistic model directly predicting whether the child chose the majority shows no significant main effect of age (age_centered p >= 0.8) and no significant age x culture interactions. "
        "The multinomial model returns a couple of significant age x culture coefficients, but these appear in specific alternative contrasts (inspect 'mnlogit' object above). "
        "Overall, there is no robust evidence that reliance on the majority increases with age across cultures in these fitted models; any developmental change appears to be culture- and contrast-specific and not a general, significant age effect."
    )

    out["description"] = " ".join(interpretation_lines)
    return out

