def extract_final_answer(model_output):
    """
    Extract key statistics about age effects on reliance on the majority from the
    model_output produced by the modeling function.

    Returns a dictionary with:
      - "object": a dict with extracted numeric results (or None if unavailable)
      - "description": a plain-language summary of what the extracted numbers mean
    """
    import numpy as np
    import pandas as pd

    # Basic validation of input
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dictionary. Expected dict with keys 'results' and 'predictions_df'."
        }

    results = model_output.get('results', None)
    preds = model_output.get('predictions_df', None)

    # Case: no fitted results (this matches the provided run where results is None)
    if results is None:
        # Try to give any useful information from predictions_df if present
        if preds is None:
            return {
                "object": None,
                "description": (
                    "No fitted model found (model_output['results'] is None) and no predictions_df present. "
                    "Cannot extract coefficients, p-values, or draw conclusions about how reliance on the majority "
                    "changes with age across sites. This usually means the model was not fit because the input data "
                    "was empty or insufficient."
                )
            }
        # predictions_df exists: check if it's empty
        try:
            empty = bool(getattr(preds, "empty", False))
        except Exception:
            empty = False
        if empty:
            return {
                "object": None,
                "description": (
                    "No fitted model found (model_output['results'] is None) and predictions_df is empty. "
                    "No information available to determine the developmental trajectory of majority reliance."
                )
            }
        else:
            # Provide a brief summary of available predictions (non-empty case)
            try:
                summary = preds['pred_prob'].describe().to_dict()
            except Exception:
                summary = None
            return {
                "object": {"pred_prob_summary": summary},
                "description": (
                    "No fitted model object available, but predictions_df was present. "
                    "Returned a summary of predicted probabilities (pred_prob). "
                    "Because the model results are missing, we cannot extract coefficients, p-values, or test Age x Site effects."
                )
            }

    # If we have a fitted model object, attempt to extract relevant statistics
    # Expecting a statsmodels results object (has params, bse, pvalues, conf_int)
    try:
        params = getattr(results, "params", None)
        pvalues = getattr(results, "pvalues", None)
        bse = getattr(results, "bse", None)
        conf = None
        if hasattr(results, "conf_int"):
            try:
                conf = results.conf_int()
            except Exception:
                conf = None

        if params is None:
            return {
                "object": None,
                "description": "The results object does not expose .params; cannot extract coefficients."
            }

        # Helper to safely pull numeric summary for a given parameter name
        def param_info(name):
            if name in params.index:
                coef = float(params.loc[name])
                se = float(bse.loc[name]) if (bse is not None and name in bse.index) else None
                p = float(pvalues.loc[name]) if (pvalues is not None and name in pvalues.index) else None
                ci = None
                if conf is not None and name in conf.index:
                    ci = (float(conf.loc[name, 0]), float(conf.loc[name, 1]))
                return {"coef": coef, "se": se, "p_value": p, "conf_int": ci}
            else:
                return None

        # Extract main terms of interest
        extracted = {}
        # Main linear and quadratic age terms
        extracted['Age_centered'] = param_info('Age_centered')
        extracted['Age_sq'] = param_info('Age_sq')

        # Interactions: parameters that include 'Age_centered' but are not the main effect
        interaction_terms = [name for name in params.index if ('Age_centered' in name and name != 'Age_centered')]
        interactions_info = {}
        for name in interaction_terms:
            interactions_info[name] = param_info(name)
        if interactions_info:
            extracted['Age_centered_interactions'] = interactions_info

        # Controls (for context)
        extracted['Female'] = param_info('Female')
        extracted['MajorityFirst'] = param_info('MajorityFirst')

        # Simple significance summary: check whether Age terms are statistically significant
        sig_summary = {}
        for term in ['Age_centered', 'Age_sq']:
            info = extracted.get(term)
            if info is None:
                sig_summary[term] = "not in model"
            else:
                p = info.get('p_value')
                if p is None:
                    sig_summary[term] = "p-value unavailable"
                else:
                    sig_summary[term] = ("significant (p < 0.05)" if p < 0.05 else f"not significant (p = {p:.3f})")

        # For interactions, report any that are significant
        interaction_summary = {}
        for name, info in interactions_info.items():
            if info is None:
                interaction_summary[name] = "no estimate"
            else:
                p = info.get('p_value')
                if p is None:
                    interaction_summary[name] = "p-value unavailable"
                else:
                    interaction_summary[name] = ("significant (p < 0.05)" if p < 0.05 else f"not significant (p = {p:.3f})")

        # Build plain-language interpretation
        interpretation_parts = []
        # Age linear/quadratic
        if extracted['Age_centered'] is None and extracted['Age_sq'] is None:
            interpretation_parts.append(
                "Model does not include Age_centered or Age_sq terms, so no inference about age-related change can be made."
            )
        else:
            # Linear
            if extracted['Age_centered'] is not None:
                coef = extracted['Age_centered']['coef']
                p = extracted['Age_centered']['p_value']
                sign = "increase" if coef > 0 else "decrease"
                if p is None:
                    interpretation_parts.append(
                        f"The linear age term (Age_centered) has coefficient {coef:.3f} (p unavailable); sign suggests a {sign} in log-odds of choosing the majority with age."
                    )
                else:
                    interpretation_parts.append(
                        f"The linear age term (Age_centered) has coefficient {coef:.3f} ({'p < 0.05' if p < 0.05 else f'p = {p:.3f}'}), indicating a {sign} in log-odds of choosing the majority with age."
                    )
            # Quadratic
            if extracted['Age_sq'] is not None:
                coef = extracted['Age_sq']['coef']
                p = extracted['Age_sq']['p_value']
                if p is None:
                    interpretation_parts.append(
                        f"The quadratic age term (Age_sq) has coefficient {coef:.3f} (p unavailable), suggesting nonlinear (curvilinear) change if meaningful."
                    )
                else:
                    interpretation_parts.append(
                        f"The quadratic age term (Age_sq) has coefficient {coef:.3f} ({'p < 0.05' if p < 0.05 else f'p = {p:.3f}'}), indicating a statistically detectable curvature in the age trajectory."
                    )

        # Interactions interpretation
        if interactions_info:
            any_sig = any(
                (info is not None and info.get('p_value') is not None and info.get('p_value') < 0.05)
                for info in interactions_info.values()
            )
            if any_sig:
                interpretation_parts.append(
                    "At least one Age x Site interaction is statistically significant, which suggests that the developmental trajectory of reliance on the majority differs across cultural sites."
                )
            else:
                interpretation_parts.append(
                    "No Age x Site interactions reached conventional significance, suggesting similar developmental trajectories across sites (no strong evidence of site moderation)."
                )

        # Compose description
        description = (
            "Extracted parameter estimates (coefficients, SEs, p-values, and CIs when available) for the linear and quadratic age terms, "
            "Age x Site interactions, and controls. Interpretation: "
            + " ".join(interpretation_parts)
        )

        return {
            "object": extracted,
            "description": description
        }

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting statistics from the fitted results object: {e}"
        }