def extract_final_answer(model_output):
    """
    Extracts the estimated effect of ReaderView on log reading speed for:
      - non-dyslexic readers (main effect of ReaderView)
      - dyslexic readers (main effect + interaction)
    and returns coefficient estimates, robust SEs, t-stats, p-values, and 95% CIs.
    Also converts log-effect to approximate percent change (exp(coef)-1).

    Returns a dict with:
      - "object": {"param_names": [...], "summary": {...}}
      - "description": human-readable summary string
    """
    import numpy as np

    res = model_output

    # Helper: get parameter names and mapping
    try:
        param_names = list(res.params.index)
    except Exception:
        # Fallback if res.params is plain array-like with attribute names
        try:
            param_names = list(res.params.keys())
        except Exception:
            param_names = []

    name_to_idx = {n: i for i, n in enumerate(param_names)}

    # Attempt to locate the ReaderView main effect term
    # Common possibilities: 'ReaderView' (numeric 0/1) or 'ReaderView[T.1]' (if treated as categorical)
    main_candidates = [n for n in param_names if n == 'ReaderView' or n.endswith('ReaderView') or n.startswith('ReaderView')]
    # Prefer exact 'ReaderView' if present
    if 'ReaderView' in param_names:
        main_name = 'ReaderView'
    elif any(n == 'ReaderView[T.1]' for n in param_names):
        main_name = next(n for n in param_names if n == 'ReaderView[T.1]')
    elif main_candidates:
        main_name = main_candidates[0]
    else:
        main_name = None

    # Attempt to locate the interaction term between ReaderView and Dyslexia
    inter_name = None
    for n in param_names:
        if ':' in n and 'ReaderView' in n and 'Dyslexia' in n:
            inter_name = n
            break
    # There can be reversed order 'Dyslexia:ReaderView'
    if inter_name is None:
        for n in param_names:
            if ':' in n:
                parts = n.split(':')
                if ('ReaderView' in parts and any('Dyslexia' == p or 'Dyslexia' in p for p in parts)):
                    inter_name = n
                    break
    # As a last resort, look for any name that contains both substrings anywhere
    if inter_name is None:
        for n in param_names:
            if 'ReaderView' in n and 'Dyslexia' in n:
                inter_name = n
                break

    results = {
        'param_names': param_names,
        'found_main_name': main_name,
        'found_interaction_name': inter_name,
    }

    # Function to run t_test for a contrast vector (1 x k)
    def run_contrast(vec):
        # vec should be length k
        vec = np.asarray(vec).reshape(1, -1)
        if vec.shape[1] != len(param_names):
            raise ValueError(f"Contrast vector length {vec.shape[1]} does not match number of parameters {len(param_names)}.")

        tt = res.t_test(vec)

        # Robust extraction of scalars/arrays from the t_test result
        def _squeeze_to_float(x):
            arr = np.asarray(x)
            # If x is a scalar or 0-d array, return float
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
            # If it's a 1-element array, also return the single float
            if arr.ndim == 1 and arr.size == 1:
                return float(arr[0])
            # Otherwise, raise because we expect a single estimate for the contrast
            raise ValueError("Unexpected result shape when extracting scalar from t_test output.")

        try:
            est = _squeeze_to_float(tt.effect)
        except Exception:
            # Some versions expose attribute 'effect' differently; attempt alternative access
            try:
                est = _squeeze_to_float(tt.effect.squeeze())
            except Exception:
                est = float(np.asarray(tt).squeeze())

        try:
            se = _squeeze_to_float(tt.sd)
        except Exception:
            # sd may be named 'sd' or 'stderr' in some versions; fallback
            se = None
            try:
                se = _squeeze_to_float(tt.stderr)
            except Exception:
                # As ultimate fallback, set NaN
                se = float("nan")

        try:
            tval = _squeeze_to_float(tt.tvalue)
        except Exception:
            tval = float("nan")

        try:
            pval = _squeeze_to_float(tt.pvalue)
        except Exception:
            pval = float("nan")

        # Confidence interval: tt.conf_int(alpha) -> array shape (1,2) or (2,)
        try:
            ci = np.asarray(tt.conf_int(alpha=0.05))
            if ci.ndim == 1 and ci.size == 2:
                ci_low, ci_high = float(ci[0]), float(ci[1])
            else:
                # take first row
                ci_low, ci_high = float(ci[0, 0]), float(ci[0, 1])
        except Exception:
            ci_low, ci_high = float("nan"), float("nan")

        return {'coef': est, 'se': se, 't': tval, 'p': pval, 'ci_low': ci_low, 'ci_high': ci_high}

    # Prepare outputs for non-dyslexic (ReaderView effect when Dyslexia=0) and dyslexic (ReaderView effect when Dyslexia=1)
    summary = {}

    k = len(param_names)
    if main_name is None or main_name not in name_to_idx:
        summary['error'] = "Could not find a parameter matching 'ReaderView' in model parameters."
    else:
        # Contrast for non-dyslexic: just the main ReaderView term
        vec_main = np.zeros(k)
        vec_main[name_to_idx[main_name]] = 1.0
        main_res = run_contrast(vec_main)
        # convert log-coef to percent change
        try:
            main_res['pct_change'] = (np.expm1(main_res['coef'])) * 100.0
        except Exception:
            main_res['pct_change'] = float("nan")
        main_res['coef_name'] = main_name
        summary['non_dyslexic_readerview_effect'] = main_res

        # If interaction exists, compute dyslexic effect (main + interaction)
        if inter_name is not None and inter_name in name_to_idx:
            vec_dys = np.zeros(k)
            vec_dys[name_to_idx[main_name]] = 1.0
            vec_dys[name_to_idx[inter_name]] = 1.0
            dys_res = run_contrast(vec_dys)
            try:
                dys_res['pct_change'] = (np.expm1(dys_res['coef'])) * 100.0
            except Exception:
                dys_res['pct_change'] = float("nan")
            dys_res['coef_name'] = f"{main_name} + {inter_name}"
            summary['dyslexic_readerview_effect'] = dys_res

            # Also report the interaction coefficient itself (difference)
            vec_inter_only = np.zeros(k)
            vec_inter_only[name_to_idx[inter_name]] = 1.0
            inter_res = run_contrast(vec_inter_only)
            try:
                inter_res['pct_change'] = (np.expm1(inter_res['coef'])) * 100.0
            except Exception:
                inter_res['pct_change'] = float("nan")
            inter_res['coef_name'] = inter_name
            summary['interaction_term'] = inter_res
        else:
            summary['note'] = "No ReaderView:Dyslexia interaction term was found; the model may not include the moderator interaction under a different name."

    # Build a concise human-readable description
    if 'error' in summary:
        description = (
            "Unable to extract ReaderView effect because the model parameters do not include a term "
            "matching 'ReaderView'. Check variable names and whether ReaderView was included as a predictor."
        )
    else:
        lines = []
        m = summary.get('non_dyslexic_readerview_effect')
        if m is not None:
            lines.append(
                "Non-dyslexic readers: ReaderView coefficient = {coef:.4f} (SE = {se:.4f}, t = {t:.3f}, p = {p:.3g}), "
                "95% CI [{low:.4f}, {high:.4f}]. This corresponds to approximately {pct:.2f}% change in reading speed.".format(
                    coef=m['coef'], se=(m['se'] if m['se'] is not None else float("nan")),
                    t=m['t'], p=m['p'], low=m['ci_low'], high=m['ci_high'], pct=m.get('pct_change', float("nan"))
                )
            )
        if 'dyslexic_readerview_effect' in summary:
            d = summary['dyslexic_readerview_effect']
            lines.append(
                "Dyslexic readers: ReaderView effect = {coef:.4f} (SE = {se:.4f}, t = {t:.3f}, p = {p:.3g}), "
                "95% CI [{low:.4f}, {high:.4f}]. Approx. {pct:.2f}% change in reading speed.".format(
                    coef=d['coef'], se=(d['se'] if d['se'] is not None else float("nan")),
                    t=d['t'], p=d['p'], low=d['ci_low'], high=d['ci_high'], pct=d.get('pct_change', float("nan"))
                )
            )
            inter = summary['interaction_term']
            lines.append(
                "Interaction (ReaderView x Dyslexia): coef = {coef:.4f} (SE = {se:.4f}, t = {t:.3f}, p = {p:.3g}), "
                "95% CI [{low:.4f}, {high:.4f}]. This is the difference in the ReaderView effect between dyslexic and non-dyslexic readers "
                "(approx. {pct:.2f}% difference).".format(
                    coef=inter['coef'], se=(inter['se'] if inter['se'] is not None else float("nan")),
                    t=inter['t'], p=inter['p'], low=inter['ci_low'], high=inter['ci_high'], pct=inter.get('pct_change', float("nan"))
                )
            )
        else:
            lines.append("No interaction term was found, so the reported ReaderView effect applies to the reference group (likely non-dyslexic).")
        description = " ".join(lines)

    # The object we return: the numeric summary dictionary plus the param names for traceability
    return {
        "object": {
            "param_names": param_names,
            "summary": summary
        },
        "description": description
    }