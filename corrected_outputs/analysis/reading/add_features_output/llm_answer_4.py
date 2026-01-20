def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View on log reading speed for:
      - non-dyslexic readers (main effect of reader_view)
      - dyslexic readers (main effect + interaction: reader_view + reader_view:dyslexia_bin)

    Returns a dictionary:
      - "object": nested dict with coefficients, standard errors, t-stats, p-values,
                  95% confidence intervals, and percent-change interpretation
                  for both groups.
      - "description": brief interpretation of what these numbers mean.
    """
    import numpy as np

    # Get parameter names and parameters
    params = model_output.params
    param_names = [str(n) for n in params.index]

    # Helper: find index of a parameter containing substrings
    def find_param_index(contains_all):
        """
        contains_all: list of substrings that must appear in the parameter name
        Returns the first matching index.
        """
        for i, name in enumerate(param_names):
            if all(sub in name for sub in contains_all):
                return i
        raise KeyError(f"No parameter name contains all of {contains_all}. Available names: {param_names}")

    try:
        # index of main reader_view coefficient (name contains 'reader_view' but not 'dyslexia_bin')
        reader_idx = find_param_index(['reader_view'])
        # ensure this isn't picking the interaction; if it is, try to find one without dyslexia_bin
        if 'dyslexia_bin' in param_names[reader_idx]:
            # find one that contains reader_view but NOT dyslexia_bin
            reader_idx = next(i for i, n in enumerate(param_names)
                              if ('reader_view' in n) and ('dyslexia_bin' not in n))
    except StopIteration:
        raise KeyError(f"Could not find main reader_view parameter in names: {param_names}")
    except KeyError:
        raise

    try:
        # index of the interaction term (name contains both 'reader_view' and 'dyslexia_bin')
        inter_idx = find_param_index(['reader_view', 'dyslexia_bin'])
    except KeyError:
        # If interaction not present, raise informative error
        raise KeyError("Interaction parameter between reader_view and dyslexia_bin not found in model parameters. "
                       f"Available parameters: {param_names}")

    k = len(param_names)
    # contrast for non-dyslexic effect: just reader_view
    contrast_non = np.zeros(k)
    contrast_non[reader_idx] = 1.0

    # contrast for dyslexic effect: reader_view + interaction
    contrast_dys = np.zeros(k)
    contrast_dys[reader_idx] = 1.0
    contrast_dys[inter_idx] = 1.0

    # Use model_output.t_test to get effect, se, t, p, and conf int for linear combinations
    res_non = model_output.t_test(contrast_non)
    res_dys = model_output.t_test(contrast_dys)

    def summarize(tres):
        # tres.effect and tres.sd may be arrays; take scalar
        effect = float(np.squeeze(tres.effect))
        sd = float(np.squeeze(tres.sd))
        tvalue = float(np.squeeze(tres.tvalue))
        pvalue = float(np.squeeze(tres.pvalue))
        # 95% CI
        ci = tres.conf_int(alpha=0.05)
        # conf_int returns ndarray shape (2,1) or (1,2) depending on version; normalize
        ci = np.asarray(ci).reshape(2,)
        lower, upper = float(ci[0]), float(ci[1])
        # percent change interpretation on original (words/sec) scale:
        pct_change = float(np.expm1(effect) * 100.0)  # (exp(effect)-1)*100
        return {
            "beta_log": effect,
            "se": sd,
            "t": tvalue,
            "p_value": pvalue,
            "ci_95_lower_log": lower,
            "ci_95_upper_log": upper,
            "percent_change": pct_change
        }

    summary_non = summarize(res_non)
    summary_dys = summarize(res_dys)

    # Also include raw parameter values for transparency
    raw = {
        "params": params.to_dict(),
        "param_names": param_names
    }

    # Short interpretation string
    description = (
        "This returns the estimated effect of turning Reader View ON on log(reading speed). "
        "For non-dyslexic readers the effect is the coefficient on 'reader_view'. "
        "For dyslexic readers the effect is the sum of 'reader_view' + 'reader_view:dyslexia_bin' (tested via linear contrast). "
        "Effects are given on the log scale (beta_log) and also translated to percent change in reading speed "
        "using (exp(beta)-1)*100. p_values test whether the effect differs from zero. "
        "If the percent_change for dyslexic readers is positive and the p_value < 0.05, that provides evidence "
        "that Reader View improves reading speed for individuals with dyslexia."
    )

    return {
        "object": {
            "non_dyslexic_reader_view_effect": summary_non,
            "dyslexic_reader_view_effect": summary_dys,
            "raw_params": raw
        },
        "description": description
    }