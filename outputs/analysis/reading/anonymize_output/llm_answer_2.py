def extract_final_answer(model_output):
    """
    Extract the estimated effect of ReaderView on log(seconds per word) for individuals with dyslexia,
    i.e. the sum of the ReaderView main effect and the ReaderView:Dyslexia interaction.

    Returns:
      {
        "object": {
            "coef": float,           # estimated combined coefficient (ReaderView + ReaderView:Dyslexia)
            "se": float,             # standard error of the combined estimate
            "t": float,              # t-statistic
            "p": float,              # two-sided p-value
            "ci_lower": float,       # lower bound of 95% CI
            "ci_upper": float,       # upper bound of 95% CI
            "percent_change": float, # (exp(coef)-1)*100, percent multiplicative change in seconds/word
            "n_params": int          # number of parameters in the model
        },
        "description": str          # brief interpretation in context
      }
    """
    import numpy as np

    # Get parameter names and locate the ReaderView and interaction terms
    params = model_output.params
    param_index = list(params.index)
    n_params = len(param_index)

    # Possible interaction name orders
    interaction_names = ['ReaderView:Dyslexia', 'Dyslexia:ReaderView']

    # Find indices for ReaderView and the interaction
    try:
        idx_reader = param_index.index('ReaderView')
    except ValueError:
        raise ValueError("Parameter 'ReaderView' not found in model parameters. "
                         "Found parameters: {}".format(param_index))

    idx_inter = None
    for name in interaction_names:
        if name in param_index:
            idx_inter = param_index.index(name)
            inter_name_used = name
            break

    if idx_inter is None:
        raise ValueError("Interaction term not found. Expected one of {}. Found parameters: {}"
                         .format(interaction_names, param_index))

    # Build contrast vector r to compute (ReaderView + ReaderView:Dyslexia)
    r = np.zeros((1, n_params))
    r[0, idx_reader] = 1.0
    r[0, idx_inter] = 1.0

    # Use statsmodels' t_test to get combined estimate, se, t, p, and confidence interval
    contrast = model_output.t_test(r)
    # summary_frame is a reliable way to get coef, se, t, p, conf int
    sf = contrast.summary_frame()
    # Identify column names used in summary_frame (they can vary slightly by version)
    # Typical columns: ['coef','std err','t','P>|t|','[0.025','0.975]']
    # We'll pick values by position to be robust
    # For a single contrast, summary_frame should have one row.
    row = sf.iloc[0]

    # Try to read by common names, else fallback to positional extraction
    # Coefficient:
    if 'coef' in sf.columns:
        coef = float(row['coef'])
    elif 'mean' in sf.columns:
        coef = float(row['mean'])
    else:
        coef = float(row[0])

    # Standard error:
    if 'std err' in sf.columns:
        se = float(row['std err'])
    elif 'std_err' in sf.columns:
        se = float(row['std_err'])
    else:
        # usually second column
        se = float(row[1])

    # t-stat:
    if 't' in sf.columns:
        tstat = float(row['t'])
    else:
        tstat = float(row[2])

    # p-value:
    pcol = None
    for cname in sf.columns:
        if cname.startswith('P') or cname.startswith('p'):
            pcol = cname
            break
    if pcol is not None:
        pval = float(row[pcol])
    else:
        # fallback to fourth column
        pval = float(row[3])

    # Confidence interval: last two columns usually
    ci_lower = float(row.iloc[-2])
    ci_upper = float(row.iloc[-1])

    # Percent multiplicative change on original scale (seconds per word):
    percent_change = (np.exp(coef) - 1.0) * 100.0

    # Build interpretation
    # Negative coef -> ReaderView reduces log(seconds/word) -> faster reading
    direction = "decrease (faster reading)" if coef < 0 else "increase (slower reading)"
    signif = "statistically significant" if pval < 0.05 else "not statistically significant"
    description = (
        f"Combined effect for dyslexic readers = ReaderView + ReaderView:Dyslexia (interaction '{inter_name_used}'). "
        f"Estimate = {coef:.4f}, SE = {se:.4f}, t = {tstat:.3f}, p = {pval:.3g}. "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. This corresponds to a {percent_change:.2f}% "
        f"{direction} in seconds per word when Reader View is ON for readers with dyslexia. "
        f"The result is {signif} at alpha=0.05."
    )

    result_object = {
        "coef": coef,
        "se": se,
        "t": tstat,
        "p": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "percent_change": percent_change,
        "n_params": n_params
    }

    return {"object": result_object, "description": description}