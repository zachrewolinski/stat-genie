# Task: Affairs Dataset — Children vs Extramarital Affairs

## Research question
Does having children decrease (if at all) the engagement in extramarital affairs?

## Background
This dataset comes from a 1969 Psychology Today survey on sex. Ray Fair (1978) analyzed a sample of 601 individuals currently married for the first time, focusing on extramarital affairs.

Your goal is to answer the research question using the provided dataset, with a transparent and reproducible statistical analysis.

## Files provided
- `affairs.csv` : tabular dataset
- `info.json` : dataset description and field metadata

## Dataset notes
- Outcome of interest: `affairs`
  - Numeric measure of how often extramarital intercourse occurred in the past year.
  - Values are coded with a non-linear scale (e.g., 0, 1, 2, 3, 7, 12).
- Key explanatory variable: whether the marriage has children (`children` in the unperturbed dataset).
- Other covariates include demographics and marriage-related variables (e.g., age, years married, rating, religiousness, etc.)

## Requirements / constraints
1. You must run code to compute all statistics you report.
2. Do not fabricate numbers. Every number in the report must come from the analysis code.
3. Treat this as an observational dataset: do NOT claim causal effects. Use language like “association” rather than “causes”.
4. Handle the fact that `affairs` is a non-negative outcome with many zeros. Consider at least one model that is appropriate for this outcome distribution.
5. You should include covariates to reduce confounding, but explain your modeling choice.

## Deliverables
Create the following files in this folder:

### 1) `analysis.py`
A Python script that:
- Loads `affairs.csv`
- Performs cleaning / preprocessing (including handling categorical variables)
- Fits at least two models:
  - Model A: a simple baseline association between children and affairs
  - Model B: a multivariable model controlling for plausible confounders
- Produces key outputs needed for the report:
  - sample sizes by children group
  - mean/median affairs by children group
  - model coefficients / marginal effects (or comparable effect summaries)
  - a small set of diagnostic/sanity checks

The script should print a concise summary of results to stdout.

### 2) `report.md`
A short report (≤ 400 words) that includes:
- A one-paragraph summary answer to the research question
- A table or bullet list of key descriptive statistics
- The main modeling result: direction + magnitude + uncertainty (CI or p-value)
- A brief robustness discussion (why results might change under modeling choices)
- A clear statement of limitations (observational data, measurement, etc.)

## Suggested approach (not mandatory)
- Start with descriptive stats: distribution of `affairs`, proportion of zeros, group means.
- Consider models appropriate for non-negative skewed outcomes:
  - OLS on transformed outcome (e.g., log(affairs + 1))
  - Tobit (as in Fair 1978), or
  - two-part / hurdle style modeling (affairs > 0 vs amount)
  - Poisson/NegBin if you treat it as count-like (explain caveats)
- Report an interpretable effect of children (e.g., difference in means, coefficient sign, marginal effect).

## Important: Column-name perturbation note
In some runs, column names may be anonymized or replaced.
You MUST rely on `info.json` (and/or data inspection) to identify:
- the outcome variable (affairs frequency)
- the key variable indicating presence of children
- reasonable control variables

If column names are unclear, explain how you identified the relevant fields.

## Success criteria
A run is considered successful if:
- `analysis.py` executes without errors
- `report.md` answers the research question with evidence from computed results
- conclusions are consistent with the analysis outputs
- the workflow is reproducible and statistically reasonable
