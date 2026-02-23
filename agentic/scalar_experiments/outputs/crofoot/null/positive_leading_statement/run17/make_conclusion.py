import json

# Based on the analyses in exploratory_analysis.py and run_logistic_models.py,
# we summarize the evidence.

explanation = (
    "I analyzed 58 intergroup contests between capuchin monkey groups using the provided dataset, "
    "modeling the focal group’s probability of winning as a function of both relative group size and contest location. "
    "I constructed key predictors capturing these concepts: (1) relative group size as the difference in total group size between "
    "the focal and other groups (n_focal − n_other) and as the ratio n_focal / n_other, and (2) contest location advantage as the "
    "difference in distance from each group’s home range center (dist_other − dist_focal, where positive values mean the focal group "
    "is closer to its own home range center). I then fit several logistic regression models with win (1 = focal won, 0 = focal lost) "
    "as the binary outcome.\n\n"
    "Across all models, neither relative group size nor location advantage showed a statistically significant association with the "
    "probability of winning. In a model with relative size difference and location advantage as predictors, both coefficients were small "
    "and non-significant (p-values around 0.31 for relative size and 0.53 for location advantage, with a very low pseudo R² of about 0.01), "
    "indicating that these variables explain virtually none of the variation in contest outcomes. Replacing the size difference with the size "
    "ratio led to similarly non-significant results (size ratio p ≈ 0.27, location advantage p ≈ 0.52). Adding focal group size as a control "
    "did not materially change this conclusion: the relative size coefficient became somewhat larger in magnitude but remained only marginal "
    "(p ≈ 0.08) and its confidence interval still included no effect; location and focal size remained clearly non-significant. Standardizing "
    "the predictors yielded the same pattern, with standardized coefficients for relative size and location both close to zero and non-significant.\n\n"
    "Simple descriptive checks tell the same story. When the focal group was numerically larger than its opponent, it actually won slightly less often "
    "than when it was not larger; when the focal group was closer to its own home range center, its win rate was similar or slightly lower than when "
    "it was not closer. Given the small sample size (58 contests), it is possible that true but modest effects exist and this dataset lacks power to "
    "detect them. However, based on the available evidence, we cannot reject the null hypothesis that relative group size and contest location have "
    "no measurable impact on the probability of winning. In other words, within this dataset, there is no statistically supported relationship between "
    "these predictors and contest outcomes.\n\n"
    "Therefore, although the prior belief stated in the research question is that the answer should be 'Yes', the data do not provide strong support for "
    "that claim. I interpret the results as a qualified 'No': this dataset offers little evidence that relative group size or contest location meaningfully "
    "influence the probability of a capuchin group winning an intergroup contest. To reflect this, I place my response at 25 on a 0–100 Likert scale, "
    "which corresponds to a moderately strong 'No' while acknowledging uncertainty due to the limited sample size and potential unmodeled factors."
)

result = {"response": 25, "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
