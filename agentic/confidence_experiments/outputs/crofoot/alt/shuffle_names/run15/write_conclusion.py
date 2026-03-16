import json

# Key results from analysis.py run
results = json.load(open('analysis_results.json'))

explanation = (
    "I modeled focal-group win (binary m_focal) with logistic regression using "
    "relative group size (focal total - other total) and contest location "
    "(difference between the two distance-to-home-range-center variables). "
    "In the multivariate model, relative size was not statistically significant "
    f"(p={results['rel_size_p']:.3f}) and contest location was also not significant "
    f"(p={results['rel_loc_p']:.3f}; LLR p=0.274; pseudo R^2 approx 0.03). "
    "Univariate models were similarly non-significant (size p approx 0.124, location p approx 0.529). "
    "Thus, the data do not provide strong evidence that either relative group size "
    "or contest location influences win probability in this sample (n=58)."
)

output = {
    "response": 35,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(output, f)
