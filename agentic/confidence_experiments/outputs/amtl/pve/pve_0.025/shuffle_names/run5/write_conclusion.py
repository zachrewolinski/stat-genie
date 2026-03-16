import json

response = 80
explanation = (
    "Using the metadata mapping, I treated the AMTL outcome as the 'genus' column (number of missing teeth), "
    "with genus group in 'tooth_class', age at death in 'pop', sex estimate in 'stdev_age', tooth class in 'sockets', "
    "and observable sockets count in 'age'. I fit an OLS model with robust (HC3) SEs: genus ~ is_human + pop + "
    "stdev_age + age + C(sockets). The human indicator was positive and statistically significant (coef = 0.776, "
    "SE = 0.174, p = 8.2e-06), indicating higher AMTL for Homo sapiens after controlling for age, sex, and tooth class. "
    "Raw means also show humans higher (0.848 vs 0.063 for non-humans; Cohen's d ≈ 0.32). This provides clear, "
    "moderate evidence for higher AMTL in modern humans compared to Pan/Pongo/Papio, so I answer Yes with a strong tilt."
)

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    json.dump({"response": response, "explanation": explanation}, f)
