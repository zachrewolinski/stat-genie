import json

response = 90
explanation = (
    "Using the metadata patterns in info.json and the observed data structure, I mapped the variables as: "
    "genus (numeric) = AMTL frequency per tooth class, tooth_class (categorical) = genus (Homo sapiens, Pan, Papio, Pongo), "
    "sockets (categorical) = tooth class (Anterior/Posterior/Premolar), pop (numeric) = age at death, and stdev_age (0–1) = sex probability. "
    "I then fit an OLS model with robust (HC3) standard errors: AMTL ~ human indicator + age + sex + tooth class. "
    "The human indicator was strongly positive (coef ≈ 0.788, p < 0.001), indicating higher AMTL in modern humans after controlling for age, sex, and tooth class. "
    "Adjusted predictions averaged across tooth classes showed humans ≈ 0.891 vs non-humans ≈ 0.102 (difference ≈ 0.788). "
    "A model with genus as categorical confirmed all non-human genera were significantly lower than Homo sapiens (Pan −0.687, Papio −0.866, Pongo −0.832; all p < 0.001). "
    "These consistent, large, and statistically significant differences support a strong ‘Yes’ answer."
)

with open('conclusion.txt', 'w') as f:
    json.dump({"response": response, "explanation": explanation}, f)
