import json

response = 20
explanation = (
    "Analyzed dyslexic readers only (73 participants, 438 page reads). "
    "Mean speed with Reader View was 323.3 vs 342.3 without, a -18.9 wpm difference. "
    "Welch t-test shows no difference (p=0.697; Cohen’s d=-0.04). "
    "Within-participant pairing (n=73 with both conditions) also shows no improvement (mean diff -18.9; p=0.635). "
    "A regression on log(speed) controlling for page, device, age, gender, and native-English status with participant-clustered SEs "
    "finds Reader View effect -0.006 (~-0.6%), 95% CI [-0.096, 0.083], p=0.890. "
    "Overall, there is no statistically significant evidence that Reader View improves reading speed for individuals with dyslexia."
)

with open("conclusion.txt", "w") as f:
    json.dump({"response": response, "explanation": explanation}, f)
