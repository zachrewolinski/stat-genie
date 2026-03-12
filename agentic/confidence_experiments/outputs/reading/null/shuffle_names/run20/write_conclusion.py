import json

conclusion = {
    "response": 25,
    "explanation": (
        "Using the metadata descriptions, reader view is the binary column labeled 'language' (1=on, 0=off) "
        "and dyslexia status is the 0/1/2 column labeled 'device'. Reading speed was computed as words per minute "
        "using the number-of-words column ('retake_trial') divided by the adjusted reading time in ms ('age'). "
        "Among dyslexic participants (device>0), reader view ON (n=209) had a lower mean speed (427 wpm; median 260) "
        "than reader view OFF (n=229; mean 528 wpm; median 286). The difference is not statistically significant "
        "(Welch t-test p=0.208; Mann–Whitney p=0.439) and the effect size is small and negative (Cohen’s d = -0.12). "
        "A robustness check using total page time ('adjusted_running_time') yields the same non-significant pattern (p=0.477, d=-0.07). "
        "Therefore, there is no evidence that reader view improves reading speed for individuals with dyslexia in this dataset; if anything, speeds are slightly lower when it is on."
    )
}

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    json.dump(conclusion, f)
