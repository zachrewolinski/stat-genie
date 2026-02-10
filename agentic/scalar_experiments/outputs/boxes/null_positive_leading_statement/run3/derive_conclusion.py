# This script encodes the reasoning from the exploratory analysis
# into a single scalar on [-100, 100] answering:
# "Do children’s reliance on social information and preference for
# majority cues vary across cultures and developmental stages?"
#
# Interpretation of findings from `analysis.py` (manually encoded):
# - Majority choice is common overall (~0.456), indicating reliance on
#   social information and some preference for majority cues.
# - There is clear cross-cultural variation in majority vs minority
#   rates (e.g., culture 1 majority=0.330 vs culture 6=0.522), showing
#   that how strongly children follow the majority varies by culture.
# - Age correlations with majority/minority choices are near zero, and
#   age-bin majority rates are relatively flat, suggesting only weak
#   developmental change in this dataset.
#
# Thus, the answer to whether reliance on social information and
# majority preference vary *across cultures* is strongly "Yes".
# Evidence for systematic variation across *developmental stages* is
# weaker (near‑zero correlations, modest bin differences).
# Combining these, we support the overall "Yes" but not at the
# maximum strength. A value around +70 on the [-100, 100] scale
# reflects strong, but not absolute, evidence that these tendencies
# vary across cultures and development.

SCALAR = 70

if __name__ == '__main__':
    # Write the scalar to conclusion.txt as required
    with open('conclusion.txt', 'w') as f:
        f.write(str(SCALAR))
