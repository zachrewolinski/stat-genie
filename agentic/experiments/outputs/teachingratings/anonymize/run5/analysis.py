import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = "teachingratings.csv"
df = pd.read_csv(DF_PATH)

# Basic descriptive stats
beauty_col = "feature6"
rating_col = "feature7"

print("Rows, cols:", df.shape)
print("Beauty summary:\n", df[beauty_col].describe())
print("Rating summary:\n", df[rating_col].describe())

# Model 1: bivariate association
model1 = smf.ols(f"{rating_col} ~ {beauty_col}", data=df).fit()
print("\nModel 1: rating ~ beauty")
print(model1.summary())

# Model 2: add observed controls
# Categorical variables: feature2, feature4, feature5, feature8, feature9, feature10
formula2 = (
    f"{rating_col} ~ {beauty_col} + feature3 + feature11 + feature12 "
    "+ C(feature2) + C(feature4) + C(feature5) + C(feature8) + C(feature9) + C(feature10)"
)
model2 = smf.ols(formula2, data=df).fit()
print("\nModel 2: rating ~ beauty + controls")
print(model2.summary())

# Save key results for convenience
results = pd.DataFrame(
    {
        "model": ["bivariate", "controls"],
        "beauty_coef": [model1.params[beauty_col], model2.params[beauty_col]],
        "beauty_p": [model1.pvalues[beauty_col], model2.pvalues[beauty_col]],
    }
)
print("\nBeauty effect summary:\n", results)
