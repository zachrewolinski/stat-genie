import pandas as pd

df = pd.read_csv("panda_nuts.csv")
print(df["feature3"].value_counts())
print(df["feature7"].value_counts())
