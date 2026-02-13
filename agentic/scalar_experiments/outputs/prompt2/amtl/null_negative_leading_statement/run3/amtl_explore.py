import json
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def summarize_genus_rates(data: pd.DataFrame, genera: List[str]) -> None:
    print("Basic AMTL rate by genus (num_amtl / sockets):")
    genus_group = data.groupby("genus", observed=True)
    rates = genus_group.apply(lambda x: x["num_amtl"].sum() / x["sockets"].sum())
    counts = genus_group.size()
    for genus in genera:
        if genus in rates.index:
            rate = rates.loc[genus]
            n_rows = counts.loc[genus]
            print(f"  {genus:12s}: rate={rate:.4f}, rows={n_rows}")
    print()


def fit_poisson_model(data: pd.DataFrame):
    formula = "num_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Poisson(),
        offset=np.log(data["sockets"]),
    ).fit()
    return model


def report_genus_effects(model) -> None:
    params = model.params
    conf_int = model.conf_int()
    pvalues = model.pvalues

    print("Genus rate ratios relative to Homo sapiens")
    print("(Poisson GLM with log(sockets) offset; rates are per socket):")
    for term in params.index:
        if term.startswith("C(genus)[T."):
            genus_name = term.split("T.")[-1].rstrip("]")
            coef = params[term]
            ci_low, ci_high = conf_int.loc[term]
            rate_ratio = float(np.exp(coef))
            rr_low = float(np.exp(ci_low))
            rr_high = float(np.exp(ci_high))
            pval = float(pvalues[term])
            direction = "higher" if rate_ratio > 1 else "lower"
            print(
                f"  {genus_name:12s}: "
                f"rate ratio={rate_ratio:.3f} ({direction} than Homo), "
                f"95% CI [{rr_low:.3f}, {rr_high:.3f}], p={pval:.3g}"
            )
    print()


def main() -> None:
    data = pd.read_csv("amtl.csv")
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    data = data[data["genus"].isin(genera_of_interest)].copy()

    # Ensure sockets are positive for the Poisson offset
    data = data[data["sockets"] > 0].copy()

    print(f"Total rows used: {len(data)}")
    print(f"Total sockets: {int(data['sockets'].sum())}")
    print(f"Total missing teeth (num_amtl): {int(data['num_amtl'].sum())}")
    print()

    summarize_genus_rates(data, genera_of_interest)

    model = fit_poisson_model(data)
    print(model.summary())
    print()
    report_genus_effects(model)


if __name__ == "__main__":
    main()

