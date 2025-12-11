import json
from stat_genie.blade_pipeline.llms.config import llm

def judge_models(llm_provider: str,
                 llm_model: str,
                 research_question: str,
                 models1: list[dict],
                 models2: list[dict]):

    model_judge = llm(provider=llm_provider, model=llm_model)

    example_research_question = (
        "What is the effect of hormonal fluctuations associated with fertility "
        "on women's religiosity?"
    )

    example_score_5 = {
        "Model Set #1": [
            {
                "model_library": "statsmodels (statsmodels.formula.api)",
                "model_class": "OLS (Ordinary Least Squares) fitted via smf.ols, returning an OLSResults object",
                "model_parameters": (
                    "cov_type='HC3' (heteroskedasticity-robust HC3 standard errors); "
                    "FertilityGroup cast to categorical and reordered so 'Low' is reference; "
                    "formula includes interaction term C(FertilityGroup) * InRelationship plus "
                    "ReportedCycleLength and DateCertainty; no additional hyperparameters specified."
                ),
                "model_formula_fitting_code":
                    "formula = 'AvgReligiosity ~ C(FertilityGroup) * InRelationship + ReportedCycleLength + DateCertainty'\n"
                    "model = smf.ols(formula, data=df).fit(cov_type='HC3')"
            }
        ],
        "Model Set #2": [
            {
                "model_library": "statsmodels (statsmodels.formula.api)",
                "model_class": "OLS (Ordinary Least Squares) via statsmodels.formula.api.ols",
                "model_parameters": (
                    "Default OLS settings; no robust covariance estimator specified; "
                    "data is cleaned by dropping NA rows; categorical coding used via C(FertilityGroup); "
                    "formula specifies interaction: AvgReligiosity ~ InRelationship * C(FertilityGroup)."
                ),
                "model_formula_fitting_code":
                    "formula = 'AvgReligiosity ~ InRelationship * C(FertilityGroup)'\n"
                    "results = smf.ols(formula=formula, data=model_df).fit()"
            }
        ],
        "Model Similarity Score": 5
    }

    example_score_1 = {
        "Model Set #1": [
            {
                "model_library": "statsmodels (statsmodels.formula.api)",
                "model_class": "OLS (Ordinary Least Squares) fitted via smf.ols",
                "model_parameters": (
                    "cov_type='HC3' (heteroskedasticity-robust HC3 standard errors); "
                    "FertilityGroup cast to categorical and reordered so 'Low' is reference; "
                    "formula includes C(FertilityGroup) * InRelationship along with "
                    "ReportedCycleLength and DateCertainty; no other hyperparameters specified."
                ),
                "model_formula_fitting_code":
                    "formula = 'AvgReligiosity ~ C(FertilityGroup) * InRelationship + ReportedCycleLength + DateCertainty'\n"
                    "model = smf.ols(formula, data=df).fit(cov_type='HC3')"
            }
        ],
        "Model Set #2": [
            {
                "model_library": "statsmodels (statsmodels.formula.api)",
                "model_class": "OLS (Ordinary Least Squares via smf.ols)",
                "model_parameters": (
                    "Uses default OLS estimation; FertilityGroup cast to categorical; "
                    "formula includes interaction C(FertilityGroup) * InRelationship and controls "
                    "DateCertainty and ReportedCycleLength_clean; no additional parameters passed to fit()."
                ),
                "model_formula_fitting_code":
                    "formula = 'AvgReligiosity ~ C(FertilityGroup) * InRelationship + DateCertainty + ReportedCycleLength_clean'\n"
                    "results = smf.ols(formula, data=model_df).fit()"
            }
        ],
        "Model Similarity Score": 1
    }


    example_score_3 = {
        "Model Set #1": [
            {
                "model_library": "statsmodels (statsmodels.formula.api)",
                "model_class": "OLS (statsmodels.formula.api.ols) – Ordinary Least Squares regression",
                "model_parameters": (
                    "cov_type='HC3' (heteroskedasticity-robust SEs); "
                    "categorical encoding for FertilityGroup via C(...); interaction term "
                    "C(FertilityGroup) * InRelationship; controls included: DaysFromOvulation, "
                    "SureMean, ReportedCycleLength_used."
                ),
                "model_formula_fitting_code":
                    "formula = 'AvgReligiosity ~ C(FertilityGroup) * InRelationship + DaysFromOvulation + SureMean + ReportedCycleLength_used'\n"
                    "results = smf.ols(formula, data=df).fit(cov_type='HC3')"
            }
        ],
        "Model Set #2": [
            {
                "model_library": "statsmodels (statsmodels.formula.api)",
                "model_class": "OLS (Ordinary Least Squares via statsmodels.formula.api.ols)",
                "model_parameters": (
                    "Default OLS settings; FertilityGroup cast to categorical; "
                    "includes interaction C(FertilityGroup) * InRelationship; controls include "
                    "SureAvg and ReportedCycleLength; no robust covariance estimator applied."
                ),
                "model_formula_fitting_code":
                    "df['FertilityGroup'] = df['FertilityGroup'].astype('category')\n"
                    "formula = 'AvgReligiosity ~ C(FertilityGroup) * InRelationship + SureAvg + ReportedCycleLength'\n"
                    "results = smf.ols(formula=formula, data=df).fit()"
            }
        ],
        "Model Similarity Score": 3
    }


    judge_system_prompt = (
        "You are a meticulous research design evaluator specializing in **model specification comparison**.\n"
        "Your responsibility is to evaluate the structural and methodological similarity between two models.\n\n"
        "Evaluate similarity based on:\n"
        "1. Model type (OLS, logistic, mixed-effects, etc.)\n"
        "2. Formula structure: predictors, interactions, coding choices, controls\n"
        "3. Functional form (e.g., categorical coding, interaction terms, covariates)\n"
        "4. Estimation approach (robust SEs, etc.)\n"
        "5. Whether both models test the same substantive hypothesis\n\n"
        "Ignore superficial naming differences. Focus on methodological equivalence.\n\n"
        "Scoring scale:\n"
        "1 = completely different\n"
        "2 = somewhat different\n"
        "3 = moderately similar\n"
        "4 = very similar\n"
        "5 = almost identical\n"
    )

    judge_user_prompt = (
        f"Research Question:\n{research_question}\n\n"
        f"==================== EXAMPLE SCORES ====================\n\n"
        f"Example Score 1:\n{example_score_1}\n\n"
        f"Example Score 3:\n{example_score_3}\n\n"
        f"Example Score 5:\n{example_score_5}\n\n"
        f"==================== MODEL SET 1 ====================\n{models1}\n\n"
        f"==================== MODEL SET 2 ====================\n{models2}\n\n"
        f"Please evaluate the similarity between these model specifications, focusing on:\n"
        f"- Predictors, interactions, covariates\n"
        f"- Model type and coding strategy\n"
        f"- Estimation and error structure\n"
        f"- Conceptual equivalence in testing the hypothesis\n\n"
        f"Return JSON only:\n"
        f"{{\n"
        f"  \"Model Similarity Score\": <number>\n"
        f"}}"
    )

    result = model_judge.generate([
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": judge_user_prompt}
    ])

    try:
        raw = result.text[0].content
    except:
        raw = result.text

    return json.loads(raw)


def make_judge_prompt(task, data_head, featA, featB, modelA, modelB, conclA, conclB):
    return (
        f"Research Question / Context:\n{task}\n\n"
        "Here is a sample of the dataset to understand the structure and variables:\n"
        f"{data_head}\n\n"
        "Compare the two trials methodologically and interpretively based on the provided variables, model specifications, and conclusions.\n\n"
        "==================== TRIAL A ====================\n\n"
        "Independent Variables:\n"
        f"{featA['independent_variables']}\n\n"
        "Control Variables:\n"
        f"{featA.get('control_variables')}\n\n"
        "Response Variables:\n"
        f"{featA['response_variables']}\n\n"
        "Model Specification:\n"
        f"{modelA}\n\n"
        "Conclusion:\n"
        f"{conclA}\n\n"
        "==================== TRIAL B ====================\n\n"
        "Independent Variables:\n"
        f"{featB['independent_variables']}\n\n"
        "Control Variables:\n"
        f"{featB.get('control_variables')}\n\n"
        "Response Variables:\n"
        f"{featB['response_variables']}\n\n"
        "Model Specification:\n"
        f"{modelB}\n\n"
        "Conclusion:\n"
        f"{conclB}\n\n"
        "Now, following your reasoning plan, provide similarity ratings as JSON only."
    )

def run_judge_evaluation_pairwise(
    task, data_head,
    features_1, features_2,
    model_info_1, model_info_2,
    conclusions_1, conclusions_2,
    llm_provider="openai", llm_model="gpt-5-mini",
    output_path=None
):
    judge_system_prompt = (
        "You are a meticulous research design evaluator. "
        "Your role is to compare two experimental trials methodologically **and interpretively**.\n\n"
        "You will go through the following reasoning plan step-by-step (internally):\n"
        "1. Understand the research question and dataset context.\n"
        "2. Examine independent, control, and response variables for both trials.\n"
        "3. Analyze the model specifications for structural or methodological similarity.\n"
        "4. Focus more on the content, less on the format.\n"
        "5. Assess whether the trials' conclusions are logically consistent given their setups.\n"
        "6. Detect whether either input is None, invalid, erroneous, or incomplete.\n"
        "   - If **one trial** shows errors or missing components but the other is valid, "
        "     impose a **strong penalty** (reduce all category scores by at least 1 point, "
        "     and cap overall similarity at 2).\n"
        "7. Synthesize your evaluation across all components.\n"
        "8. Output a numerical rating for each category.\n\n"
        "DO NOT include your reasoning — only the final JSON object.\n\n"
        "Scoring scale:\n"
        "1 = completely different\n"
        "2 = somewhat different\n"
        "3 = moderately similar\n"
        "4 = very similar\n"
        "5 = almost identical\n\n"
        "Return output **strictly in JSON format**:\n"
        "{\n"
        "  \"independent_variables\": <number>,\n"
        "  \"control_variables\": <number>,\n"
        "  \"response_variables\": <number>,\n"
        "  \"model_specification\": <number>,\n"
        "  \"conclusions\": <number>,\n"
        "  \"overall_similarity\": <number>\n"
        "}"
    )

    llm_judge = llm(provider=llm_provider, model=llm_model)

    pairwise_results = {}
    nA = len(features_1)
    nB = len(features_2)

    for i in range(nA):
        for j in range(nB):

            user_prompt = make_judge_prompt(
                task, data_head,
                features_1[i], features_2[j],
                model_info_1[i], model_info_2[j],
                conclusions_1[i], conclusions_2[j]
            )

            result = llm_judge.generate([
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            if hasattr(result, "text"):
                text = result.text
            elif hasattr(result, "content"):
                text = result.content
            else:
                text = str(result)

            text = str(text).strip()

            clean = (
                text.replace("```json", "")
                    .replace("```", "")
                    .strip()
            )

            pairwise_results[(i, j)] = clean

    if output_path:
        serializable = {}
        for k, v in pairwise_results.items():
            try:
                serializable[str(k)] = json.loads(v)
            except:
                serializable[str(k)] = v 

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

    return pairwise_results


