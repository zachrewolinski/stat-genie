import json
from typing import Any, Dict, List, Optional, Tuple

from stat_genie.blade_pipeline.llms.config import llm


def judge_features(llm_provider: str,
                   llm_model: str,
                   research_question: str,
                   feature_type: str,
                   features1: list[dict],
                   features2: list[dict],
                   use_cache: bool = True):
    
    # instantiate the LLM
    feature_judge = llm(provider=llm_provider, model=llm_model, use_cache=use_cache)
    
    # define the feature type descriptions
    feature_type_description = {
        "independent_variables": "independent variables",
        "control_variables": "control variables",
        "response_variables": "response variables"
    }
    
    # define some examples
    example_research_question = "What is the effect of hormonal fluctuations associated with fertility on women's religiosity?"
    example_score_1 = {
        "Feature Type": "independent variables",
        "Feature Set #1": [[{'description': "Women's fertility status at time of testing, derived from cycle day relative to ovulation (High vs Low fertility). Coded as 'High-Fertility' or 'Low-Fertility'.",
                             'columns': ['FertilityGroup'],
                             # 'transform_code': ["df['ExpectedNextPeriod'] = df['feature10'] + pd.to_timedelta(df['ReportedCycleLength'], unit='d')\ndf['OvulationDate'] = df['ExpectedNextPeriod'] - pd.to_timedelta(14, unit='d')\ndf['CycleDay'] = (df['feature9'] - df['OvulationDate']).dt.days + 14\n\ndef assign_fertility(cd):\n    if pd.isnull(cd):\n        return 'Other'\n    try:\n        cd_int = int(cd)\n    except Exception:\n        return 'Other'\n    if 6 <= cd_int <= 14:\n        return 'High-Fertility'\n    elif 17 <= cd_int <= 27:\n        return 'Low-Fertility'\n    else:\n        return 'Other'\n\ndf['FertilityGroup'] = df['CycleDay'].apply(assign_fertility)\ndf = df[df['FertilityGroup'].isin(['High-Fertility', 'Low-Fertility'])].copy()\ndf['FertilityGroup'] = df['FertilityGroup'].astype('category')"]
                             }]],
        "Feature Set #2": [[{'description': "Women's fertility. Operationalized by continuous proximity to ovulation measured in days ('DaysFromOvulation', where 0 = ovulation day, negative = days before ovulation).",
                             'columns': ['DaysFromOvulation'],
                             # 'transform_code': ["def _fert_group(x):\n    if pd.isna(x):\n        return 'Other'\n    if -5 <= x <= 0:\n        return 'High-Fertility'\n    if 7 <= x <= 14:\n        return 'Low-Fertility'\n    return 'Other'\n\ndf['FertilityGroup'] = df['DaysFromOvulation'].apply(_fert_group)\n\ndf = df[df['FertilityGroup'].isin(['High-Fertility', 'Low-Fertility'])].copy()",
                             #                    "df['ExpectedNextPeriod'] = df['StartDateofLastPeriod'] + pd.to_timedelta(df['ReportedCycleLength'], unit='d')\ndf['OvulationDate'] = df['ExpectedNextPeriod'] - pd.to_timedelta(14, unit='d')\ndf['DaysFromOvulation'] = (df['DateTesting'] - df['OvulationDate']).dt.days"]
                             }]],
        "Overall Similarity": 1,
    }
    example_score_3 = {
        "Feature Type": "control variables",
        "Feature Set #1": [[{'description': 'Binary indicator for whether the participant is in a romantic relationship (0 = not dating/romantically involved, 1 = dating/engaged/married). Treated as a moderator on the effect of fertility on religiosity.',
                             'is_moderator': True,
                             'moderator_on': "Women's fertility",
                             'columns': ['InRelationship'],
                             # 'transform_code': ["df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')\ndf['InRelationship'] = df['feature7'].apply(lambda x: 0 if x == 1 else (1 if not pd.isna(x) else np.nan))"]
                             },
                            {'description': 'Average confidence in the reported start dates of last and prior period (1-9). Controls for measurement error in cycle timing.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['AvgDateConfidence'],
                             # 'transform_code': ["df['feature5'] = pd.to_numeric(df['feature5'], errors='coerce')\ndf['feature6'] = pd.to_numeric(df['feature6'], errors='coerce')\ndf['AvgDateConfidence'] = df[['feature5', 'feature6']].mean(axis=1)"]
                             },
                            {'description': "Final cycle length used for timing calculations (days). Either the participant's reported cycle length or the interval between reported start dates if reported is missing or implausible.",
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['CycleLengthFinal'],
                             # 'transform_code': ["# Construct CycleLengthFinal: prefer reported (feature8) if plausible (21-38 days), otherwise compute from dates\ndf['feature8'] = pd.to_numeric(df['feature8'], errors='coerce')\n\n# Compute cycle length from the two reported start dates if both present\ndf['CycleLength_from_dates'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days\n\n# Use reported if between 21 and 38, else use computed, else NaN\ndef choose_cycle_length(row):\n    rep = row['feature8']\n    comp = row['CycleLength_from_dates']\n    if not pd.isna(rep) and 21 <= rep <= 38:\n        return float(rep)\n    if not pd.isna(comp) and 21 <= comp <= 38:\n        return float(comp)\n    # fallback to reported if present even if slightly outside range\n    if not pd.isna(rep):\n        return float(rep)\n    return np.nan\n\ndf['CycleLengthFinal'] = df.apply(choose_cycle_length, axis=1)"]
                             }]],
        "Feature Set #2": [[{'description': 'Binary indicator for whether participant is in any romantic relationship (0 = not dating/romantically involved [feature7 == 1], 1 = dating/engaged/married [feature7 in 2,3,4]). This variable is modeled as a moderator of the fertility effect.',
                             'is_moderator': True,
                             'moderator_on': 'FertilityGroup',
                             'columns': ['InRelationship'],
                             # 'transform_code': ["df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')\n\ndef in_relationship_code(x):\n    if pd.isna(x):\n        return np.nan\n    return 0 if x == 1 else 1\n\ndf['InRelationship'] = df['feature7'].apply(in_relationship_code)\n\ndf = df.dropna(subset=['InRelationship', 'IsCommitted'])\n\ndf['InRelationship'] = df['InRelationship'].astype('int64')"]
                             },
                            {'description': 'Mean confidence in the reported period start dates (average of feature5 and feature6). Used as a control for data quality / date recall accuracy.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['SureMean'],
                             # 'transform_code': ["# Confidence in date reports: mean of feature5 and feature6 (if one is missing, mean will use the other)\nsure_cols = [c for c in ['feature5', 'feature6'] if c in df.columns]\nif sure_cols:\n    df['SureMean'] = df[sure_cols].mean(axis=1)\nelse:\n    # If neither confidence item is present, set SureMean to NaN (keeps column contract)\n    df['SureMean'] = np.nan"]
                             },
                            {'description': 'Binary indicator for committed relationship status (1 = engaged or married; feature7 in [3,4], 0 otherwise). Included as an additional relationship-related control for robustness checks.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['IsCommitted'],
                             # 'transform_code': ["df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')\n\ndef is_committed_code(x):\n    if pd.isna(x):\n        return np.nan\n    return 1 if x in [3, 4] else 0\n\ndf['IsCommitted'] = df['feature7'].apply(is_committed_code)\n\ndf = df.dropna(subset=['InRelationship', 'IsCommitted'])\n\ndf['IsCommitted'] = df['IsCommitted'].astype('int64')"]
                             }]],
        "Control Variables Similarity Score": 3,
    }
    example_score_5 = {
        "Feature Type": "control variables",
        "Feature Set #1": [[{'description': 'Binary indicator for whether the participant is in a romantic relationship (0 = not dating/romantically involved, 1 = dating/engaged/married). Treated as a moderator on the effect of fertility on religiosity.',
                             'is_moderator': True,
                             'moderator_on': "Women's fertility",
                             'columns': ['InRelationship'],
                             # 'transform_code': ["df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')\ndf['InRelationship'] = df['feature7'].apply(lambda x: 0 if x == 1 else (1 if not pd.isna(x) else np.nan))"]
                             },
                            {'description': 'Average confidence in the reported start dates of last and prior period (1-9). Controls for measurement error in cycle timing.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['AvgDateConfidence'],
                             # 'transform_code': ["df['feature5'] = pd.to_numeric(df['feature5'], errors='coerce')\ndf['feature6'] = pd.to_numeric(df['feature6'], errors='coerce')\ndf['AvgDateConfidence'] = df[['feature5', 'feature6']].mean(axis=1)"]
                             },
                            {'description': "Final cycle length used for timing calculations (days). Either the participant's reported cycle length or the interval between reported start dates if reported is missing or implausible.",
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['CycleLengthFinal'],
                             # 'transform_code': ["# Construct CycleLengthFinal: prefer reported (feature8) if plausible (21-38 days), otherwise compute from dates\ndf['feature8'] = pd.to_numeric(df['feature8'], errors='coerce')\n\n# Compute cycle length from the two reported start dates if both present\ndf['CycleLength_from_dates'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days\n\n# Use reported if between 21 and 38, else use computed, else NaN\ndef choose_cycle_length(row):\n    rep = row['feature8']\n    comp = row['CycleLength_from_dates']\n    if not pd.isna(rep) and 21 <= rep <= 38:\n        return float(rep)\n    if not pd.isna(comp) and 21 <= comp <= 38:\n        return float(comp)\n    # fallback to reported if present even if slightly outside range\n    if not pd.isna(rep):\n        return float(rep)\n    return np.nan\n\ndf['CycleLengthFinal'] = df.apply(choose_cycle_length, axis=1)"]
                             }]],
        "Feature Set #2": [[{'description': 'Binary indicator for whether the respondent is currently in a romantic relationship (0 = not dating/romantically involved, 1 = dating/engaged/married). Treated as a moderator of the fertility effect.',
                             'is_moderator': True,
                             'moderator_on': "Women's fertility",
                             'columns': ['InRelationship'],
                             # 'transform_code': ["# Relationship: 1 = not dating, 2 = dating/one partner, 3 = engaged/living together, 4 = married\n# InRelationship = 0 if not dating, 1 otherwise\ndf['InRelationship'] = df['Relationship'].apply(lambda x: 0 if (pd.isna(x) or int(x) == 1) else 1)"]
                             },
                            {'description': 'Reported (or calculated) cycle length in days. Controls for individual differences in cycle timing that affect ovulation estimation.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['CycleLength'],
                             # 'transform_code': ["df['CycleLength'] = pd.to_numeric(df['CycleLength'], errors='coerce')\ndf['CalcCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days\ndf['CycleLength'] = df['CycleLength'].fillna(df['CalcCycleLength'])\ndf = df.dropna(subset=['DateTesting', 'StartDateofLastPeriod', 'CycleLength'])\ndf = df[(df['CycleLength'] > 18) & (df['CycleLength'] < 45)]"]
                             },
                            {'description': 'Average certainty about reported start dates for the last two periods (higher = more certain). Controls for measurement reliability of derived fertility timing.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['SureMean'],
                             # 'transform_code': ["df['Sure1'] = pd.to_numeric(df['Sure1'], errors='coerce')\ndf['Sure2'] = pd.to_numeric(df['Sure2'], errors='coerce')\ndf['SureMean'] = df[['Sure1', 'Sure2']].mean(axis=1)"]
                             }]],
        "Control Variables Similarity Score": 5,
    }
    
    # define the system prompt
    judge_system_prompt = (
        f"You are a meticulous research design evaluator specializing in feature set comparison. "
        f"Your role is to assess the similarity between two sets of {feature_type} used to answer "
        f"research questions.\n\n"
        f"Each feature in the sets includes at least the following:\n"
        f"- A description of what the variable represents\n"
        f"- A column name from the dataset\n"
        # f"- Associated transformation/cleaning/preprocessing code\n\n"
        f"Your evaluation should focus on **structural or methodological similarity** rather than "
        f"superficial naming conventions. When comparing features:\n"
        f"1. **Prioritize similar descriptions** over similar column names. Two features with "
        f"   different column names but conceptually identical descriptions should be considered "
        f"   highly similar.\n"
        f"2. Assess whether the features serve the same analytical purpose in answering the "
        f"   research question, even if implemented differently.\n"
        f"3. Look for semantic equivalence in descriptions rather than exact string matches.\n"
        # f"4. Consider the methodological approach: Are the transformations, cleaning steps, and "
        # f"   preprocessing methods structurally similar or equivalent?\n\n"
    )
    
    # define the user prompt
    judge_user_prompt = (
        f"Research Question:\n{research_question}\n\n"
        f"Compare the following two sets of {feature_type_description[feature_type]} "
        f"and assess their similarity based on structural and methodological equivalence.\n\n"
        f"Scoring scale:\n"
        f"1 = completely different\n"
        f"2 = somewhat different\n"
        f"3 = moderately similar\n"
        f"4 = very similar\n"
        f"5 = almost identical\n"
        f"==================== EXAMPLE SCORES ====================\n\n"
        f"Research Question: {example_research_question}\n"
        f"Example Score 1: {example_score_1}\n"
        f"Example Score 3: {example_score_3}\n"
        f"Example Score 5: {example_score_5}\n\n"
        f"==================== FEATURE SET 1 ====================\n\n"
        f"{features1}\n\n"
        f"==================== FEATURE SET 2 ====================\n\n"
        f"{features2}\n\n"
        f"Please evaluate the similarity between these two feature sets, focusing on:\n"
        f"- Conceptual equivalence in descriptions (prioritize this over column name matches)\n"
        # f"- Structural similarity in transformations and preprocessing methods\n"
        f"- Methodological equivalence in how they serve the research question\n\n"
        f"Provide your similarity score as JSON only:\n"
        f"{{\n"
        f"  \"{feature_type_description[feature_type].title()} Similarity Score\": <number>\n"
        f"}}"
    )
    
    # generate the result
    result = feature_judge.generate([
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": judge_user_prompt}
    ])
    
    # get text result and convert to json -> dictionary
    result = result.text[0].content
    result = json.loads(result)
    
    return result
    
def judge_models(llm_provider: str,
                 llm_model: str,
                 research_question: str,
                 models1: list[dict],
                 models2: list[dict],
                 use_cache: bool = True):

    model_judge = llm(provider=llm_provider, model=llm_model, use_cache=use_cache)

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
        f"Example Research Question: {example_research_question}"
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


# def make_judge_prompt(task, data_head, featA, featB, modelA, modelB, conclA, conclB):
#     return (
#         f"Research Question / Context:\n{task}\n\n"
#         "Here is a sample of the dataset to understand the structure and variables:\n"
#         f"{data_head}\n\n"
#         "Compare the two trials methodologically and interpretively based on the provided variables, model specifications, and conclusions.\n\n"
#         "==================== TRIAL A ====================\n\n"
#         "Independent Variables:\n"
#         f"{featA['independent_variables']}\n\n"
#         "Control Variables:\n"
#         f"{featA.get('control_variables')}\n\n"
#         "Response Variables:\n"
#         f"{featA['response_variables']}\n\n"
#         "Model Specification:\n"
#         f"{modelA}\n\n"
#         "Conclusion:\n"
#         f"{conclA}\n\n"
#         "==================== TRIAL B ====================\n\n"
#         "Independent Variables:\n"
#         f"{featB['independent_variables']}\n\n"
#         "Control Variables:\n"
#         f"{featB.get('control_variables')}\n\n"
#         "Response Variables:\n"
#         f"{featB['response_variables']}\n\n"
#         "Model Specification:\n"
#         f"{modelB}\n\n"
#         "Conclusion:\n"
#         f"{conclB}\n\n"
#         "Now, following your reasoning plan, provide similarity ratings as JSON only."
#     )

def _combine_judge_responses(variables_dict: Dict, modeling_dict: Dict, 
                              conclusions_dict: Dict) -> Dict:
    """
    Combine three separate judge responses into a single dictionary with overall_similarity.
    
    Args:
        variables_dict: Dictionary with independent_variables, control_variables, response_variables
        modeling_dict: Dictionary with model_specification
        conclusions_dict: Dictionary with conclusions
        
    Returns:
        Combined dictionary with all scores plus overall_similarity
    """
    combined = {}
    combined.update(variables_dict)
    combined.update(modeling_dict)
    combined.update(conclusions_dict)
    
    # weighted average - all categories get equal weight for now
    weights = {
        'Independent Variables Similarity Score': 1.0,
        'Control Variables Similarity Score': 1.0,
        'Response Variables Similarity Score': 1.0,
        'Model Similarity Score': 1.0,
        'conclusions': 1.0
    }
    
    weighted_sum = sum(combined[k] * weights[k] for k in weights.keys())
    total_weight = sum(weights.values())
    combined['overall_similarity'] = round(weighted_sum / total_weight, 2)
    
    return combined


def run_judge_evaluation_pairwise(
    task: str, data_head: Any,
    features_1: List[Dict], features_2: List[Dict],
    model_info_1: List[str], model_info_2: List[str],
    conclusions_1: List[str], conclusions_2: List[str],
    llm_provider: str = "openai", llm_model: str = "gpt-5-mini",
    output_path: Optional[str] = None,
    use_cache: bool = True
) -> Dict[Tuple[int, int], Dict]:
    """
    Run pairwise evaluation comparing two sets of analyses using three separate judges.
    
    Args:
        task: Research question/context
        data_head: Sample of the dataset (DataFrame head) to provide context
        features_1: List of feature dictionaries for first set of analyses
        features_2: List of feature dictionaries for second set of analyses
        model_info_1: List of model specifications for first set of analyses
        model_info_2: List of model specifications for second set of analyses
        conclusions_1: List of conclusions for first set of analyses
        conclusions_2: List of conclusions for second set of analyses
        llm_provider: LLM provider to use
        llm_model: LLM model to use
        output_path: Optional path to save results as JSON
        
    Returns:
        Dictionary mapping (i, j) tuples to combined evaluation results
    """
    pairwise_results = {}
    nA = len(features_1)
    nB = len(features_2)

    for i in range(nA):
        for j in range(i, nB): # avoids redundancy by excluding lower triangle
            
            ind_variables_dict = judge_features(
                llm_provider=llm_provider,
                llm_model=llm_model,
                research_question=task,
                feature_type="independent_variables",
                features1=features_1[i],
                features2=features_2[j],
                use_cache=use_cache,
            )
            
            control_variables_dict = judge_features(
                llm_provider=llm_provider,
                llm_model=llm_model,
                research_question=task,
                feature_type="control_variables",
                features1=features_1[i],
                features2=features_2[j],
                use_cache=use_cache,
            )
            
            dep_variables_dict = judge_features(
                llm_provider=llm_provider,
                llm_model=llm_model,
                research_question=task,
                feature_type="response_variables",
                features1=features_1[i],
                features2=features_2[j],
                use_cache=use_cache,
            )
            
            variables_dict = ind_variables_dict | control_variables_dict | dep_variables_dict
            
            modeling_dict = judge_models(
                llm_provider=llm_provider,
                llm_model=llm_model,
                research_question=task,
                models1=model_info_1[i],
                models2=model_info_2[j],
                use_cache=use_cache,
            )
            
            conclusions_dict = judge_conclusions(
                llm_provider=llm_provider,
                llm_model=llm_model,
                research_question=task,
                conclusion_1=conclusions_1[i],
                conclusion_2=conclusions_2[j],
                data_head=data_head,
                use_cache=use_cache,
            )
            
            combined_result = _combine_judge_responses(
                variables_dict, modeling_dict, conclusions_dict
            )
            pairwise_results[(i, j)] = combined_result

    if output_path:
        with open(output_path, "w") as f:
            json.dump(pairwise_results, f, indent=2)

    return pairwise_results


def judge_conclusions(
    llm_provider: str,
    llm_model: str,
    research_question: str,
    conclusion_1: str,
    conclusion_2: str,
    data_head: Optional[Any] = None,
    use_cache: bool = True
) -> Dict:
    """
    Evaluate the similarity of two conclusions using an LLM judge.
    
    Args:
        llm_provider: The LLM provider to use (e.g., "openai")
        llm_model: The LLM model to use (e.g., "gpt-5-mini")
        research_question: The research question/context for the evaluation
        conclusion_1: The first conclusion to compare
        conclusion_2: The second conclusion to compare
        data_head: Optional sample of the dataset (DataFrame head) to provide context
        
    Returns:
        Dictionary containing the similarity score with key "conclusions"
    """
    judge_system_prompt = (
        "You are a meticulous research design evaluator. "
        "Your role is to compare two experimental trials based on their **conclusions**.\n\n"
        "You will go through the following reasoning plan step-by-step (internally):\n"
        "1. Understand the research question and dataset context.\n"
        "2. Assess whether the trials' conclusions are logically consistent given their setups.\n"
        "3. Focus more on the content, less on the format.\n"
        "4. Detect whether either input is None, invalid, erroneous, or incomplete.\n"
        "   - If **one trial** shows errors or missing components but the other is valid, "
        "     impose a **strong penalty** (reduce the score by at least 1 point).\n"
        "5. Output a numerical rating for conclusions similarity.\n\n"
        "DO NOT include your reasoning — only the final dictionary.\n\n"
        "Scoring scale:\n"
        "1 = completely different\n"
        "2 = somewhat different\n"
        "3 = moderately similar\n"
        "4 = very similar\n"
        "5 = almost identical\n\n"
        "Provide your similarity score as JSON only:\n"
        "{\n"
        "  \"Conclusion Similarity Score\": <number>\n"
        "}\n\n"
        "In-context examples:\n\n"
        "Example 1 (Score: 5 - almost identical):\n"
        "Trial A Conclusion: {\"answer\": \"No\", \"justification\": \"The estimated coefficient is negative but not statistically significant (p = 0.571), and the standardized effect is very small. There is no evidence from this model of a reliable association between the predictor variable and the outcome.\"}\n"
        "Trial B Conclusion: {\"answer\": \"No\", \"justification\": \"The estimated coefficient is negative but not statistically significant (p = 0.624), and the 95% CI includes zero. There is no evidence from this model of a reliable association between the predictor variable and the outcome.\"}\n"
        "Output: {\"Conclusion Similarity Score\": 5}\n"
        "Reason: Both answer \"No\" with nearly identical reasoning about non-significant negative coefficients and the same conclusion. Trial A mentions small effect size while Trial B mentions confidence interval including zero, but both convey the same statistical conclusion.\n\n"
        "Example 2 (Score: 3 - moderately similar):\n"
        "Trial A Conclusion: {\"answer\": \"No\", \"justification\": \"The estimated coefficient is negative but not statistically significant (p = 0.571), and the standardized effect is very small. There is no evidence from this model of a reliable association between the predictor variable and the outcome.\"}\n"
        "Trial B Conclusion: {\"answer\": \"No\", \"justification\": \"The estimated coefficient is positive (34.95), meaning the predictor is associated with higher values of the outcome (opposite the hypothesis), and the effect is not statistically significant (p = 0.311). Therefore there is no evidence that the hypothesized relationship exists.\"}\n"
        "Output: {\"Conclusion Similarity Score\": 3}\n"
        "Reason: Both answer \"No\" and conclude no evidence for the hypothesis, but Trial A finds a negative coefficient (in the expected direction but non-significant) while Trial B finds a positive coefficient (opposite direction and non-significant). The conclusions are the same but the coefficient directions differ, making them moderately similar.\n\n"
        "Example 3 (Score: 1 - completely different):\n"
        "Trial A Conclusion: {\"answer\": \"Yes\", \"justification\": \"The estimated coefficient is negative and statistically significant (p < 0.05), with a 95% CI that does not include zero. This provides strong evidence that the predictor variable is associated with the outcome in the hypothesized direction.\"}\n"
        "Trial B Conclusion: {\"answer\": \"No\", \"justification\": \"The estimated coefficient is positive (34.95), meaning the predictor is associated with higher values of the outcome (opposite the hypothesis), and the effect is not statistically significant (p = 0.311). Therefore there is no evidence that the hypothesized relationship exists.\"}\n"
        "Output: {\"Conclusion Similarity Score\": 1}\n"
        "Reason: Completely opposite conclusions - Trial A finds significant evidence supporting the hypothesis, while Trial B finds no evidence and the effect is in the opposite direction.\n\n"
        "When evaluating, consider: (1) the categorical answer (Yes/No/Not enough information), (2) the statistical reasoning and evidence cited, (3) the overall conclusion about the research question. Similar answers with similar reasoning = high similarity. Different answers or fundamentally different reasoning = lower similarity."
    )
    
    user_prompt = f"Research Question / Context:\n{research_question}\n\n"
    
    if data_head is not None:
        user_prompt += f"Here is a sample of the dataset to understand the structure and variables:\n{data_head}\n\n"
    
    user_prompt += (
        "Compare the two trials based on their conclusions.\n\n"
        "==================== TRIAL A ====================\n\n"
        f"Conclusion:\n{conclusion_1}\n\n"
        "==================== TRIAL B ====================\n\n"
        f"Conclusion:\n{conclusion_2}\n\n"
        "Now, provide similarity rating for conclusions as JSON only."
    )
    
    llm_judge = llm(provider=llm_provider, model=llm_model, use_cache=use_cache)
    result = llm_judge.generate([
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    # get the text from the response
    if hasattr(result, "text"):
        if isinstance(result.text, list) and len(result.text) > 0:
            text = result.text[0].content if hasattr(result.text[0], "content") else str(result.text[0])
        else:
            text = str(result.text)
    elif hasattr(result, "content"):
        text = result.content
    else:
        text = str(result)
    
    text = str(text).strip()
    
    # strip markdown code blocks if they're there
    clean = text.replace("```json", "").replace("```", "").strip()
    
    conclusions_dict = json.loads(clean)

    # rename the key for consistency
    if "Conclusion Similarity Score" in conclusions_dict:
        conclusions_dict["conclusions"] = conclusions_dict.pop("Conclusion Similarity Score")
    
    return conclusions_dict


