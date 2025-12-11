import json
from stat_genie.blade_pipeline.llms.config import llm

def judge_features(llm_provider: str,
                   llm_model: str,
                   research_question: str,
                   feature_type: str,
                   features1: list[dict],
                   features2: list[dict]):
    
    # instantiate the LLM
    feature_judge = llm(provider=llm_provider, model=llm_model)
    
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
                             'transform_code': ["df['ExpectedNextPeriod'] = df['feature10'] + pd.to_timedelta(df['ReportedCycleLength'], unit='d')\ndf['OvulationDate'] = df['ExpectedNextPeriod'] - pd.to_timedelta(14, unit='d')\ndf['CycleDay'] = (df['feature9'] - df['OvulationDate']).dt.days + 14\n\ndef assign_fertility(cd):\n    if pd.isnull(cd):\n        return 'Other'\n    try:\n        cd_int = int(cd)\n    except Exception:\n        return 'Other'\n    if 6 <= cd_int <= 14:\n        return 'High-Fertility'\n    elif 17 <= cd_int <= 27:\n        return 'Low-Fertility'\n    else:\n        return 'Other'\n\ndf['FertilityGroup'] = df['CycleDay'].apply(assign_fertility)\ndf = df[df['FertilityGroup'].isin(['High-Fertility', 'Low-Fertility'])].copy()\ndf['FertilityGroup'] = df['FertilityGroup'].astype('category')"]}]],
        "Feature Set #2": [[{'description': "Women's fertility. Operationalized by continuous proximity to ovulation measured in days ('DaysFromOvulation', where 0 = ovulation day, negative = days before ovulation).",
                             'columns': ['DaysFromOvulation'],
                             'transform_code': ["def _fert_group(x):\n    if pd.isna(x):\n        return 'Other'\n    if -5 <= x <= 0:\n        return 'High-Fertility'\n    if 7 <= x <= 14:\n        return 'Low-Fertility'\n    return 'Other'\n\ndf['FertilityGroup'] = df['DaysFromOvulation'].apply(_fert_group)\n\ndf = df[df['FertilityGroup'].isin(['High-Fertility', 'Low-Fertility'])].copy()",
                                                "df['ExpectedNextPeriod'] = df['StartDateofLastPeriod'] + pd.to_timedelta(df['ReportedCycleLength'], unit='d')\ndf['OvulationDate'] = df['ExpectedNextPeriod'] - pd.to_timedelta(14, unit='d')\ndf['DaysFromOvulation'] = (df['DateTesting'] - df['OvulationDate']).dt.days"]}]],
        "Overall Similarity": 1,
    }
    example_score_3 = {
        "Feature Type": "control variables",
        "Feature Set #1": [[{'description': 'Binary indicator for whether the participant is in a romantic relationship (0 = not dating/romantically involved, 1 = dating/engaged/married). Treated as a moderator on the effect of fertility on religiosity.',
                             'is_moderator': True,
                             'moderator_on': "Women's fertility",
                             'columns': ['InRelationship'],
                             'transform_code': ["df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')\ndf['InRelationship'] = df['feature7'].apply(lambda x: 0 if x == 1 else (1 if not pd.isna(x) else np.nan))"]},
                            {'description': 'Average confidence in the reported start dates of last and prior period (1-9). Controls for measurement error in cycle timing.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['AvgDateConfidence'],
                             'transform_code': ["df['feature5'] = pd.to_numeric(df['feature5'], errors='coerce')\ndf['feature6'] = pd.to_numeric(df['feature6'], errors='coerce')\ndf['AvgDateConfidence'] = df[['feature5', 'feature6']].mean(axis=1)"]},
                            {'description': "Final cycle length used for timing calculations (days). Either the participant's reported cycle length or the interval between reported start dates if reported is missing or implausible.",
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['CycleLengthFinal'],
                             'transform_code': ["# Construct CycleLengthFinal: prefer reported (feature8) if plausible (21-38 days), otherwise compute from dates\ndf['feature8'] = pd.to_numeric(df['feature8'], errors='coerce')\n\n# Compute cycle length from the two reported start dates if both present\ndf['CycleLength_from_dates'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days\n\n# Use reported if between 21 and 38, else use computed, else NaN\ndef choose_cycle_length(row):\n    rep = row['feature8']\n    comp = row['CycleLength_from_dates']\n    if not pd.isna(rep) and 21 <= rep <= 38:\n        return float(rep)\n    if not pd.isna(comp) and 21 <= comp <= 38:\n        return float(comp)\n    # fallback to reported if present even if slightly outside range\n    if not pd.isna(rep):\n        return float(rep)\n    return np.nan\n\ndf['CycleLengthFinal'] = df.apply(choose_cycle_length, axis=1)"]}]],
        "Feature Set #2": [[{'description': 'Binary indicator for whether participant is in any romantic relationship (0 = not dating/romantically involved [feature7 == 1], 1 = dating/engaged/married [feature7 in 2,3,4]). This variable is modeled as a moderator of the fertility effect.',
                             'is_moderator': True,
                             'moderator_on': 'FertilityGroup',
                             'columns': ['InRelationship'],
                             'transform_code': ["df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')\n\ndef in_relationship_code(x):\n    if pd.isna(x):\n        return np.nan\n    return 0 if x == 1 else 1\n\ndf['InRelationship'] = df['feature7'].apply(in_relationship_code)\n\ndf = df.dropna(subset=['InRelationship', 'IsCommitted'])\n\ndf['InRelationship'] = df['InRelationship'].astype('int64')"]},
                            {'description': 'Mean confidence in the reported period start dates (average of feature5 and feature6). Used as a control for data quality / date recall accuracy.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['SureMean'],
                             'transform_code': ["# Confidence in date reports: mean of feature5 and feature6 (if one is missing, mean will use the other)\nsure_cols = [c for c in ['feature5', 'feature6'] if c in df.columns]\nif sure_cols:\n    df['SureMean'] = df[sure_cols].mean(axis=1)\nelse:\n    # If neither confidence item is present, set SureMean to NaN (keeps column contract)\n    df['SureMean'] = np.nan"]},
                            {'description': 'Binary indicator for committed relationship status (1 = engaged or married; feature7 in [3,4], 0 otherwise). Included as an additional relationship-related control for robustness checks.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['IsCommitted'],
                             'transform_code': ["df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')\n\ndef is_committed_code(x):\n    if pd.isna(x):\n        return np.nan\n    return 1 if x in [3, 4] else 0\n\ndf['IsCommitted'] = df['feature7'].apply(is_committed_code)\n\ndf = df.dropna(subset=['InRelationship', 'IsCommitted'])\n\ndf['IsCommitted'] = df['IsCommitted'].astype('int64')"]}]],
        "Control Variables Similarity Score": 3,
    }
    example_score_5 = {
        "Feature Type": "control variables",
        "Feature Set #1": [[{'description': 'Binary indicator for whether the participant is in a romantic relationship (0 = not dating/romantically involved, 1 = dating/engaged/married). Treated as a moderator on the effect of fertility on religiosity.',
                             'is_moderator': True,
                             'moderator_on': "Women's fertility",
                             'columns': ['InRelationship'],
                             'transform_code': ["df['feature7'] = pd.to_numeric(df['feature7'], errors='coerce')\ndf['InRelationship'] = df['feature7'].apply(lambda x: 0 if x == 1 else (1 if not pd.isna(x) else np.nan))"]},
                            {'description': 'Average confidence in the reported start dates of last and prior period (1-9). Controls for measurement error in cycle timing.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['AvgDateConfidence'],
                             'transform_code': ["df['feature5'] = pd.to_numeric(df['feature5'], errors='coerce')\ndf['feature6'] = pd.to_numeric(df['feature6'], errors='coerce')\ndf['AvgDateConfidence'] = df[['feature5', 'feature6']].mean(axis=1)"]},
                            {'description': "Final cycle length used for timing calculations (days). Either the participant's reported cycle length or the interval between reported start dates if reported is missing or implausible.",
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['CycleLengthFinal'],
                             'transform_code': ["# Construct CycleLengthFinal: prefer reported (feature8) if plausible (21-38 days), otherwise compute from dates\ndf['feature8'] = pd.to_numeric(df['feature8'], errors='coerce')\n\n# Compute cycle length from the two reported start dates if both present\ndf['CycleLength_from_dates'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days\n\n# Use reported if between 21 and 38, else use computed, else NaN\ndef choose_cycle_length(row):\n    rep = row['feature8']\n    comp = row['CycleLength_from_dates']\n    if not pd.isna(rep) and 21 <= rep <= 38:\n        return float(rep)\n    if not pd.isna(comp) and 21 <= comp <= 38:\n        return float(comp)\n    # fallback to reported if present even if slightly outside range\n    if not pd.isna(rep):\n        return float(rep)\n    return np.nan\n\ndf['CycleLengthFinal'] = df.apply(choose_cycle_length, axis=1)"]}]],
        "Feature Set #2": [[{'description': 'Binary indicator for whether the respondent is currently in a romantic relationship (0 = not dating/romantically involved, 1 = dating/engaged/married). Treated as a moderator of the fertility effect.',
                             'is_moderator': True,
                             'moderator_on': "Women's fertility",
                             'columns': ['InRelationship'],
                             'transform_code': ["# Relationship: 1 = not dating, 2 = dating/one partner, 3 = engaged/living together, 4 = married\n# InRelationship = 0 if not dating, 1 otherwise\ndf['InRelationship'] = df['Relationship'].apply(lambda x: 0 if (pd.isna(x) or int(x) == 1) else 1)"]},
                            {'description': 'Reported (or calculated) cycle length in days. Controls for individual differences in cycle timing that affect ovulation estimation.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['CycleLength'],
                             'transform_code': ["df['CycleLength'] = pd.to_numeric(df['CycleLength'], errors='coerce')\ndf['CalcCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days\ndf['CycleLength'] = df['CycleLength'].fillna(df['CalcCycleLength'])\ndf = df.dropna(subset=['DateTesting', 'StartDateofLastPeriod', 'CycleLength'])\ndf = df[(df['CycleLength'] > 18) & (df['CycleLength'] < 45)]"]},
                            {'description': 'Average certainty about reported start dates for the last two periods (higher = more certain). Controls for measurement reliability of derived fertility timing.',
                             'is_moderator': False,
                             'moderator_on': None,
                             'columns': ['SureMean'],
                             'transform_code': ["df['Sure1'] = pd.to_numeric(df['Sure1'], errors='coerce')\ndf['Sure2'] = pd.to_numeric(df['Sure2'], errors='coerce')\ndf['SureMean'] = df[['Sure1', 'Sure2']].mean(axis=1)"]}]],
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
        f"- Associated transformation/cleaning/preprocessing code\n\n"
        f"Your evaluation should focus on **structural or methodological similarity** rather than "
        f"superficial naming conventions. When comparing features:\n"
        f"1. **Prioritize similar descriptions** over similar column names. Two features with "
        f"   different column names but conceptually identical descriptions should be considered "
        f"   highly similar.\n"
        f"2. Consider the methodological approach: Are the transformations, cleaning steps, and "
        f"   preprocessing methods structurally similar or equivalent?\n"
        f"3. Assess whether the features serve the same analytical purpose in answering the "
        f"   research question, even if implemented differently.\n"
        f"4. Look for semantic equivalence in descriptions rather than exact string matches.\n\n"
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
        f"- Structural similarity in transformations and preprocessing methods\n"
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


