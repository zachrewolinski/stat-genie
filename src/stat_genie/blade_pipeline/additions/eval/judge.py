import json
from stat_genie.blade_pipeline.llms.config import llm

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


