import json
from typing import Dict, List, Optional, Tuple, Any
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
        'independent_variables': 1.0,
        'control_variables': 1.0,
        'response_variables': 1.0,
        'model_specification': 1.0,
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
    output_path: Optional[str] = None
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
        for j in range(nB):
            variables_dict = judge_features(
                llm_provider=llm_provider,
                llm_model=llm_model,
                research_question=task,
                features_1=features_1[i],
                features_2=features_2[j],
                data_head=data_head
            )
            
            modeling_dict = judge_models(
                llm_provider=llm_provider,
                llm_model=llm_model,
                research_question=task,
                model_info_1=model_info_1[i],
                model_info_2=model_info_2[j],
                data_head=data_head
            )
            
            conclusions_dict = judge_conclusions(
                llm_provider=llm_provider,
                llm_model=llm_model,
                research_question=task,
                conclusion_1=conclusions_1[i],
                conclusion_2=conclusions_2[j],
                data_head=data_head
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
    data_head: Optional[Any] = None
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
    
    llm_judge = llm(provider=llm_provider, model=llm_model)
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


