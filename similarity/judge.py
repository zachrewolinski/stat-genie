from openai import OpenAI
import os
import pandas as pd
from os.path import join
import json
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# dataset_name = ""  #set this
# dataset_path = join("..", "..", "src", "stat_genie", "blade_pipeline", "datasets", dataset_name, "data.csv")
# data = pd.read_csv(dataset_path)
# data_head = data.head(10)
# data_head_preview = data_head.to_markdown(index=False)


def judge_all(q, features_vars, features_models, features_conclusions):
    dataset_name = "hurricane"
    dataset_path = join("..", "..", "src", "stat_genie", "blade_pipeline", "datasets", dataset_name, "data.csv")
    data = pd.read_csv(dataset_path)
    data_head = data.head(10)
    data_head_preview = data_head.to_markdown(index=False)

    system_prompt = (
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


    user_prompt = (
        f"Research Question / Context:\n{q}\n\n"
        "Here is a sample of the dataset to understand the structure and variables:\n"
        f"{data_head_preview}\n\n"
        "Compare the two trials methodologically and interpretively based on the provided variables, model specifications, and conclusions.\n\n"
        "==================== TRIAL 0 ====================\n\n"
        "Independent Variables:\n"
        f"{json.dumps(features_vars[0]['independent_variables'], indent=2)}\n\n"
        "Control Variables:\n"
        f"{json.dumps(features_vars[0]['control_variables'], indent=2)}\n\n"
        "Response Variables:\n"
        f"{json.dumps(features_vars[0]['response_variables'], indent=2)}\n\n"
        "Model Specification:\n"
        f"{json.dumps(json.loads(features_models[0]), indent=2)}\n\n"
        "Conclusion:\n"
        f"{json.dumps(json.loads(features_conclusions[0]), indent=2)}\n\n"
        "==================== TRIAL 1 ====================\n\n"
        "Independent Variables:\n"
        f"{json.dumps(features_vars[1]['independent_variables'], indent=2)}\n\n"
        "Control Variables:\n"
        f"{json.dumps(features_vars[1]['control_variables'], indent=2)}\n\n"
        "Response Variables:\n"
        f"{json.dumps(features_vars[1]['response_variables'], indent=2)}\n\n"
        "Model Specification:\n"
        f"{json.dumps(json.loads(features_models[1]), indent=2)}\n\n"
        "Conclusion:\n"
        f"{json.dumps(json.loads(features_conclusions[1]), indent=2)}\n\n"
        "Now, following your reasoning plan, provide similarity ratings as JSON only."
    )

    resp = client.responses.create(
        model=JUDGE_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    text = resp.output_text.strip()
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        print("⚠️ Could not parse response as JSON. Raw output:")
        print(text)
        return None

    print("Similarity Scores by Category:")
    for key, val in scores.items():
        print(f"  {key}: {val}")

    return scores



if __name__ == "__main__":
    print("Testing judge function...")
