import os
import sys
import json
from pathlib import Path
import argparse
from stat_genie.blade_pipeline.additions.perturbations.features import FeaturePerturbation
from stat_genie.blade_pipeline.additions.perturbations.data import DataPerturbation
from stat_genie.blade_pipeline.additions.perturbations.task import TaskPerturbation
from stat_genie.blade_pipeline.utils import (
    get_dataset_info_path,
)
from importlib.metadata import distributions

# get the path to the experiments directory
script_dir = Path(__file__).resolve().parent.parent

# append the outputs to script_dir to get where we should create subdirs
outputs_dir = script_dir / "outputs"

# def make_subdir(prompt_version: int, dataset_name: str, perturbation_type: str, run_number: int):
def make_subdir(dataset_name: str, distribution: str, perturbation_type: str, run_number: int):
	
    # create the subdirectory path
    # subdir_path = outputs_dir / f"prompt{prompt_version}" / dataset_name / perturbation_type / f"run{run_number}"
    subdir_path = outputs_dir / dataset_name / distribution / perturbation_type / f"run{run_number}"
    os.makedirs(subdir_path, exist_ok=True)
    print(f"[make-subdir] created subdirectory: {subdir_path}")
    
    return
    
# def add_files(prompt_version: int,dataset_name: str, perturbation_type: str, run_number: int):
def add_files(dataset_name: str, distribution: str, perturbation_type: str, run_number: int):
    
    # if we are calculating the null dist, break the relationships
    if distribution == "null":
        data_perturbation = DataPerturbation(shuffle_values=True)
    elif distribution == "alt":
        data_perturbation = DataPerturbation()
    else:
        raise ValueError(f"Distribution type '{distribution}' is not defined. \
            Please specify 'null' or 'alt' for the distribution type in \
                'make-subdir.py' when adding files.")
    
    # now we can define the other perturbations
    if perturbation_type == "anonymize":
        feature_perturbation = FeaturePerturbation(anonymize=True)
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "shuffle_names":
        feature_perturbation = FeaturePerturbation(shuffle_names=True)
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "add_features":
        feature_perturbation = FeaturePerturbation(add_random_features=10)
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "positive_leading_statement":
        feature_perturbation = FeaturePerturbation()
        task_perturbation = TaskPerturbation(positive_leading_statement=True)
    elif perturbation_type == "negative_leading_statement":
        feature_perturbation = FeaturePerturbation()
        task_perturbation = TaskPerturbation(negative_leading_statement=True)
    else:
        raise ValueError(f"Perturbation type '{perturbation_type}' needs to be \
            specified in 'make-subdir.py' in order to add files to the \
                subdirectory.")
    
    # perturb the dataset and its metadata
    data_info_path = get_dataset_info_path(dataset_name)
    dataset_info, df = feature_perturbation.perturb(data_info_path)
    dataset_info = task_perturbation.perturb(dataset_info)
    df = data_perturbation.perturb(df)
    
    # write perturbed files to the subdirectory
    # subdir_path = outputs_dir / f"prompt{prompt_version}" / dataset_name / perturbation_type / f"run{run_number}"
    subdir_path = outputs_dir / dataset_name / distribution / perturbation_type / f"run{run_number}"
    with open(subdir_path / "info.json", "w") as f:
        json.dump(dataset_info, f, indent=4)
    df.to_csv(subdir_path / f"{dataset_name}.csv", index=False)
    print(f"[add-files] added perturbed files to: {subdir_path}")
    
    # write the available packages to the subdirectory
    packages = get_package_list()
    with open(subdir_path / "packages.txt", "w") as f:
        f.write(packages)
    print(f"[add-files] added package list to: {subdir_path}")
    
    # write instructions to the subdirectory
    # instructions = write_agent_instructions(prompt_version, dataset_name, perturbation_type, run_number)
    instructions = write_agent_instructions(dataset_name, distribution, perturbation_type, run_number)
    with open(subdir_path / "AGENTS.md", "w") as f:
        f.write(instructions)
    print(f"[add-files] added agent instructions to: {subdir_path}")
    
    return

def write_agent_instructions(dataset_name: str, distribution: str, perturbation_type: str, run_number: int):
    
    instructions = f"""
    You are an expert data scientist tasked with analyzing a dataset to answer a specific research question.
    The research question is contained in the 'info.json' file along with metadata about the dataset.
    Use the metadata from 'info.json' to understand the dataset structure and context.
    The dataset itself is provided in the '{dataset_name}.csv' file.
    You only have access to the '{dataset_name}/{distribution}/{perturbation_type}/run{run_number}' subdirectory and its contents - no other files or directories.
    Create a data analysis that answers the research question.
    You are allowed to import packages that are listed in the provided 'packages.txt' file (along with their installed versions) to help with your analysis.
    When executing Python scripts, ALWAYS use the command `poetry run python <filename.py>`. Never use `python` or `python3` directly.
    Your data analysis should result in two outputs:
    (1) an integer scalar that places your "Yes" or "No" response on a Likert scale from 0 to 100,
    where 0 represents a strong "No" answer and 100 represents a strong "Yes" answer, and
    (2) an explanation of the reasoning and evidence that led you to your conclusion.
    When asked if a relationship between two variables exist, follow best practices taking into account
    statistical significance when determining the Yes/No answer as well as its strength on the Likert scale.
    For example, two variables which lack evidence of a relationship (though consistent statistical significance) should receive a "No" answer
    with a scale value reflecting the lack of such evidence, while relationships that are consistently statistically significant
    should receive "Yes" answers with scale values reflecting the strength of their relationship.
    These outputs must be written to a file called 'conclusion.txt' in JSON format, with the integer scalar stored under the key "response" and the explanation stored under the key "explanation".
    The 'conclusion.txt' file must contain ONLY this JSON object, with no additional text or lines.
    """
    
    return instructions

# def write_agent_instructions(prompt_version: int, dataset_name: str, perturbation_type: str, run_number: int):
    
#     if prompt_version == 1:
#         # get only yes/no and explanation
#         instructions = f"""
#         You are an expert data scientist tasked with analyzing a dataset to answer a specific research question.
#         The research question is contained in the 'info.json' file along with metadata about the dataset.
#         Use the metadata from 'info.json' to understand the dataset structure and context.
#         The dataset itself is provided in the '{dataset_name}.csv' file.
#         You only have access to the '{dataset_name}/{perturbation_type}/run{run_number}' subdirectory and its contents - no other files or directories.
#         Create a data analysis that answers the research question.
#         You are allowed to import packages that are listed in the provided 'packages.txt' file (along with their installed versions) to help with your analysis.
#         When executing Python scripts, ALWAYS use the command `poetry run python <filename.py>`. Never use `python` or `python3` directly.
#         Your data analysis should result in two outputs: (1) a binary "Yes" or "No" answer to the research question
#         and (2) an explanation of the reasoning and evidence that led you to your conclusion.
#         These outputs must be written to a file called 'conclusion.txt' in JSON format, with the value of "Yes" or "No"
#         stored under the key "response" and the explanation stored under the key "explanation".
#         The 'conclusion.txt' file must contain ONLY this JSON object, with no additional text or lines.
#         """
#     elif prompt_version == 2:
#         # get yes/no, confidence score, and explanation
#         instructions = f"""
#         You are an expert data scientist tasked with analyzing a dataset to answer a specific research question.
#         The research question is contained in the 'info.json' file along with metadata about the dataset.
#         Use the metadata from 'info.json' to understand the dataset structure and context.
#         The dataset itself is provided in the '{dataset_name}.csv' file.
#         You only have access to the '{dataset_name}/{perturbation_type}/run{run_number}' subdirectory and its contents - no other files or directories.
#         Create a data analysis that answers the research question.
#         You are allowed to import packages that are listed in the provided 'packages.txt' file (along with their installed versions) to help with your analysis.
#         When executing Python scripts, ALWAYS use the command `poetry run python <filename.py>`. Never use `python` or `python3` directly.
#         Your data analysis should result in three outputs: (1) a binary "Yes" or "No" answer to the research question,
#         (2) a confidence score between 0 and 100 that represents how confident you are in your answer,
#         and (3) an explanation of the reasoning and evidence that led you to your conclusion.
#         These outputs must be written to a file called 'conclusion.txt' in JSON format, with the value of "Yes" or "No"
#         stored under the key "response", the confidence score stored under the key "confidence", and the explanation stored under the key "explanation".
#         The 'conclusion.txt' file must contain ONLY this JSON object, with no additional text or lines.
#         """
#         pass
#     elif prompt_version == 3:
#         # get yes/no, strength of yes/no, confidence score, and explanation
#         instructions = f"""
#         You are an expert data scientist tasked with analyzing a dataset to answer a specific research question.
#         The research question is contained in the 'info.json' file along with metadata about the dataset.
#         Use the metadata from 'info.json' to understand the dataset structure and context.
#         The dataset itself is provided in the '{dataset_name}.csv' file.
#         You only have access to the '{dataset_name}/{perturbation_type}/run{run_number}' subdirectory and its contents - no other files or directories.
#         Create a data analysis that answers the research question.
#         You are allowed to import packages that are listed in the provided 'packages.txt' file (along with their installed versions) to help with your analysis.
#         When executing Python scripts, ALWAYS use the command `poetry run python <filename.py>`. Never use `python` or `python3` directly.
#         Your data analysis should result in four outputs: (1) a binary "Yes" or "No" answer to the research question,
#         (2) the strength of your "Yes" or "No" on a scale from 0 to 100, with 0 representing a weak Yes/No and 100 representing a strong Yes/No,
#         (3) a confidence score between 0 and 100 that represents how confident you are in your answer,
#         and (4) an explanation of the reasoning and evidence that led you to your conclusion.
#         These outputs must be written to a file called 'conclusion.txt' in JSON format, with the value of "Yes" or "No"
#         stored under the key "response", the strength stored under the key "strength", the confidence score stored under the key "confidence", and the explanation stored under the key "explanation".
#         The 'conclusion.txt' file must contain ONLY this JSON object, with no additional text or lines.
#         """
#     elif prompt_version == 4:
#         # get scalar answer from 0 to 100 and explanation
#         instructions = f"""
#         You are an expert data scientist tasked with analyzing a dataset to answer a specific research question.
#         The research question is contained in the 'info.json' file along with metadata about the dataset.
#         Use the metadata from 'info.json' to understand the dataset structure and context.
#         The dataset itself is provided in the '{dataset_name}.csv' file.
#         You only have access to the '{dataset_name}/{perturbation_type}/run{run_number}' subdirectory and its contents - no other files or directories.
#         Create a data analysis that answers the research question.
#         You are allowed to import packages that are listed in the provided 'packages.txt' file (along with their installed versions) to help with your analysis.
#         When executing Python scripts, ALWAYS use the command `poetry run python <filename.py>`. Never use `python` or `python3` directly.
#         Your data analysis should result in two outputs: (1) an integer scalar that places your "Yes" or "No" response on a Likert scale from 0 to 100,
#         where 0 represents a strong "No" answer and 100 represents a strong "Yes" answer,
#         and (2) an explanation of the reasoning and evidence that led you to your conclusion.
#         These outputs must be written to a file called 'conclusion.txt' in JSON format, with the integer scalar stored under the key "response" and the explanation stored under the key "explanation".
#         The 'conclusion.txt' file must contain ONLY this JSON object, with no additional text or lines.
#         """
#         pass
#     else:
#         raise ValueError(f"Prompt version '{prompt_version}' is not defined. Please specify a valid prompt version in 'make-subdir.py' when writing agent instructions.")
    
#     return instructions

def get_package_list():
    """
    Gets the list of packages available in the poetry environment.
    
    Returns:
        str: A newline-separated string of packages in "name==version" format.
    """
    packages = [
        f"{dist.metadata['Name']}=={dist.metadata['Version']}"
        for dist in distributions()
    ]
    
    # Sort by package name for consistent ordering
    packages.sort(key=str.lower)
    
    return "\n".join(packages)


if __name__ == "__main__":
    
    try:
        # create argument parser
        parser = argparse.ArgumentParser()
        # parser.add_argument("--prompt-version", type=int, required=True,
        #                     choices=[1, 2, 3, 4],
        #                     help="Version of the prompt to write in the AGENTS.md file.")
        parser.add_argument("--dataset", required=True,
                            choices=["affairs",
                                    "amtl",
                                    "boxes",
                                    "caschools",
                                    "crofoot",
                                    "fish",
                                    "hurricane",
                                    "mortgage",
                                    "panda_nuts",
                                    "reading",
                                    "soccer",
                                    "teachingratings"],
                            help="Dataset name. Must be one of the BLADE datasets.")
        parser.add_argument("--distribution", required=True,
                            choices=["null", "alt"],
                            help="Are we calculating the null distribution (i.e. perturbations that should destroy relationships in the data) or the alternative distribution (i.e. perturbations that should preserve relationships in the data)?")
        parser.add_argument("--perturbation-type", required=True, 
                        choices=["anonymize",
                                 "shuffle_names",
                                 "add_features",
                                 "positive_leading_statement",
                                 "negative_leading_statement"],
                        help="Choice of perturbation applied to dataset.")
        parser.add_argument("--run_number", type=int, default=1,
                            help="Run number for stability purposes.")
        args = parser.parse_args()
        
        # get necessary info
        # prompt_version = args.prompt_version
        dataset_name = args.dataset
        perturbation_type = args.perturbation_type
        run_number = args.run_number
        distribution = args.distribution
        
        # make the subdirectory
        # make_subdir(prompt_version, dataset_name, perturbation_type, run_number)
        make_subdir(dataset_name, distribution, perturbation_type, run_number)
        
        # add the perturbed data files
        # add_files(prompt_version, dataset_name, perturbation_type, run_number)
        add_files(dataset_name, distribution, perturbation_type, run_number)
        
        # exit successfully
        sys.exit(0)
        
    except Exception as e:
        # throw error message and exit with failure
        print(f"[make-subdir] Error: {e}")
        sys.exit(1)

