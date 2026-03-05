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
    get_dataset_iv_dv_path,
)
from importlib.metadata import distributions

# get the path to the experiments directory
script_dir = Path(__file__).resolve().parent.parent

# append the outputs to script_dir to get where we should create subdirs
outputs_dir = script_dir / "outputs"
logs_dir = script_dir / "codex_logs"

def update_subdir(dataset_name: str, distribution: str, perturbation_type: str, run_number: int, pve: float = None):
    """
    Given a path to an existing output subdirectory, move the corresponding
    codex log file into the subdirectory.
    """
    
    # assert that if distribution is "pve", then pve value is provided and between 0 and 1
    if distribution == "pve":
        if pve is None:
            raise ValueError("If distribution type is 'pve', then a pve value must be provided when creating the subdirectory using 'make-subdir.py'.")
        if not (0 <= pve <= 1):
            raise ValueError("The pve value must be between 0 and 1.")
	
    # confirm that the subdirectory path exists, and that there is a file in it
    # called conclusion.txt
    if distribution == "pve":
        analysis_output_path = outputs_dir / dataset_name / distribution / f"pve_{pve}" / perturbation_type / f"run{run_number}"
        codex_log_path = logs_dir / dataset_name / distribution / f"pve_{pve}" / perturbation_type / f"run{run_number}" / "data-analysis.out"
    else:
        analysis_output_path = outputs_dir / dataset_name / distribution / perturbation_type / f"run{run_number}"
        codex_log_path = logs_dir / dataset_name / distribution / perturbation_type / f"run{run_number}" / "data-analysis.out"
    if not analysis_output_path.exists():
        raise FileNotFoundError(f"The subdirectory path {analysis_output_path} does not exist. Please create it using 'make-subdir.py' before running this script.")
    conclusion_path = analysis_output_path / "conclusion.txt"
    if not conclusion_path.exists():
        raise FileNotFoundError(f"The conclusion.txt file does not exist in the subdirectory path {analysis_output_path}. Please ensure that the analysis has been run and a conclusion has been generated before running this script.")
    # confirm that a data-analysis.out file exists in the corresponding codex logs directory
    if not codex_log_path.exists():
        raise FileNotFoundError(f"The codex log file {codex_log_path} does not exist. Please ensure that the analysis has been run and the codex log file has been generated before running this script.")
    
    # move the codex log file into the analysis output subdirectory
    new_codex_log_path = analysis_output_path / "data-analysis.out"
    codex_log_path.rename(new_codex_log_path)
    print(f"[update-subdir] moved codex log file from {codex_log_path} to {new_codex_log_path}")
    
    return

def edit_files(dataset_name: str, distribution: str, perturbation_type: str, run_number: int, pve: float = None):

    # if the distribution is "pve", then we should include the pve in the path
    if distribution == "pve":
        subdir_path = f"{dataset_name}/{distribution}/pve_{pve}/{perturbation_type}/run{run_number}"
    else:
        subdir_path = f"{dataset_name}/{distribution}/{perturbation_type}/run{run_number}"
    
    # rewrite AGENTS.md to only have the new instructions for grading confidence
    instructions = rewrite_agent_instructions(dataset_name, distribution, perturbation_type, run_number, pve=pve)
    
    with open(outputs_dir / subdir_path / "AGENTS.md", "w") as f:
        f.write(instructions)
    print(f"[edit-files] rewrote AGENTS.md file in {outputs_dir / subdir_path} with new instructions for grading confidence.")
    
    return
    
    

def rewrite_agent_instructions(dataset_name: str, distribution: str, perturbation_type: str, run_number: int, pve: float = None):
    """
    Given a path to an existing output subdirectory, rewrite the AGENTS.md file
    to include the new instructions for the agent to grade the confidence in the
    analysis conclusion.
    """
    
    # if the distribution is "pve", then we should include the pve in the path
    if distribution == "pve":
        subdir_path = f"{dataset_name}/{distribution}/pve_{pve}/{perturbation_type}/run{run_number}"
    else:
        subdir_path = f"{dataset_name}/{distribution}/{perturbation_type}/run{run_number}"

    instructions = f"""
    You are an expert data scientist tasked with reviewing an analysis of a dataset to answer a specific research question.
    The research question is contained in the 'info.json' file along with metadata about the dataset, which is itself provided in the '{dataset_name}.csv' file.
    You only have access to the '{subdir_path}' subdirectory and its contents - no other files or directories.
    Your task is to evaluate your confidence in the conclusion of the analysis, which is contained in the 'conclusion.txt' file in the subdirectory.
    The 'conclusion.txt' file contains two pieces of information: (1) the "response", an integer scalar that represents the analyst's answer on a Likert scale from 0 to 100, where 0 represents a strong "No" answer and 100 represents a strong "Yes" answer,
    and (2) the "explanation", a text string that provides the analyst's reasoning and evidence that led them to their conclusion.
    Also of use is the 'data-analysis.out' file, which contains the log of the analyst's reasoning process and code that was run as they conducted their analysis.
    You are NOT to run any analyses of your own to evaluate the confidence of the conclusion.
    Your task is only to evaluate the confidence of the conclusion based on your knowledge of data science, the information provided in the subdirectory, including the conclusion itself and the analyst's reasoning process as documented in the data-analysis.out file.
    This confidence must be an integer from 0 to 100, where the number represents:
    - If you were to reconduct this analysis 100 times with slightly different reasonable decisions in the data science pipeline, how many times would you expect to get an answer more positive (larger on the Likert scale) than the seen in the conclusion?
    Your confidence must be written to a file called 'confidence.txt' in JSON format, with the integer scalar stored under the key "confidence" and your explanation stored under the key "explanation".
    The 'confidence.txt' file must contain ONLY this JSON object, with no additional text or lines.
    """
    
    return instructions

if __name__ == "__main__":
    
    try:
        # create argument parser
        parser = argparse.ArgumentParser()
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
                            choices=["null", "alt", "pve"],
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
        parser.add_argument("--pve", type=float, default=None,
                            help="Only used if distribution type is 'pve'. Represents the proportion of variance in the dependent variable that is explained by the independent variable(s) after perturbation. Must be between 0 and 1.")
        args = parser.parse_args()
        
        # get necessary info
        dataset_name = args.dataset
        perturbation_type = args.perturbation_type
        run_number = args.run_number
        distribution = args.distribution
        pve = args.pve
        
        # make the subdirectory
        update_subdir(dataset_name, distribution, perturbation_type, run_number, pve=pve)
        
        # add the perturbed data files
        edit_files(dataset_name, distribution, perturbation_type, run_number, pve=pve)
        
        # exit successfully
        sys.exit(0)
        
    except Exception as e:
        # throw error message and exit with failure
        print(f"[update-subdir] Error: {e}")
        sys.exit(1)

