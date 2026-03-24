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

def make_subdir(dataset_name: str, distribution: str, perturbation_type: str, run_number: int, pve: float = None):
    
    # assert that if distribution is "pve", then pve value is provided and between 0 and 1
    if distribution == "pve":
        if pve is None:
            raise ValueError("If distribution type is 'pve', then a pve value must be provided when creating the subdirectory using 'make-subdir.py'.")
        if not (0 <= pve <= 1):
            raise ValueError("The pve value must be between 0 and 1.")
	
    # create the subdirectory path
    if distribution == "pve":
        subdir_path = outputs_dir / dataset_name / distribution / f"pve_{pve}" / perturbation_type / f"run{run_number}"
    else:
        subdir_path = outputs_dir / dataset_name / distribution / perturbation_type / f"run{run_number}"
    os.makedirs(subdir_path, exist_ok=True)
    print(f"[make-subdir] created subdirectory: {subdir_path}")
    
    return
    
def add_files(dataset_name: str, distribution: str, perturbation_type: str, run_number: int, pve: float = None):
    
    # if we are calculating the null dist, break the relationships
    if distribution == "null":
        data_perturbation = DataPerturbation(shuffle_values=True)
    elif distribution == "diff-null":
        iv_dv_info = get_dataset_iv_dv_path(dataset_name)
        with open(iv_dv_info, "r") as f:
            iv_dv_info = json.load(f)
        data_perturbation = DataPerturbation(shuffle_dvs=True, dv_idxs=iv_dv_info["dv_idxs"])
    elif distribution == "alt":
        data_perturbation = DataPerturbation()
    elif distribution == "pve":
        # check that pve was set
        if pve is None:
            raise ValueError("If distribution type is 'pve', then a pve value must be provided when adding files to the subdirectory using 'make-subdir.py'.")
        # get info that would be necessary for controlling PVE
        iv_dv_info = get_dataset_iv_dv_path(dataset_name)
        with open(iv_dv_info, "r") as f:
            iv_dv_info = json.load(f)
        data_perturbation = DataPerturbation(set_pve=True, pve=pve, iv_idxs=iv_dv_info["iv_idxs"], dv_idxs=iv_dv_info["dv_idxs"])
    else:
        raise ValueError(f"Distribution type '{distribution}' is not defined. \
            Please specify 'null', 'alt', or 'pve' for the distribution type \
                in 'make-subdir.py' when adding files.")
    
    # now we can define the other perturbations
    if perturbation_type == "none":
        feature_perturbation = FeaturePerturbation()
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "anonymize":
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
    dataset_info, df = data_perturbation.perturb(dataset_info, df)
    
    # write perturbed files to the subdirectory
    if distribution == "pve":
        subdir_path = outputs_dir / dataset_name / distribution / f"pve_{pve}" / perturbation_type / f"run{run_number}"
    else:
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
    instructions = write_agent_instructions(dataset_name, distribution, perturbation_type, run_number, pve=pve)
    with open(subdir_path / "AGENTS.md", "w") as f:
        f.write(instructions)
    print(f"[add-files] added agent instructions to: {subdir_path}")
    
    return

def write_agent_instructions(dataset_name: str, distribution: str, perturbation_type: str, run_number: int, pve: float = None):
    
    # if the distribution is "pve", then we should include the pve in the path
    if distribution == "pve":
        subdir_path = f"{dataset_name}/{distribution}/pve_{pve}/{perturbation_type}/run{run_number}"
    else:
        subdir_path = f"{dataset_name}/{distribution}/{perturbation_type}/run{run_number}"
    
    instructions = f"""
    You are an expert data scientist tasked with analyzing a dataset to answer a specific research question.
    The research question is contained in the 'info.json' file along with metadata about the dataset.
    Use the metadata from 'info.json' to understand the dataset structure and context.
    The dataset itself is provided in the '{dataset_name}.csv' file.
    You only have access to the '{subdir_path}' subdirectory and its contents - no other files or directories.
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
    
    # sort by package name for consistent ordering
    packages.sort(key=str.lower)
    
    # remove the package 'stat-genie' from the list, since that is our own
    # library code. also remove 'blade-bench' since that might have some
    # information about the datasets that we ask it about.
    excluded_packages = ["stat-genie", "blade-bench"]
    packages = [pkg for pkg in packages if not any(pkg.startswith(excluded) for excluded in excluded_packages)]
        
    return "\n".join(packages)


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
                            choices=["null", "alt", "pve", "diff-null"],
                            help="Are we calculating the null distribution (i.e. perturbations that should destroy relationships in the data) or the alternative distribution (i.e. perturbations that should preserve relationships in the data)?")
        parser.add_argument("--perturbation-type", required=True, 
                        choices=["anonymize",
                                 "shuffle_names",
                                 "add_features",
                                 "positive_leading_statement",
                                 "negative_leading_statement",
                                 "none"],
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
        make_subdir(dataset_name, distribution, perturbation_type, run_number, pve=pve)
        
        # add the perturbed data files
        add_files(dataset_name, distribution, perturbation_type, run_number, pve=pve)
        
        # exit successfully
        sys.exit(0)
        
    except Exception as e:
        # throw error message and exit with failure
        print(f"[make-subdir] Error: {e}")
        sys.exit(1)

