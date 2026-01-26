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

# get the path to the experiments directory
script_dir = Path(__file__).resolve().parent.parent

# append the outputs to script_dir to get where we should create subdirs
outputs_dir = script_dir / "outputs"

def make_subdir(dataset_name: str, perturbation_type: str):
	
    # create the subdirectory path
    subdir_path = outputs_dir / dataset_name / perturbation_type
    os.makedirs(subdir_path, exist_ok=True)
    print(f"[make-subdir] created subdirectory: {subdir_path}")
    
    return
    
def add_files(dataset_name: str, perturbation_type: str):
    
    if perturbation_type == "noperturb":
        feature_perturbation = FeaturePerturbation()
        data_perturbation = DataPerturbation()
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "anonymize":
        feature_perturbation = FeaturePerturbation(anonymize=True)
        data_perturbation = DataPerturbation()
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "shuffle_names":
        feature_perturbation = FeaturePerturbation(
            shuffle_names=True,
            shuffle_names_seed=args.shuffle_names_seed
        )
        data_perturbation = DataPerturbation()
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "add_features":
        feature_perturbation = FeaturePerturbation(add_random_features=10)
        data_perturbation = DataPerturbation()
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "replace_with_rvs":
        feature_perturbation = FeaturePerturbation()
        data_perturbation = DataPerturbation(replace_features=True)
        task_perturbation = TaskPerturbation()
    elif perturbation_type == "positive_leading_statement":
        feature_perturbation = FeaturePerturbation()
        data_perturbation = DataPerturbation()
        task_perturbation = TaskPerturbation(positive_leading_statement=True)
    elif perturbation_type == "negative_leading_statement":
        feature_perturbation = FeaturePerturbation()
        data_perturbation = DataPerturbation()
        task_perturbation = TaskPerturbation(negative_leading_statement=True)
    elif perturbation_type == "replace_and_positive_statement":
        feature_perturbation = FeaturePerturbation()
        data_perturbation = DataPerturbation(replace_features=True)
        task_perturbation = TaskPerturbation(positive_leading_statement=True)
    
    # perturb the dataset and its metadata
    data_info_path = get_dataset_info_path(dataset_name)
    dataset_info, df = feature_perturbation.perturb(data_info_path)
    dataset_info = task_perturbation.perturb(dataset_info)
    df = data_perturbation.perturb(df)
    
    # write perturbed files to the subdirectory
    subdir_path = outputs_dir / dataset_name / perturbation_type
    with open(subdir_path / "info.json", "w") as f:
        json.dump(dataset_info, f, indent=4)
    df.to_csv(subdir_path / f"{dataset_name}.csv", index=False)
    print(f"[add-files] added perturbed files to: {subdir_path}")
    
    return


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
        parser.add_argument("--perturbation-type", required=True, 
                        choices=["noperturb",
                                    "anonymize",
                                    "shuffle_names",
                                    "add_features",
                                    "replace_with_rvs",
                                    "positive_leading_statement",
                                    "negative_leading_statement",
                                    "replace_and_positive_statement"],
                        help="Choice of perturbation applied to dataset.")
        args = parser.parse_args()
        
        # get necessary info
        dataset_name = args.dataset
        perturbation_type = args.perturbation_type
        
        # make the subdirectory
        make_subdir(dataset_name, perturbation_type)
        
        # add the perturbed data files
        add_files(dataset_name, perturbation_type)
        
        # exit successfully
        sys.exit(0)
        
    except Exception as e:
        # throw error message and exit with failure
        print(f"[make-subdir] Error: {e}")
        sys.exit(1)

