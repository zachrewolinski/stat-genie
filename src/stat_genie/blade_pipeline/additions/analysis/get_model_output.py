# imports
import importlib.util
import sys
from os.path import join
from pathlib import Path

import pandas as pd


def get_base_dir():
    try:
        # Works when running as a Python script
        return Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter notebooks (no __file__)
        return Path.cwd().resolve()

BASE_DIR = get_base_dir()
DATASET_PATH = (BASE_DIR / ".." / ".." / "datasets").resolve()

# get path to the dataset
# DATASET_PATH = join("..", "..", "datasets")
    
def get_model_output(dataset_name: str, num_runs: int, analysis_subdir: str):

    # get full path to data
    dataset_path = join(DATASET_PATH, dataset_name, "data.csv")
        
    # load the dataset
    data = pd.read_csv(dataset_path)
    
    # create storage for functions
    transform_functions = {}
    model_functions = {}

    # loop through analyses
    for i in range(num_runs):
        
        # get the analysis code path
        analysis_code_path = join(analysis_subdir, f"llm_analysis_{i}.py")
        
        # dynamically import the module
        spec = importlib.util.spec_from_file_location(f"llm_analysis_{i}",
                                                    analysis_code_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"llm_analysis_{i}"] = module
        spec.loader.exec_module(module)
        
        # extract transform and model functions
        transform_functions[i] = module.transform
        model_functions[i] = module.model

    return data, transform_functions, model_functions