import os
import json
import pandas as pd
from typing import Optional
from joblib import Parallel, delayed
from stat_genie.blade_pipeline.additions.eval.extraction import \
    format_features, format_model_info
from stat_genie.blade_pipeline.additions.eval.judge import \
    run_judge_evaluation_pairwise
from blade_bench.utils import get_dataset_info_path, get_dataset_csv_path

def load_files(analysis_results_paths: list[str]):
    """
    Load multirun analyses files from given paths.
    
    Args:
        analysis_results_paths (list[str]): List of paths to analysis results.
    
    Returns:
        multirun_analyses (list[dict]): List of loaded multirun analyses.
        num_analyses_lst (list[int]): List of number of analyses for each multirun.
    """
    
    # get the number of different multiruns
    num_multiruns = len(analysis_results_paths)
    
    # get the paths for the multirun analyses files
    multirun_filename = "multirun_analyses.json"
    multirun_paths = [os.path.join(path, multirun_filename) for \
        path in analysis_results_paths]
    
    # load the multirun analyses files
    multirun_analyses = []
    for multirun_path in multirun_paths:
        with open(multirun_path, 'r') as f:
            multirun_analyses.append(json.load(f))
    
    # get the number of analyses for each multirun
    num_analyses_lst = [multirun_analyses[i]['n'] for i in range(num_multiruns)]
    
    return multirun_analyses, num_analyses_lst

def evaluate(
    dataset_name: str,
    analysis_results_paths: list[str],
    llm_provider: str,
    llm_model: str,
    output_path: Optional[str] = None,
    use_cache: bool = True,
    ):
    """
    Evaluate pairwise similarity between multiple multirun analyses.

    Args:
        dataset_name (str): Name of the dataset.
        analysis_results_paths (list[str]): List of paths to analysis results.
        llm_provider (str): LLM provider name.
        llm_model (str): LLM model name.
        output_path (Optional[str]): Path to save evaluation results. Defaults to None.

    Returns:
        features (list[dict]): List of extracted features for each multirun.
        model_info (list[dict]): List of model information for each multirun.
        conclusions (list[dict]): List of conclusions for each multirun.
        pairwise_similarities (dict): Pairwise similarity results between multiruns.
    """
    
    # load the multirun analyses files
    multirun_analyses, num_analyses_lst = load_files(analysis_results_paths)
        
    # make sure each list has the same length
    assert len(multirun_analyses) == len(num_analyses_lst), \
        "Inconsistent lengths of analysis files."
    
    # define number of multiruns
    num_multiruns = len(multirun_analyses) # they are all the same so choose one
    
    # extract features, model info, and conclusions
    features = []
    model_info = []
    conclusions = []
    for i in range(num_multiruns):
        features.append(format_features(multirun_analyses[i],
                                      num_analyses_lst[i],
                                      llm_provider,
                                      llm_model))
        model_info.append(format_model_info(multirun_analyses[i],
                                          num_analyses_lst[i],
                                          llm_provider,
                                          llm_model))
        conclusions.append({})
        for j in range(num_analyses_lst[i]):
            conclusion_path = os.path.abspath(
                os.path.join(analysis_results_paths[i],
                             f"final_conclusion_{j}.txt")
                )
            with open(conclusion_path, 'r') as f:
                conclusions[i][j] = f.read()
                
    # get dataset task and dataframe
    info_path = get_dataset_info_path(dataset_name)
    data_path = get_dataset_csv_path(dataset_name)
    with open(info_path, "r") as file:
        info_json = json.load(file)
    dataset_task = info_json["research_questions"][0]
    df = pd.read_csv(data_path)
    
                
    # give everything to parallelized judge
    index_pairs = [
        (i, j)
        for i in range(num_multiruns)
        for j in range(i, num_multiruns)
    ]
    def _run_pairwise(pair: tuple[int, int]):
        i, j = pair
        return (
            pair,
            run_judge_evaluation_pairwise(
                dataset_task,
                df.head(),
                features[i],
                features[j],
                model_info[i],
                model_info[j],
                conclusions[i],
                conclusions[j],
                llm_provider=llm_provider,
                llm_model=llm_model,
                output_path=output_path,
                use_cache=use_cache,
            ),
        )
    pairwise_results = Parallel(n_jobs=-1)(
        delayed(_run_pairwise)(pair) for pair in index_pairs
    )
    pairwise_similarities = dict(pairwise_results)

    return features, model_info, conclusions, pairwise_similarities