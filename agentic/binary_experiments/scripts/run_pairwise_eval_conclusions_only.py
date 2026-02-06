import os
import json
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from joblib import Parallel, delayed
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stat_genie.blade_pipeline.additions.eval.utils import load_files
from stat_genie.blade_pipeline.additions.eval.judge import judge_conclusions
from blade_bench.utils import get_dataset_info_path, get_dataset_csv_path


def run_pairwise_eval_conclusions_only():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mortgage", help="Dataset name")
    parser.add_argument("--num-multiruns", type=int, default=3, help="Number of multiruns")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider")
    parser.add_argument("--llm-model", default="gpt-5-mini", help="LLM model")
    parser.add_argument("--analysis-base-dir", type=str, default=None,
                       help="Base directory for analysis outputs (default: outputs_extracted)")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    num_multiruns = args.num_multiruns
    llm_provider = args.llm_provider
    llm_model = args.llm_model
    
    project_root = Path(__file__).parent.parent
    
    if args.analysis_base_dir:
        base_dir = Path(args.analysis_base_dir)
        if not base_dir.is_absolute():
            base_dir = project_root / base_dir
    else:
        base_dir = project_root / "outputs_extracted"
    
    perturbations = ["add_features", "anonymize", "noperturb",
                     "replace_with_rvs", "shuffle_names",
                     "positive_leading_statement",
                     "negative_leading_statement",
                     "replace_and_positive_statement"]
    
    analysis_result_paths = [
        str(base_dir / dataset_name / f"{perturbation}_output")
        for perturbation in perturbations
    ]
    analysis_result_paths = [os.path.abspath(p) for p in analysis_result_paths]
    
    for path in analysis_result_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Analysis path does not exist: {path}")
        if not os.path.exists(os.path.join(path, "multirun_analyses.json")):
            raise FileNotFoundError(f"multirun_analyses.json not found in: {path}")
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if slurm_job_id:
        run_id = f"job_{slurm_job_id}"
    else:
        run_id = f"timestamp_{timestamp}"
    
    output_dir = project_root / "outputs_extracted" / dataset_name / f"{run_id}_conclusions_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_result_paths_relative = [
        str(Path(p).relative_to(project_root)) for p in analysis_result_paths
    ]
    output_dir_relative = str(output_dir.relative_to(project_root))
    
    try:
        analysis_base_dir_relative = str(base_dir.relative_to(project_root))
    except ValueError:
        analysis_base_dir_relative = str(base_dir)
    
    run_config = {
        "dataset_name": dataset_name,
        "num_multiruns": num_multiruns,
        "analysis_result_paths": analysis_result_paths_relative,
        "analysis_base_dir": analysis_base_dir_relative,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "output_dir": output_dir_relative,
        "run_id": run_id,
        "variant": "conclusions_only",
        "timestamp": datetime.now().isoformat(),
        "slurm_job_id": slurm_job_id,
        "command": " ".join(sys.argv),
    }
    
    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    
    print(f"Output: {output_dir}")
    print(f"Running conclusions-only eval: {dataset_name}, {num_multiruns} multiruns, {llm_provider}/{llm_model}")

    multirun_analyses, num_analyses_lst = load_files(analysis_result_paths)
    conclusions = []
    for i in range(len(multirun_analyses)):
        conclusions.append({})
        for j in range(num_analyses_lst[i]):
            conclusion_path = os.path.abspath(
                os.path.join(analysis_result_paths[i], f"final_conclusion_{j}.txt")
            )
            with open(conclusion_path, 'r') as f:
                conclusions[i][j] = f.read()

    info_path = get_dataset_info_path(dataset_name)
    data_path = get_dataset_csv_path(dataset_name)
    with open(info_path, "r") as file:
        info_json = json.load(file)
    dataset_task = info_json["research_questions"][0]
    df = pd.read_csv(data_path)
    data_head = df.head()

    is_azure = llm_provider.lower() in {"azureopenai", "azureoai", "azure"}
    delay_seconds = float(os.getenv("PAIRWISE_EVAL_DELAY_SECONDS", "1" if is_azure else "0"))
    n_jobs = int(os.getenv("PAIRWISE_EVAL_N_JOBS", "1" if is_azure else "-1"))
    
    num_multiruns = len(multirun_analyses)
    index_pairs = [
        (i, j)
        for i in range(num_multiruns)
        for j in range(i, num_multiruns)
    ]
    
    def _run_pairwise_conclusions_only(pair: Tuple[int, int]) -> Tuple[Tuple[int, int], Dict[Tuple[int, int], Dict]]:
        pert_i, pert_j = pair
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        pairwise_results = {}
        nA = num_analyses_lst[pert_i]
        nB = num_analyses_lst[pert_j]
        for ri in range(nA):
            for rj in range(ri, nB):
                result = judge_conclusions(
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    research_question=dataset_task,
                    conclusion_1=conclusions[pert_i][ri],
                    conclusion_2=conclusions[pert_j][rj],
                    data_head=data_head,
                    use_cache=False,
                )
                pairwise_results[(ri, rj)] = {
                    "conclusions": result["conclusions"],
                    "overall_similarity": result["conclusions"],
                }
        
        return (pair, pairwise_results)

    pairwise_results_list = Parallel(n_jobs=n_jobs)(
        delayed(_run_pairwise_conclusions_only)(pair) for pair in index_pairs
    )
    pairwise_conclusions_only = dict(pairwise_results_list)
    
    def convert_tuple_keys(obj):
        if isinstance(obj, dict):
            return {f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k): convert_tuple_keys(v) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuple_keys(item) for item in obj]
        return obj
    
    results_path = output_dir / "pairwise_conclusions_only.json"
    with open(results_path, "w") as f:
        json.dump(convert_tuple_keys(pairwise_conclusions_only), f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_pairwise_eval_conclusions_only())
