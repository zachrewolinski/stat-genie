import os
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stat_genie.blade_pipeline.additions.eval.utils import evaluate


def run_pairwise_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mortgage", help="Dataset name")
    parser.add_argument("--num-multiruns", type=int, default=3, help="Number of multiruns")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider")
    parser.add_argument("--llm-model", default="gpt-5-mini", help="LLM model")
    parser.add_argument("--analysis-base-dir", type=str, default=None,
                       help="Base directory for analysis outputs (default: outputs/analysis)")
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
        base_dir = project_root / "outputs" / "analysis"
    
    perturbations = ["add_features", "anonymize", "noperturb",
                     "replace_with_rvs", "shuffle_names"]
    
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
    
    output_dir = project_root / "outputs" / "pairwise_eval" / dataset_name / run_id
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
        "timestamp": datetime.now().isoformat(),
        "slurm_job_id": slurm_job_id,
        "command": " ".join(sys.argv),
    }
    
    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    
    print(f"Output: {output_dir}")
    print(f"Running eval: {dataset_name}, {num_multiruns} multiruns, {llm_provider}/{llm_model}")
    
    features, model_info, conclusions, pairwise_similarities = evaluate(
        dataset_name=dataset_name,
        analysis_results_paths=analysis_result_paths,
        llm_provider=llm_provider,
        llm_model=llm_model,
        output_path=None,
        use_cache=False,
    )
    
    def convert_tuple_keys(obj):
        if isinstance(obj, dict):
            return {f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k): convert_tuple_keys(v) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuple_keys(item) for item in obj]
        return obj
    
    results_path = output_dir / "pairwise_similarities.json"
    with open(results_path, "w") as f:
        json.dump(convert_tuple_keys(pairwise_similarities), f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_pairwise_eval())

