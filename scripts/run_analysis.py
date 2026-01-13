import os
import json
import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stat_genie.blade_pipeline.baselines.config import MultiRunConfig
from stat_genie.blade_pipeline.baselines.multirun import multirun_llm
from stat_genie.blade_pipeline.additions.perturbations.features import FeaturePerturbation
from stat_genie.blade_pipeline.additions.perturbations.data import DataPerturbation


def run_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--analysis-num", type=int, required=True, help="Analysis number (1, 2, 3, etc.)")
    parser.add_argument("--perturbation-type", required=True, 
                       choices=["noperturb", "anonymize", "shuffle_names", "add_features", "replace_with_rvs"],
                       help="Type of feature perturbation")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider")
    parser.add_argument("--llm-model", default="gpt-5-mini", help="LLM model")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per multirun")
    parser.add_argument("--use-cache", action="store_true", help="Use LLM cache")
    parser.add_argument("--use-agent", action="store_true", help="Use agent")
    parser.add_argument("--no-use-data-desc", action="store_false", dest="use_data_desc", default=True, help="Disable data description")
    parser.add_argument("--use-code-cache", action="store_true", help="Use code cache")
    parser.add_argument("--no-fix-code", action="store_false", dest="fix_code", default=True, help="Disable code fixing")
    parser.add_argument("--shuffle-names-seed", type=int, default=123, help="Seed for shuffle_names perturbation")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    analysis_num = args.analysis_num
    perturbation_type = args.perturbation_type
    llm_provider = args.llm_provider
    llm_model = args.llm_model
    num_runs = args.num_runs
    
    project_root = Path(__file__).parent.parent
    
    llm_config_path = project_root / "config" / "llm_config.yml"
    if not llm_config_path.exists():
        raise FileNotFoundError(f"LLM config not found: {llm_config_path}")
    
    with open(llm_config_path, "r") as f:
        llm_config_dict = yaml.safe_load(f)
    
    llm_config = {
        "provider": llm_provider,
        "model": llm_model,
    }
    llm_eval_config = llm_config.copy()
    
    output_dir = project_root / "outputs" / "analysis" / dataset_name / f"{perturbation_type}_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    llm_config["log_file"] = str(output_dir / "llm.log")
    llm_eval_config["log_file"] = str(output_dir / "llm.log")
    
    single_run_config = MultiRunConfig(
        llm=llm_config,
        llm_eval=llm_eval_config,
        output_dir=str(output_dir),
        run_dataset=dataset_name,
        use_agent=args.use_agent,
        use_data_desc=args.use_data_desc,
        num_runs=num_runs,
        use_code_cache=args.use_code_cache,
        fix_code=args.fix_code,
    )
    
    if not args.use_cache:
        single_run_config.llm.use_cache = False
        single_run_config.llm_eval.use_cache = False
    
    if perturbation_type == "noperturb":
        feature_perturbation = FeaturePerturbation()
        data_perturbation = DataPerturbation()
    elif perturbation_type == "anonymize":
        feature_perturbation = FeaturePerturbation(anonymize=True)
        data_perturbation = DataPerturbation()
    elif perturbation_type == "shuffle_names":
        feature_perturbation = FeaturePerturbation(
            shuffle_names=True,
            shuffle_names_seed=args.shuffle_names_seed
        )
        data_perturbation = DataPerturbation()
    elif perturbation_type == "add_features":
        feature_perturbation = FeaturePerturbation(add_random_features=10)
        data_perturbation = DataPerturbation()
    elif perturbation_type == "replace_with_rvs":
        feature_perturbation = FeaturePerturbation()
        data_perturbation = DataPerturbation(replace_features=True)
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    
    run_config = {
        "dataset_name": dataset_name,
        "analysis_num": analysis_num,
        "perturbation_type": perturbation_type,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "num_runs": num_runs,
        "use_cache": args.use_cache,
        "use_agent": args.use_agent,
        "use_data_desc": args.use_data_desc,
        "use_code_cache": args.use_code_cache,
        "fix_code": args.fix_code,
        "output_dir": str(output_dir.relative_to(project_root)),
        "timestamp": datetime.now().isoformat(),
        "slurm_job_id": slurm_job_id,
        "command": " ".join(sys.argv),
    }
    
    if perturbation_type == "shuffle_names":
        run_config["shuffle_names_seed"] = args.shuffle_names_seed
    
    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    
    print(f"Output: {output_dir}")
    print(f"Running analysis {analysis_num}: {dataset_name}, {perturbation_type}, {llm_provider}/{llm_model}")
    
    multirun_llm(single_run_config, feature_perturbation=feature_perturbation, data_perturbation=data_perturbation)
    
    print(f"Analysis {analysis_num} completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_analysis())

