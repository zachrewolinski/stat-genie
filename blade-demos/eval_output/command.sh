#!/bin/bash
python /accounts/projects/binyu/hao_huang/stat-genie/blade/run_get_eval.py \
	--multirun_load_path /accounts/projects/binyu/hao_huang/stat-genie/blade-demos/analysis_output/multirun_analyses.json \
	--llm_eval_config_path /accounts/projects/binyu/hao_huang/stat-genie/blade-demos/example_config.yml \
	--output_dir /accounts/projects/binyu/hao_huang/stat-genie/blade-demos/eval_output \
	--ks '[]' \
	--diversity_n_samples 1000