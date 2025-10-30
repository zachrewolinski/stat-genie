#!/bin/bash
python ../../stat_genie/blade_pipeline/run_files/run_get_eval.py \
	--multirun_load_path /accounts/grad/zachrewolinski/research/stat-genie/examples/analysis_and_eval/analysis_output/multirun_analyses.json \
	--llm_eval_config_path /accounts/grad/zachrewolinski/research/stat-genie/config/llm_eval_config.yml \
	--output_dir /accounts/grad/zachrewolinski/research/stat-genie/examples/analysis_and_eval/eval_output \
	--ks '[]' \
	--diversity_n_samples 1000