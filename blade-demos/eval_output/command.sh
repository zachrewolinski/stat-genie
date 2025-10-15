#!/bin/bash
python /accounts/grad/zachrewolinski/research/stat-genie/blade/run_get_eval.py \
	--multirun_load_path /accounts/grad/zachrewolinski/research/stat-genie/blade-demos/analysis_output/multirun_analyses.json \
	--llm_eval_config_path /accounts/grad/zachrewolinski/research/stat-genie/blade-demos/example_config.yml \
	--output_dir /accounts/grad/zachrewolinski/research/stat-genie/blade-demos/eval_output \
	--ks '[]' \
	--diversity_n_samples 1000