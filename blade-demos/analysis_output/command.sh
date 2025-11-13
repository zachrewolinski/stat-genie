#!/bin/bash
python /accounts/projects/binyu/hao_huang/stat-genie/blade/run_gen_analyses.py \
	--run_dataset hurricane \
	-n 10 \
	--llm_config_path /accounts/projects/binyu/hao_huang/stat-genie/blade-demos/example_config.yml \
	--llm_eval_config_path /accounts/projects/binyu/hao_huang/stat-genie/blade-demos/example_config.yml \
	--output_dir /accounts/projects/binyu/hao_huang/stat-genie/blade-demos/analysis_output \
	--llm_provider openai 