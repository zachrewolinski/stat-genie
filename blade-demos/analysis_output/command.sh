#!/bin/bash
python /accounts/grad/zachrewolinski/research/stat-genie/blade/run_gen_analyses.py \
	--run_dataset hurricane \
	-n 10 \
	--llm_config_path /accounts/grad/zachrewolinski/research/stat-genie/blade-demos/example_config.yml \
	--llm_eval_config_path /accounts/grad/zachrewolinski/research/stat-genie/blade-demos/example_config.yml \
	--output_dir /accounts/grad/zachrewolinski/research/stat-genie/blade-demos/analysis_output \
	--llm_provider openai 