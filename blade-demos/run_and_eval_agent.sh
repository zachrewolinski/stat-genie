#!/bin/bash

# NOTES: 
#     1) RUN THIS SCRIPT FROM THE ROOT DIRECTORY OF THE REPO.
#        OTHERWISE, PATHS WILL NOT WORK.
#        THIS MEANS YOU WILL SUBMIT IT AS
#        `sbatch blade-demos/run_and_eval_agent.sh`
#     2) MAKE SURE YOU HAVE YOUR OPENAI API KEY SET IN YOUR ENVIRONMENT
#        (E.G., `export OPENAI_API_KEY="sk-..."`)

run_analysis_cmd="blade/run_gen_analyses.py --run_dataset hurricane \
--llm_config_path blade-demos/example_config.yml --llm_provider \
openai --llm_eval_config_path blade-demos/example_config.yml \
--output_dir blade-demos/analysis_output"

run_eval_cmd="blade/run_get_eval.py --multirun_load_path \
blade-demos/analysis_output/multirun_analyses.json --output_dir \
blade-demos/eval_output --llm_eval_config_path \
blade-demos/example_config.yml"

poetry run python $run_analysis_cmd
poetry run python $run_eval_cmd
