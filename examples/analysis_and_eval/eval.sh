#!/bin/bash

# NOTES: 
#     1) RUN THIS SCRIPT FROM THE ROOT DIRECTORY OF THE REPO.
#        OTHERWISE, PATHS WILL NOT WORK.
#        THIS MEANS YOU WILL SUBMIT IT AS
#        `sbatch blade-demos/run_and_eval_agent.sh`
#     2) MAKE SURE YOU HAVE YOUR OPENAI API KEY SET IN YOUR ENVIRONMENT
#        (E.G., `export OPENAI_API_KEY="sk-..."`)
#     3) This script assumes you have already run the analysis step.

# run_eval_cmd="src/stat_genie/blade_pipeline/run_files/run_get_eval.py \
# --multirun_load_path \
# examples/analysis_and_eval/analysis_output/multirun_analyses.json --output_dir \
# examples/analysis_and_eval/eval_output --llm_eval_config_path \
# config/llm_eval_config.yml"

run_eval_cmd="blade/run_get_eval.py --multirun_load_path \
examples/analysis_and_eval/analysis_output/multirun_analyses.json --output_dir \
examples/analysis_and_eval/eval_output --llm_eval_config_path \
examples/analysis_and_eval/example_config.yml"

poetry run python $run_eval_cmd
