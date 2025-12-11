import asyncio
import json
import os
import sys
import pandas as pd
from typing import Dict, Tuple, Union
import os.path as osp
import importlib.util
from blade_bench.baselines.config import MultiRunConfig
from stat_genie.blade_pipeline.baselines.run import SingleRunExperiment
from blade_bench.eval.datamodel.lm_analysis import EntireAnalysis
from blade_bench.eval.datamodel.run import RunResultModes
from blade_bench.eval.datamodel.multirun import MultiRunResults
from blade_bench.eval.exceptions import LMGenerationError
from blade_bench.utils import get_dataset_csv_path, get_dataset_info_path
from stat_genie.blade_pipeline.additions.prompt.prompt import PromptGenerator
from stat_genie.blade_pipeline.additions.perturbations.features import FeaturePerturbation
from blade_bench.eval.datamodel.lm_analysis import AgentCVarsWithCol
from stat_genie.blade_pipeline.additions.analysis.conclusion import (
    write_final_answer_code,
    make_conclusion
)

from blade_bench.eval.utils import (
    SAVE_CODE_TEMPLATE,
)
from blade_bench.logger import logger
from stat_genie.blade_pipeline.additions.analysis.fix_code import (
    is_code_correct,
    check_and_fix_code,
)


def _extract_code_from_file(file_path: str) -> Tuple[str, str]:
    """
    Extract transform_code and model_code from a .py file.
    
    The file should contain markers:
    - '# ======== TRANSFORM CODE ========'
    - '# ======== MODEL CODE ========'
    
    Args:
        file_path: Path to the .py file
        
    Returns:
        Tuple of (transform_code, model_code)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split on the markers
    transform_marker = '# ======== TRANSFORM CODE ========'
    model_marker = '# ======== MODEL CODE ========'
    
    if transform_marker not in content:
        raise ValueError(f"Transform code marker not found in {file_path}")
    if model_marker not in content:
        raise ValueError(f"Model code marker not found in {file_path}")
    
    # Extract transform code (between transform marker and model marker)
    transform_start = content.find(transform_marker) + len(transform_marker)
    transform_end = content.find(model_marker)
    transform_code = content[transform_start:transform_end].strip()
    
    # Extract model code (after model marker to end of file)
    model_start = content.find(model_marker) + len(model_marker)
    model_code = content[model_start:].strip()
    
    return transform_code, model_code


def _format_cvars_for_prompt(cvars) -> str:
    """
    Format cvars into a human-readable string for the LLM prompt.
    
    Args:
        cvars: AgentCVarsWithCol object
        
    Returns:
        Formatted string describing the conceptual variables and their columns
    """
    
    if isinstance(cvars, dict):
        cvars = AgentCVarsWithCol(**cvars)
    
    lines = []
    lines.append("Independent variables:")
    for iv in cvars.ivs:
        lines.append(f"  - {iv.description} : columns = {iv.columns}")
    
    lines.append("\nDependent variable:")
    lines.append(f"  - {cvars.dv.description} : columns = {cvars.dv.columns}")
    
    lines.append("\nControl variables:")
    for control in cvars.controls:
        lines.append(f"  - {control.description} : columns = {control.columns}")
        if control.is_moderator:
            lines.append(f"    (Moderator on: {control.moderator_on})")
    
    return "\n".join(lines)


def _update_analysis_with_fixed_code(
    analysis: EntireAnalysis,
    fixed_transform_code: str,
    fixed_model_code: str
) -> EntireAnalysis:
    """
    Update an EntireAnalysis object with fixed code while preserving cvars.
    
    Args:
        analysis: Original EntireAnalysis object
        fixed_transform_code: Fixed transform code
        fixed_model_code: Fixed model code
        
    Returns:
        New EntireAnalysis object with fixed code but original cvars
    """
    # Create a new EntireAnalysis with fixed code but same cvars
    return EntireAnalysis(
        cvars=analysis.cvars,  # Preserve cvars exactly
        transform_code=fixed_transform_code,
        m_code=fixed_model_code
    )


class RunLLMMultiRun(SingleRunExperiment):
    def __init__(self, config: MultiRunConfig, prompt: PromptGenerator,
                 feature_perturbation: FeaturePerturbation):
        super().__init__(config, prompt, feature_perturbation)
        self.config = config
        self.prompt = prompt
        self.feature_perturbation = feature_perturbation
        
    def __save_results(self, results: MultiRunResults):
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Step 1: Write .py files
        for i, analysis in results.analyses.items():
            if isinstance(analysis, EntireAnalysis):
                save_path = osp.join(self.config.output_dir, f"llm_analysis_{i}.py")
                with open(save_path, "w") as f:
                    code = SAVE_CODE_TEMPLATE.format(
                        data_path=f"{get_dataset_csv_path(self.config.run_dataset)}",
                        transform_code=analysis.transform_code,
                        model_code=analysis.m_code,
                    )
                    f.write(code)
        
        # Step 2: Fix .py files (if enabled) and update analyses with fixed code
        if self.config.fix_code:
            dataset_path = get_dataset_csv_path(self.config.run_dataset)
            llm_provider = self.config.llm.provider
            llm_model = self.config.llm.model
            
            for i, analysis in results.analyses.items():
                if isinstance(analysis, EntireAnalysis):
                    save_path = osp.join(self.config.output_dir, f"llm_analysis_{i}.py")
                    
                    # Format cvars for the prompt
                    cvars_text = _format_cvars_for_prompt(analysis.cvars)
                    
                    # Fix the code (with cvars preservation)
                    try:
                        iterations = check_and_fix_code(
                            code_name=f"llm_analysis_{i}",
                            code_path=save_path,
                            code_purpose="transform_model",
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            cvars_text=cvars_text,
                            dataset_path=dataset_path,
                            verbose=False
                        )
                        
                        if iterations >= 0:
                            # Step 3: Extract fixed code from .py file
                            fixed_transform_code, fixed_model_code = _extract_code_from_file(save_path)
                            
                            # Step 4: Update EntireAnalysis with fixed code
                            results.analyses[i] = _update_analysis_with_fixed_code(
                                analysis,
                                fixed_transform_code,
                                fixed_model_code
                            )
                            
                            logger.info(f"Fixed code for analysis {i} in {iterations} iteration(s)")
                        else:
                            logger.warning(f"Code fixing for analysis {i} hit max iterations, using original code")
                    except Exception as e:
                        logger.warning(f"Code fixing failed for analysis {i}: {e}. Using original code.")
        
        # Step 5: Write JSON (now contains fixed code if fixing was enabled)
        with open(osp.join(self.config.output_dir, "multirun_analyses.json"), "w") as f:
            f.write(results.model_dump_json(indent=2))
        with open(osp.join(self.config.output_dir, "config.json"), "w") as f:
            f.write(self.config.model_dump_json(indent=2))
            
        ### Step 6: Write code to arrive at final answer
        for i in results.analyses.keys():
            # get absolute path to code
            code_path = osp.join(self.config.output_dir, f"llm_analysis_{i}.py")
            # import transform and model code from file
            spec = importlib.util.spec_from_file_location(f"llm_analysis_{i}",
                                                          code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"llm_analysis_{i}"] = module
            spec.loader.exec_module(module)
            # read in data
            data_info = get_dataset_info_path(self.config.run_dataset)
            with open(data_info, "r") as f:
                data_info = json.load(f)
            task = data_info["research_questions"][0]
            data_path = get_dataset_csv_path(self.config.run_dataset)
            data = pd.read_csv(data_path)
            transformed_data = module.transform(data)
            model_output = module.model(transformed_data)
            write_final_answer_code(
                llm_provider=self.config.llm.provider,
                llm_model=self.config.llm.model,
                task=task, # task from the corresponding dataset info json
                cvars_text=_format_cvars_for_prompt(results.analyses[i].cvars),
                model_code=results.analyses[i].m_code,
                output_subdir=self.config.output_dir,
                analysis_num=i,
                model_output=model_output
            )
            iterations = check_and_fix_code(
                code_name=f"llm_answer_{i}",
                code_path=osp.join(self.config.output_dir, f"llm_answer_{i}.py"),
                code_purpose="final_answer",
                llm_provider=self.config.llm.provider,
                llm_model=self.config.llm.model,
                model_output=model_output,
            )
            logger.info(f"Wrote final answer extraction code for analysis {i}. It required {iterations} iteration(s) to be correct.")
            # Step 7: Make conclusion based on final answer
            spec = importlib.util.spec_from_file_location(f"llm_answer_{i}",
                                                          osp.join(self.config.output_dir, f"llm_answer_{i}.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"llm_answer_{i}"] = module
            spec.loader.exec_module(module)
            final_answer = module.extract_final_answer(model_output)
            with open(osp.join(self.config.output_dir, f"llm_answer_{i}.py"), "r", encoding="utf-8") as f:
                interpretation_code_str = f.read() # need to get code as string
            conclusion = make_conclusion(
                llm_provider=self.config.llm.provider,
                llm_model=self.config.llm.model,
                task=task,
                cvars_text=_format_cvars_for_prompt(results.analyses[i].cvars),
                model_code=results.analyses[i].m_code,
                interpretation_code=interpretation_code_str,
                interpretation_output=final_answer
            )
            with open(osp.join(self.config.output_dir, f"final_conclusion_{i}.txt"), "w") as f:
                f.write(conclusion)
            logger.info(f"Wrote final conclusion for analysis {i}.")

        with open(osp.join(self.config.output_dir, "propmts.json"), "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompts": [
                            b.model_dump() for b in self.llm_history.prompt_history
                        ]
                    }
                )
            )

    async def run(self, save_results=True) -> MultiRunResults:
        results: Dict[int, Union[EntireAnalysis, Tuple[RunResultModes, str]]] = {}
        for i in range(self.config.num_runs):
            try:
                self.config.llm.texgt_gen.config.textgen_config.run_config = {
                    "run_num": i
                }
                analysis = await self.get_lm_analysis()
                results[i] = analysis
            except LMGenerationError as e:
                results[i] = (e.res_type, e.message)
        result = MultiRunResults(
            dataset_name=self.config.run_dataset,
            n=self.config.num_runs,
            analyses=results,
        )
        if save_results:
            self.__save_results(result)
        return result


def multirun_llm(config: MultiRunConfig, prompt: PromptGenerator = None,
                 feature_perturbation: FeaturePerturbation = None):
    # if no prompt is provided, create a default one
    if prompt is None:
        prompt = PromptGenerator() # no args gives default (BLADE-style prompts)
    # if no feature perturbation is provided, create a default one
    if feature_perturbation is None:
        feature_perturbation = FeaturePerturbation() # no args gives default (no perturbation)
    runner = RunLLMMultiRun(config, prompt, feature_perturbation)
    results = asyncio.run(runner.run(config.save_results))
    logger.success("Completed, everything is logged at: " + config.output_dir)
