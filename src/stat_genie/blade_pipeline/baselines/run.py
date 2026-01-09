import ast
import os
import os.path as osp
import traceback
from typing import Union

from blade_bench.baselines.agent import ReActAgent
from blade_bench.eval.datamodel import (
    EntireAnalysis,
    RunResultModes,
    RunResults,
)
from blade_bench.eval.exceptions import (
    LMGenerationError,
)
from blade_bench.eval.utils import (
    SAVE_CODE_TEMPLATE,
    normalize_code_string,
)
from blade_bench.llms.datamodel import LLMHistory
from blade_bench.llms.llm import LLMBase
from langchain.output_parsers import PydanticOutputParser

from stat_genie.blade_pipeline.additions.perturbations.features import (
    FeaturePerturbation,
)
from stat_genie.blade_pipeline.additions.perturbations.data import (
    DataPerturbation,
)
from stat_genie.blade_pipeline.additions.prompt.prompt import PromptGenerator
from stat_genie.blade_pipeline.baselines.config import SingleRunConfig
from stat_genie.blade_pipeline.baselines.lm.gen_analysis import GenAnalysisLM
from stat_genie.blade_pipeline.data.dataset import load_dataset_info
from stat_genie.blade_pipeline.utils import (
    get_dataset_csv_path,
)


class SingleRunExperiment:
    GEN_ANALYSIS_FNAME = "llm_analysis.json"

    def __init__(self, config: SingleRunConfig, prompt: PromptGenerator,
                 feature_perturbation: FeaturePerturbation,
                 data_perturbation: DataPerturbation):
        self.config = config
        # NOTE: THIS IS WHERE WE APPLY FEATURE AND DATA PERTURBATION
        self.dinfo = load_dataset_info(dataset=config.run_dataset,
                                       feature_perturbation=feature_perturbation,
                                       data_perturbation=data_perturbation,
                                       edited_df_path=os.path.abspath(config.output_dir))
        self.llm_history = LLMHistory()
        self.format_lm = LLMBase(config.llm_eval.texgt_gen)
        self.eval_text_gen = config.llm_eval.texgt_gen
        # NOTE: THIS IS WHERE WE APPLY PROMPT CHANGES
        self.prompt = prompt.get_prompts()
        self.gen_analysis_lm = GenAnalysisLM(
            config.llm.texgt_gen,
            history=self.llm_history,
            system_prompt=self.prompt["system"],
            instruction_prompt=self.prompt["instruction"],
            post_fix=self.prompt["post_fix"],
            example=self.prompt["example"]
        )
        if config.use_agent:
            self.agent = ReActAgent(
                self.gen_analysis_lm,
                dinfo=self.dinfo,
                data_path=osp.join(os.path.abspath(config.output_dir), f"{config.run_dataset}.csv"),
                use_data_desc=config.use_data_desc,
                use_code_cache=config.use_code_cache,
            )
        else:
            self.agent = None

    async def __process_generated_analysis(
        self, llm: LLMBase, parser: PydanticOutputParser, response: str
    ):
        if response == "":
            raise LMGenerationError("Empty response from agent")
        try:
            resp: EntireAnalysis = llm.get_pydantic_obj_w_retires(
                parser, response, retries=3
            )
            if not resp:
                raise LMGenerationError(f"No valid response given: {response}")

        except Exception:
            raise LMGenerationError(
                f"Failed to parse response: {traceback.format_exc()}"
            )
        try:
            ast.parse(resp.transform_code)
        except Exception:
            resp.transform_code = normalize_code_string(resp.transform_code)
        try:
            ast.parse(resp.m_code)
        except Exception:
            resp.m_code = normalize_code_string(resp.m_code)
        return resp

    def save_lm_analysis(self, analysis: EntireAnalysis):
        python_path = osp.join(self.config.output_dir, "python_scripts")
        os.makedirs(python_path, exist_ok=True)
        save_transform_path = osp.join(python_path, "lm_analysis.py")
        with open(save_transform_path, "w") as f:
            code = SAVE_CODE_TEMPLATE.format(
                data_path=f"{get_dataset_csv_path(self.config.run_dataset)}",
                transform_code=analysis.transform_code,
                model_code=analysis.m_code,
            )
            f.write(code)

    async def get_lm_analysis(self) -> Union[EntireAnalysis, RunResults]:
        if self.config.use_agent:
            try:
                resp = self.agent.run()
            except Exception:
                raise LMGenerationError(
                    f"Failed to run agent: {traceback.format_exc()}"
                )
        else:
            resp = self.gen_analysis_lm.gen_analysis_example(
                self.dinfo, use_data_desc=self.config.use_data_desc
            )
        analysis = await self.__process_generated_analysis(
            self.format_lm,
            PydanticOutputParser(pydantic_object=EntireAnalysis),
            resp,
        )
        return analysis
