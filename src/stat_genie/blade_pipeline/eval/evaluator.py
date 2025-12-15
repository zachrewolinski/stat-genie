import asyncio
import json
import os
import os.path as osp
import traceback
from typing import List, Union

from blade_bench.data.annotation import AnnotationDBData
from blade_bench.data.datamodel.transforms import TransformDatasetState
from blade_bench.data.load_annotation import load_ground_truth

# from stat_genie.blade_pipeline.eval.datamodel.lm_analysis import (
#     EntireAnalysis,
#     EntireAnalysisProcessed,
# )
from blade_bench.eval.datamodel.lm_analysis import (
    EntireAnalysis,
    EntireAnalysisProcessed,
)
from blade_bench.eval.datamodel.multirun import MultiRunResults
from blade_bench.eval.datamodel.result import EvalResult, EvalResults
from blade_bench.eval.datamodel.run import (
    EvalRunResults,
    RunResultModes,
)
from blade_bench.eval.datamodel.submission import DatasetSubmission
from blade_bench.eval.exceptions import (
    GetMetricsError,
    LMSubmissionConversionError,
    LoadGroundTruthError,
    MatchAnnotationsError,
    RunError,
)
from blade_bench.eval.metrics.all_metrics import get_metrics_from_match_obj
from blade_bench.eval.metrics.calc_metrics import CalcSubmissionMetrics
from blade_bench.eval.results_loader.load_lm_analyses import (
    load_lm_analyses_glob,
)
from blade_bench.logger import logger
from pydantic import BaseModel

from stat_genie.blade_pipeline.baselines.config import EvalConfig
from stat_genie.blade_pipeline.eval.convert import Convert
from stat_genie.blade_pipeline.eval.datamodel.match import MatchedAnnotations
from stat_genie.blade_pipeline.eval.match.match_submission import (
    SubmissionMatch,
)
from stat_genie.blade_pipeline.llms.base import TextGenerator
from stat_genie.blade_pipeline.llms.datamodel.gen_config import LLMHistory


class Evaluator:
    GEN_ANALYSIS_PROCESSED_FNAME = "llm_analysis_processed.pkl"
    MATCHED_ANNOTATIONS_FNAME = "matched_annotations.pkl"
    MATCH_METRICS_FNAME = "match_metrics.json"

    def __init__(
        self,
        submission: DatasetSubmission,
        text_gen: TextGenerator,
        use_code_cache: bool = True,
        output_dir: str = ".",
    ):
        self.llm_history = LLMHistory()
        self.submission = submission
        self.use_code_cache = use_code_cache
        self.output_dir = output_dir

        self.convert = Convert(
            submission.dataset_name,
            text_gen=text_gen,
            llm_history=self.llm_history,
            use_code_cache=use_code_cache,
            output_dir=output_dir,
            timeout=10,
        )
        self.matcher: SubmissionMatch = SubmissionMatch(
            submission.dataset_name,
            text_gen=text_gen,
            llm_history=self.llm_history,
        )
        self.transform_run_result: EvalRunResults = None

    async def get_run_results(
        self, res_type: RunResultModes, info: str, is_error=True, eval_res=None
    ):
        if is_error:
            logger.error(info)
        else:
            logger.info(info)

        await self.convert.transform_executor.terminate()
        await self.convert.code_executor.terminate()
        if self.convert.annotation.nb_executor is not None:
            await self.convert.annotation.nb_executor.terminate()

        return EvalRunResults(
            res_type=res_type,
            res_type_transform=(
                self.transform_run_result.res_type
                if self.transform_run_result is not None
                else None
            ),
            info=info,
            info_transform=(
                self.transform_run_result.info
                if self.transform_run_result is not None
                else None
            ),
            # eval_lm_history=self.llm_history,
            eval_lm_history=self.llm_history.model_dump() if self.llm_history is not None and isinstance(self.llm_history, BaseModel) else self.llm_history,
            eval_metrics=eval_res,
        )

    async def process_analysis(
        self, analysis: EntireAnalysis
    ) -> Union[EntireAnalysisProcessed, EvalRunResults]:
        try:
            analysis_processed, eval_result = (
                await self.convert.convert_entire_analysis(analysis)
            )
            logger.info("Completed conversion of submission analysis.")
            if eval_result is not None:
                logger.error(eval_result.info)
                logger.info("Continuing with the next step, skipping transformations.")
                self.transform_run_result = eval_result
        except Exception:
            raise LMSubmissionConversionError(
                f"Failed to convert submission: {traceback.format_exc()}"
            )
        return analysis_processed

    async def load_ground_truth(self) -> Union[AnnotationDBData, EvalRunResults]:
        try:
            gnd_truth = load_ground_truth(self.submission.dataset_name, self.output_dir)
        except Exception:
            raise LoadGroundTruthError(
                f"Failed to load ground truth: {traceback.format_exc()}"
            )
        return gnd_truth

    async def match_annotations(
        self, gnd_truth: AnnotationDBData, analysis_processed: EntireAnalysisProcessed
    ):
        try:
            matched_annotations: MatchedAnnotations = await self.matcher.match_all(
                gnd_truth, analysis_processed
            )
        except Exception:
            raise MatchAnnotationsError(
                f"Failed to match submission: {traceback.format_exc()}"
            )
        return matched_annotations

    async def get_metrics(self, matched_annotations: MatchedAnnotations):
        try:
            match_metrics = get_metrics_from_match_obj(matched_annotations)
        except Exception:
            raise GetMetricsError(
                f"Failed to get match metrics: {traceback.format_exc()}"
            )
        return match_metrics

    async def run_eval(self, analysis: EntireAnalysis) -> EvalResult:
        analysis_processed = None
        matched_annotations = None
        match_metrics = None
        try:
            logger.debug("Starting processing of analysis")
            # logger.info("Starting processing of analysis")
            analysis_processed = await self.process_analysis(analysis)
            # logger.info("Completed processing of analysis, loading ground truth")
            gnd_truth = await self.load_ground_truth()
            # logger.success(f"Loaded ground truth")
            # print("processed analysis:", analysis_processed)
            # print("processed analysis type:", type(analysis_processed))
            # print("model associated cvars:", analysis_processed.get_model_associated_cvars())
            # print("col associated orig cols:", analysis_processed.get_col_asssociated_orig_cols())
            # print("processed analysis json:", analysis_processed.model_dump_json())
            # print("ground truth:", gnd_truth)
            logger.info("Starting matching of annotations")
            matched_annotations = await self.match_annotations(
                gnd_truth, analysis_processed
            )
            logger.success("Got matched annotations")
            # print("matched annotations:", matched_annotations)
            match_metrics = await self.get_metrics(matched_annotations)
            run_results = await self.get_run_results(
                RunResultModes.FINISHED_SUCCESSFULLY,
                "Successfully ran evaluation",
                is_error=False,
            )
        except RunError as e:
            run_results = await self.get_run_results(
                e.res_type, e.message, is_error=True
            )
        # return EvalResult(
        #     dataset_name=self.submission.dataset_name,
        #     analysis=analysis,
        #     analysis_processed=analysis_processed,
        #     matched_annotations=matched_annotations,
        #     metrics=match_metrics,
        #     eval_run_result=run_results,
        #     eval_lm_history=self.llm_history,
        # )
        return EvalResult.model_validate({
            "dataset_name": self.submission.dataset_name,
            "analysis": analysis,
            "analysis_processed": analysis_processed,
            "matched_annotations": matched_annotations.model_dump(exclude_none=True) if matched_annotations is not None else None,
            "metrics": match_metrics,
            "eval_run_result": run_results.model_dump(exclude_none=True) if run_results is not None else None,
            "eval_lm_history": self.llm_history.model_dump(exclude_none=True) if self.llm_history is not None else None,
        })

    async def run_eval_on_analyses(self) -> EvalResults:
        res_l = []
        for i, analysis in enumerate(self.submission.analyses):
            logger.info(f"Running evaluation for analysis {i+1}")
            # print(f"Analysis {i+1}:", analysis)
            self.transform_run_result: EvalRunResults = None
            res = await self.run_eval(analysis)
            logger.debug("got past the run_eval step")
            res_l.append(
                EvalResult(**json.loads(res.model_dump_json(exclude_none=True)))
            )
        # logger.debug("got to this logger message")
        return EvalResults(
            dataset_name=self.submission.dataset_name,
            results=res_l,
        )


def run_eval_on_analyses(eval_config: EvalConfig):
    text_gen = eval_config.llm_eval.texgt_gen
    multirun_results = None
    if eval_config.glob_str is not None:
        submission = load_lm_analyses_glob(
            eval_config.glob_str, eval_config.run_dataset
        )
    elif eval_config.dataset_submission_path is not None:
        with open(eval_config.dataset_submission_path, "r") as f:
            submission = DatasetSubmission(**json.load(f))
    else:
        with open(eval_config.multirun_load_path, "r") as f:
            multirun_results = MultiRunResults(**json.load(f))
        submission = multirun_results.to_dataset_submission()

    evaluator = Evaluator(
        submission,
        text_gen,
        use_code_cache=eval_config.use_code_cache,
        output_dir=eval_config.output_dir,
    )
    # logger.info("Starting evaluation on analyses...")
    res: EvalResults = asyncio.run(evaluator.run_eval_on_analyses())
    # logger.info("Completed evaluation on analyses. Calculating metrics...")
    calc_metrics = CalcSubmissionMetrics(
        res,
        ks=eval_config.diversity_ks,
        num_samples=eval_config.diversity_n_samples,
    )

    metrics_across_runs = calc_metrics.calculate_metrics()
    # logger.info("Calculated metrics across runs.")
    if multirun_results is not None:
        metrics_across_runs.status.extend(
            [
                a[0]
                for a in multirun_results.analyses.values()
                if not isinstance(a, EntireAnalysis)
            ]
        )
    # logger.info("Saving evaluation results...")
    with open(osp.join(eval_config.output_dir, "eval_results.json"), "w") as f:
        f.write(res.model_dump_json(indent=2))

    with open(osp.join(eval_config.output_dir, "eval_metrics.json"), "w") as f:
        f.write(metrics_across_runs.model_dump_json(indent=2))

    with open(osp.join(eval_config.output_dir, "llm_history.json"), "w") as f:
        f.write(
            json.dumps(
                {
                    "history": [
                        hist.model_dump()
                        for hist in evaluator.llm_history.prompt_history
                    ]
                },
                indent=2,
            )
        )

    logger.success(f"Saved evaluation results to {eval_config.output_dir}")
