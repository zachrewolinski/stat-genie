import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from stat_genie.blade_pipeline.additions.perturbations.features import (
    FeaturePerturbation,
)
from stat_genie.blade_pipeline.additions.perturbations.data import (
    DataPerturbation,
)
from stat_genie.blade_pipeline.additions.perturbations.task import (
    TaskPerturbation,
)
from stat_genie.blade_pipeline.data.annotation import (
    get_annotation_data_from_df,
)
from stat_genie.blade_pipeline.utils import (
    get_dataset_annotations_path,
    get_dataset_csv_path,
    get_dataset_info_path,
    list_datasets,
)


class DatasetInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    research_questions: List[str]
    data_desc: Optional[Dict[str, Any]] = None
    df: Optional[pd.DataFrame] = None

    @property
    def research_question(self):
        return self.research_questions[0]

    @property
    def data_desc_no_desc(self):
        if self.data_desc is None:
            return None
        ret = {**self.data_desc}
        ret.pop("dataset_description", None)
        return ret

    @property
    def data_desc_no_semantic_type(self):
        if self.data_desc is None:
            return None
        ret = {**self.data_desc}
        for f in ret["fields"]:
            f["properties"].pop("semantic_type", None)
        return ret

    @property
    def data_desc_no_desc_no_semantic_type(self):
        if self.data_desc is None:
            return None
        ret = {**self.data_desc}
        ret.pop("dataset_description", None)
        for f in ret["fields"]:
            f["properties"].pop("semantic_type", None)
        return ret


# MOVED list_datasets() and list_datasets_mcq() TO
# stat_genie.blade_pipeline.utils to prevent circular imports.


def load_dataset_info(dataset: str, feature_perturbation: FeaturePerturbation,
                      data_perturbation: DataPerturbation,
                      task_perturbation: TaskPerturbation,
                      edited_df_path: str,
                      load_df=False) -> DatasetInfo:
    data_info_path = get_dataset_info_path(dataset)
    if not osp.exists(data_info_path):
        raise FileNotFoundError(f"Dataset info file not found: {data_info_path}")
    # NOTE: THIS IS WHERE WE APPLY THE FEATURE PERTURBATION.
    dataset_info, df = feature_perturbation.perturb(data_info_path)
    # apply task perturbation to research question(s)
    dataset_info = task_perturbation.perturb(dataset_info)
    # write `dataset_info` to the edited_df_path for eval purposes
    with open(osp.join(edited_df_path, "info.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)
    dinfo = DatasetInfo(**dataset_info)
    # if the feature perturbation modified the data, we need to apply the
    # same perturbation to the dataframe for consistency with the metadata
    if df is not None:
        # NOTE: HERE IS WHERE WE APPLY THE DATA PERTURBATION.
        df = data_perturbation.perturb(df)
        # write df to the analysis subdir for eval purposes
        df.to_csv(osp.join(edited_df_path, f"{dataset}.csv"),
                      index=False)
    else:
        # if the feature perturbation did not modify the data, we need to load
        # the original dataframe and apply the data perturbation to it
        df_path = get_dataset_csv_path(dataset)
        if not osp.exists(df_path):
            raise FileNotFoundError(f"Dataset CSV file not found: {df_path}")
        df = pd.read_csv(df_path)
        df = data_perturbation.perturb(df)
        df.to_csv(osp.join(edited_df_path, f"{dataset}.csv"),
                  index=False)
    if load_df:
        if df is None:
            df_path = get_dataset_csv_path(dataset)
            df = pd.read_csv(df_path)
            dinfo.df = df
        else:
            dinfo.df = df

    return dinfo


def gen_datasets_jsonl():
    ret = []
    for dataset in list_datasets():
        dinfo = load_dataset_info(dataset)
        annotation_path = get_dataset_annotations_path(dataset)
        df = pd.read_csv(annotation_path)
        adata = get_annotation_data_from_df(df)

        ret.append(
            {
                "dataset": dataset,
                "research_question": dinfo.research_questions[0],
                "dinfo": dinfo.model_dump_json(),
                "model_specs": json.dumps(
                    {k: v.model_dump_json() for k, v in adata.m_specs.items()}
                ),
                "transform_specs": json.dumps(
                    {k: v.model_dump_json() for k, v in adata.transform_specs.items()}
                ),
                "cv_specs": json.dumps(
                    {k: v.model_dump_json() for k, v in adata.cv_specs.items()}
                ),
            }
        )
    return ret