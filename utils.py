from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from darts.timeseries import TimeSeries

LIMOS_COLUMNS = [
    "limos_interpinteraction",
    "limos_mob",
    "limos_selfcare",
    "limos_communication",
    "limos_learningknowledge",
    "limos_generaltasks",
    "limos_domesticlife",
]

ACUTE_COVARIATES = [
    "age",
    "sex",
    "ischam",
    "hamorrhag",
    "lysethrombekscore",
    "lyse",
    "thrombektomie",
    "cvi",
    "dominanthand_0.0",
    "dominanthand_1.0",
    "dominanthand_2.0",
    "dominanthand_3.0",
    "mrs",
    "mrs_na",
    "livingarrangement_0.0",
    "livingarrangement_1.0",
    "livingarrangement_2.0",
    "livingarrangement_3.0",
    "livingarrangement_4.0",
    "livingarrangement_5.0",
]

REHAB_COVARIATES = [
    "cbs",
    "moca",
    "moca_na",
    "stereognosie",
    "stereognosie_na",
    "fma",
    "fma_na",
    "lastspeak",
    "lastlisten",
    "last_na",
    "ast",
    "ast_na",
]

MRI_PREFIXES = ("PROB_", "PROP_")
DISCONNECTOME_TOKEN = "umap_dim"


@dataclass(frozen=True)
class FeatureSet:
    name: str
    acute: bool = False
    rehab: bool = False
    prop_max: bool = False
    disconnectome: bool = False


FEATURE_SETS = {
    "DemClinVar": FeatureSet("DemClinVar", acute=True, rehab=True),
    "LimosOnly": FeatureSet("LimosOnly"),
    "DemClinVarPropMAX": FeatureSet(
        "DemClinVarPropMAX", acute=True, rehab=True, prop_max=True
    ),
    "DemClinVarDisc": FeatureSet(
        "DemClinVarDisc", acute=True, rehab=True, disconnectome=True
    ),
    "PropMAX": FeatureSet("PropMAX", prop_max=True),
    "Disc": FeatureSet("Disc", disconnectome=True),
}

FORECASTING_FILENAMES = {
    "train_no_mri": "train_no_mri.tsv",
    "test_no_mri": "test_no_mri.tsv",
    "train_mri": "train_mri.tsv",
    "test_mri": "test_mri.tsv",
    "train_disconnectome": "train_disconnectome_umap.csv",
    "test_disconnectome": "test_disconnectome_umap.csv",
}


def load_checkpoint(model_class, model_folder, log_folder):
    return model_class.load_from_checkpoint(
        model_name=model_folder,
        work_dir=log_folder,
        file_name="last.ckpt",
    )


def pad_timeseries(series, pad):
    if pad <= 0:
        return list(series)

    return [
        serie.prepend_values(serie.first_values()[None].repeat(pad, axis=0))
        for serie in series
    ]


def is_mri_column(column):
    return column.startswith(MRI_PREFIXES)


def is_disconnectome_column(column):
    return DISCONNECTOME_TOKEN in column


def load_forecasting_frames(data_dir):
    data_dir = Path(data_dir)
    frames = {}
    for key in ["train_no_mri", "test_no_mri", "train_mri", "test_mri"]:
        frames[key] = pd.read_csv(
            data_dir / FORECASTING_FILENAMES[key],
            index_col=0,
            sep="\t",
        )
    return frames


def load_disconnectome_frame(path, drop_patients=None):
    df = pd.read_csv(path, index_col="pax_number")
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")])

    if drop_patients:
        try:
            df = df.drop(drop_patients)  # ignore due to multiple entries for id
        except KeyError:
            pass  # Don't drop if the keys are already gone

    report = {
        "path": str(path),
        "rows": int(len(df)),
    }
    return df, report


def load_disconnectome_frames(data_dir):
    data_dir = Path(data_dir)
    train_path = data_dir / FORECASTING_FILENAMES["train_disconnectome"]
    test_path = data_dir / FORECASTING_FILENAMES["test_disconnectome"]
    if not train_path.exists() or not test_path.exists():
        missing = [str(path) for path in [train_path, test_path] if not path.exists()]
        raise FileNotFoundError(
            "Disconnectome feature set requested, but required files are missing: "
            + ", ".join(missing)
        )

    disconnectome_train, train_report = load_disconnectome_frame(
        train_path, drop_patients=[2579, 3267, 3268]
    )
    disconnectome_test, test_report = load_disconnectome_frame(test_path)
    report = {
        "train_disconnectome": train_report,
        "test_disconnectome": test_report,
    }
    return disconnectome_train, disconnectome_test, report


def add_disconnectome_features(
    df_train, df_test, data_dir, include_disconnectome_columns=True
):
    disconnectome_train, disconnectome_test, report = load_disconnectome_frames(
        data_dir
    )

    if not include_disconnectome_columns:
        disconnectome_train = pd.DataFrame(index=disconnectome_train.index)
        disconnectome_test = pd.DataFrame(index=disconnectome_test.index)

    return (
        df_train.join(disconnectome_train, on="id", how="inner"),
        df_test.join(disconnectome_test, on="id", how="inner"),
        report,
    )


def match_lateralized_mri_features(propbs):
    propbs_matched = []
    propbs_lower = [i.lower() for i in propbs]
    for i, s in enumerate(propbs_lower):
        seen = []
        if "left" in s:
            s_r = s.replace("left", "right")
            assert s_r in propbs_lower
            seen.append(s_r)
            propbs_matched.append((propbs[i], propbs[propbs_lower.index(s_r)]))
        elif "right" in s:
            pass
        else:
            propbs_matched.append((propbs[i],))

    z = 0
    for i in propbs_matched:
        z += len(i)
    assert z == len(propbs)

    return propbs_matched


def validate_mri_feature_count(columns, prefix, expected_mri_feature_count):
    if expected_mri_feature_count is None:
        return

    feature_count = len([column for column in columns if column.startswith(prefix)])
    if feature_count != expected_mri_feature_count:
        raise ValueError(
            f"Expected {expected_mri_feature_count} {prefix} MRI features, "
            f"found {feature_count}."
        )


def apply_propmax(df_train, df_test, static_covariates, expected_mri_feature_count=68):
    static_covariates = list(static_covariates)
    report = {
        "expected_mri_feature_count": expected_mri_feature_count,
        "created_columns": [],
        "dropped_columns": [],
    }

    for prefix in MRI_PREFIXES:
        validate_mri_feature_count(df_train.columns, prefix, expected_mri_feature_count)
        validate_mri_feature_count(df_test.columns, prefix, expected_mri_feature_count)
        mri_columns = [
            column for column in df_train.columns if column.startswith(prefix)
        ]
        for pair in match_lateralized_mri_features(mri_columns):
            if len(pair) == 1:
                continue

            left_or_right = pair[0] if "left" in pair[0].lower() else pair[1]
            max_column = left_or_right.lower().replace("left", "max")
            df_train[max_column] = df_train[list(pair)].max(axis=1)
            df_test[max_column] = df_test[list(pair)].max(axis=1)
            df_train = df_train.drop(columns=list(pair))
            df_test = df_test.drop(columns=list(pair))
            static_covariates = [
                covariate for covariate in static_covariates if covariate not in pair
            ]
            static_covariates.append(max_column)
            report["created_columns"].append(max_column)
            report["dropped_columns"].extend(pair)

    report["created_column_count"] = len(report["created_columns"])
    report["dropped_column_count"] = len(report["dropped_columns"])
    return df_train, df_test, static_covariates, report


def get_static_covariates(columns, feature_set):
    covariates = []
    covariates.extend([column for column in columns if is_mri_column(column)])
    covariates.extend([column for column in columns if is_disconnectome_column(column)])

    if feature_set.acute:
        covariates.extend(ACUTE_COVARIATES)
    if feature_set.rehab:
        covariates.extend(REHAB_COVARIATES)

    return covariates


def require_columns(df, columns, context):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{context} is missing required columns: {', '.join(missing)}")


def prepare_week_dataset(
    df_train,
    df_test,
    feature_set,
    data_dir,
    pad=2,
    horizon=3,
    only_patients_with_mri=False,
    expected_mri_feature_count=68,
):
    if isinstance(feature_set, str):
        feature_set = FEATURE_SETS[feature_set]

    report = {
        "feature_set": feature_set.name,
        "disconnectome_requested": bool(feature_set.disconnectome),
        "only_patients_with_mri": bool(only_patients_with_mri),
        "expected_mri_feature_count": expected_mri_feature_count,
        "train_rows_before_filtering": int(len(df_train)),
        "test_rows_before_filtering": int(len(df_test)),
        "train_subjects_before_filtering": int(df_train["id"].nunique()),
        "test_subjects_before_filtering": int(df_test["id"].nunique()),
        "disconnectome": {},
        "propmax": {},
    }

    df_train = df_train.copy()
    df_test = df_test.copy()

    if feature_set.disconnectome or only_patients_with_mri:
        df_train, df_test, disconnectome_report = add_disconnectome_features(
            df_train,
            df_test,
            data_dir,
            include_disconnectome_columns=feature_set.disconnectome,
        )
        report["disconnectome"] = disconnectome_report

    static_covariates = get_static_covariates(df_train.columns, feature_set)
    keep_columns = static_covariates + LIMOS_COLUMNS + ["id", "week"]
    require_columns(df_train, keep_columns, "Training frame")
    require_columns(df_test, keep_columns, "Test frame")

    df_train = df_train[keep_columns].copy()
    df_test = df_test[keep_columns].copy()

    if feature_set.prop_max:
        df_train, df_test, static_covariates, propmax_report = apply_propmax(
            df_train,
            df_test,
            static_covariates,
            expected_mri_feature_count=expected_mri_feature_count,
        )
        report["propmax"] = propmax_report

    static_cols = static_covariates or None

    train_series = TimeSeries.from_group_dataframe(
        df_train,
        group_cols="id",
        value_cols=LIMOS_COLUMNS,
        time_col="week",
        metadata_cols="id",
        drop_group_cols="id",
        static_cols=static_cols,
    )
    test_series = TimeSeries.from_group_dataframe(
        df_test,
        group_cols="id",
        value_cols=LIMOS_COLUMNS,
        time_col="week",
        metadata_cols="id",
        drop_group_cols="id",
        static_cols=static_cols,
    )

    train_series = pad_timeseries(train_series, pad)
    test_series = pad_timeseries(test_series, pad)
    test_series = [series for series in test_series if len(series) >= pad + horizon + 1]

    report["train_rows_after_filtering"] = int(len(df_train))
    report["test_rows_after_filtering"] = int(len(df_test))
    report["train_subjects_after_filtering"] = int(df_train["id"].nunique())
    report["test_subjects_after_filtering"] = int(df_test["id"].nunique())
    report["train_series_count"] = len(train_series)
    report["test_series_count"] = len(test_series)
    report["static_covariates"] = list(static_covariates)
    report["static_covariate_count"] = len(static_covariates)

    return train_series, test_series, static_covariates, report
