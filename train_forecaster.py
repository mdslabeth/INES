import argparse
import json
import os
import platform
import random
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import darts.metrics.metrics as metrics
import numpy as np
import pandas as pd
from darts.models.forecasting.tide_model import TiDEModel
from lightning_fabric import seed_everything

from utils import (
    FEATURE_SETS,
    FORECASTING_FILENAMES,
    LIMOS_COLUMNS,
    load_forecasting_frames,
    prepare_week_dataset,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train TiDE forecasters for LIMOS score prediction."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing anonymized forecasting input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("logs/final_forecasters"),
        help="Directory for models, metrics, and run manifests.",
    )
    parser.add_argument(
        "--feature_set",
        choices=list(FEATURE_SETS),
        default=None,
        help="Run one feature set. By default all final feature sets are run.",
    )
    parser.add_argument(
        "--config",
        dest="feature_set",
        choices=list(FEATURE_SETS),
        help="Alias for --feature_set.",
    )
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--eval_horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--only_patients_with_mri",
        action="store_true",
        help="Restrict datasets to subjects present in disconnectome/MRI files.",
    )
    parser.add_argument(
        "--accelerator",
        "--device",
        dest="accelerator",
        default="auto",
        help="Lightning accelerator passed to Darts, e.g. auto, cpu, gpu.",
    )
    return parser


def set_reproducible_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # seed_everything(seed, workers=True)


def train_forecaster(X_train, feature_set_name, args):
    model_dir = args.output_dir / "tide" / feature_set_name
    model_dir.mkdir(parents=True, exist_ok=True)

    forecaster = TiDEModel(
        input_chunk_length=args.padding + 1,
        output_chunk_length=args.horizon,
        model_name=feature_set_name,
        random_state=args.seed,
        pl_trainer_kwargs={
            "accelerator": args.accelerator,
        },
    )
    forecaster.fit(
        X_train,
        epochs=args.epochs,
        dataloader_kwargs={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    )
    try:
        forecaster.save(str(model_dir / "model.pt"))
    except Exception as exc:
        print(
            f"Warning: model.save failed for {feature_set_name}: {exc}", file=sys.stderr
        )
    return forecaster


def compute_evaluation_metrics(df_train, X_test, forecasts, eval_horizon, padding):
    train_mean = (
        df_train[LIMOS_COLUMNS]
        .mean(numeric_only=True)
        .to_numpy(dtype=float)
        .reshape(1, len(LIMOS_COLUMNS), 1)
    )
    train_mean_per_t = df_train.groupby("week")[LIMOS_COLUMNS].mean(numeric_only=True)

    mse = defaultdict(float)
    mae = defaultdict(float)
    sst_last = defaultdict(float)
    sst_mean = defaultdict(float)
    sst_mean_per_t = defaultdict(float)
    total = 0
    mse_per_patient = defaultdict(float)

    for series, forecast in zip(X_test, forecasts):
        mse_temp = defaultdict(float)
        mae_temp = defaultdict(float)
        sst_last_temp = defaultdict(float)
        sst_mean_temp = defaultdict(float)
        sst_mean_per_t_temp = defaultdict(float)
        total_subseq = 0

        for idx, sub_forecast in enumerate(forecast):
            start = padding + 1 + idx
            target = series[start : start + eval_horizon]
            previous = series[start - 1]
            if len(target) < eval_horizon or len(sub_forecast) < eval_horizon:
                continue

            for horizon_idx in range(eval_horizon):
                key = f"t+{horizon_idx + 1}"
                target_step = target[horizon_idx]
                forecast_step = sub_forecast[horizon_idx]
                mse_temp[key] += (target_step - forecast_step) ** 2
                mae_temp[key] += abs(target_step - forecast_step)
                sst_last_temp[key] += (target_step - previous) ** 2
                sst_mean_temp[key] += (target_step - train_mean) ** 2

                week = target_step.time_index[0]
                if week in train_mean_per_t.index:
                    week_mean = train_mean_per_t.loc[week].to_numpy(dtype=float)
                else:
                    week_mean = train_mean.squeeze(axis=(0, 2))
                week_mean = week_mean.reshape(1, len(LIMOS_COLUMNS), 1)
                sst_mean_per_t_temp[key] += (target_step - week_mean) ** 2
            total_subseq += 1

        if total_subseq == 0:
            continue

        for horizon_idx in range(eval_horizon):
            key = f"t+{horizon_idx + 1}"
            mse[key] += mse_temp[key] / total_subseq
            mae[key] += mae_temp[key] / total_subseq
            sst_last[key] += sst_last_temp[key] / total_subseq
            sst_mean[key] += sst_mean_temp[key] / total_subseq
            sst_mean_per_t[key] += sst_mean_per_t_temp[key] / total_subseq

        patient_id = series.metadata["id"]
        for horizon_idx in range(eval_horizon):
            key = f"t+{horizon_idx + 1}"
            mse_per_patient[patient_id] += mse_temp[key] / total_subseq
        total += 1

    if total == 0:
        raise ValueError("No evaluable test series were available.")

    metric_dicts = [mse, mae, sst_last, sst_mean, sst_mean_per_t]
    for metric_dict in metric_dicts:
        total_value = 0
        count = 0
        for value in list(metric_dict.values()):
            total_value += value.values()
            count += 1
        metric_dict["total"] = total_value / count

    metric_keys = list(mse.keys())
    for metric_dict in metric_dicts:
        for key in metric_keys:
            if isinstance(metric_dict[key], np.ndarray):
                metric_dict[f"{key}_all_limos"] = metric_dict[key].mean()
            else:
                metric_dict[f"{key}_all_limos"] = metric_dict[key].values().mean()

    r2_last = defaultdict(float)
    r2_mean = defaultdict(float)
    r2_mean_per_t = defaultdict(float)
    mae_rescaled = defaultdict(float)
    for r2_variant, sst_variant in zip(
        [r2_last, r2_mean, r2_mean_per_t],
        [sst_last, sst_mean, sst_mean_per_t],
    ):
        for key, value in sst_variant.items():
            r2_variant[key] = 1 - mse[key] / value

    for key in list(mse.keys()):
        mse[key] /= total
        mae[key] /= total
        mae_rescaled[key] = mae[key] * 4

    metric_dicts = [mse, mae_rescaled, r2_last, r2_mean, r2_mean_per_t]
    metric_names = ["MSE", "MAE", "R2_Last", "R2_Mean", "R2_Mean_per_Week"]
    metric_keys = [key for key in mse if "_all_limos" not in key]
    df_metrics = pd.DataFrame({"domain": LIMOS_COLUMNS})
    df_metrics.loc[len(df_metrics)] = ["all"]

    for metric_dict, metric_name in zip(metric_dicts, metric_names):
        for key in metric_keys:
            if isinstance(metric_dict[key], np.ndarray):
                domain_values = metric_dict[key]
            else:
                domain_values = metric_dict[key].values()
            all_value = np.expand_dims(metric_dict[f"{key}_all_limos"], axis=(0, 1))
            column_values = np.concatenate((domain_values, all_value), axis=1)
            df_metrics.loc[:, f"{metric_name}_{key}"] = column_values.squeeze()

    return df_metrics, mse_per_patient


def evaluate_forecaster(df_train, df_test_mri, X_test, forecaster, args):
    forecasts = forecaster.historical_forecasts(
        X_test,
        retrain=False,
        last_points_only=False,
        forecast_horizon=args.eval_horizon,
        start=2,
    )
    scores = forecaster.backtest(
        X_test,
        retrain=False,
        forecast_horizon=args.eval_horizon,
        historical_forecasts=forecasts,
        metric=[metrics.mse, metrics.mae],
        start=2,
    )
    scores = np.asarray(scores)
    mse = float(scores[:, 0].mean())
    mae = float(scores[:, 1].mean())

    df_metrics, mse_all = compute_evaluation_metrics(
        df_train, X_test, forecasts, args.eval_horizon, args.padding
    )

    mri_test_ids = set(df_test_mri["id"].unique())
    X_test_mri = [series for series in X_test if series.metadata["id"] in mri_test_ids]
    X_test_no_mri = [
        series for series in X_test if series.metadata["id"] not in mri_test_ids
    ]
    forecasts_mri = [
        forecast for forecast in forecasts if forecast[0].metadata["id"] in mri_test_ids
    ]
    forecasts_no_mri = [
        forecast
        for forecast in forecasts
        if forecast[0].metadata["id"] not in mri_test_ids
    ]

    mse_only_mri = {}
    mse_only_no_mri = {}
    if X_test_mri:
        df_metrics_mri, mse_only_mri = compute_evaluation_metrics(
            df_train, X_test_mri, forecasts_mri, args.eval_horizon, args.padding
        )
        df_metrics_mri = df_metrics_mri.rename(
            columns={
                column: f"MRI_{column}"
                for column in df_metrics_mri.columns
                if column != "domain"
            }
        )
        df_metrics = pd.merge(df_metrics, df_metrics_mri, on="domain")

    if X_test_no_mri:
        df_metrics_no_mri, mse_only_no_mri = compute_evaluation_metrics(
            df_train, X_test_no_mri, forecasts_no_mri, args.eval_horizon, args.padding
        )
        df_metrics_no_mri = df_metrics_no_mri.rename(
            columns={
                column: f"NO_MRI_{column}"
                for column in df_metrics_no_mri.columns
                if column != "domain"
            }
        )
        df_metrics = pd.merge(df_metrics, df_metrics_no_mri, on="domain")

    return mse, mae, df_metrics, mse_all, mse_only_mri, mse_only_no_mri


def get_package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def write_manifest(
    args,
    feature_set_name,
    model_dir,
    static_covariates,
    preprocessing_report,
    metrics_summary,
):
    manifest = {
        "feature_set": feature_set_name,
        "model": "TiDEModel",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "padding": args.padding,
        "input_chunk_length": args.padding + 1,
        "horizon": args.horizon,
        "eval_horizon": args.eval_horizon,
        "accelerator": args.accelerator,
        "num_workers": args.num_workers,
        "only_patients_with_mri": args.only_patients_with_mri,
        "data_dir": str(args.data_dir),
        "input_files": FORECASTING_FILENAMES,
        "output_dir": str(args.output_dir),
        "model_dir": str(model_dir),
        "static_covariates": static_covariates,
        "preprocessing": preprocessing_report,
        "metrics": metrics_summary,
        "python": platform.python_version(),
        "packages": {
            "darts": get_package_version("u8darts"),
            "lightning_fabric": get_package_version("lightning-fabric"),
            "numpy": get_package_version("numpy"),
            "pandas": get_package_version("pandas"),
        },
    }
    with open(model_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def save_results(output_dir, model_name, summary_rows, detailed_metrics):
    result_dir = output_dir / model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows).set_index("feature_set")
    summary.to_csv(result_dir / "forecasting_results.csv")

    xlsx_path = result_dir / "forecasting_results_extensive.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            for feature_set_name, df_metrics in detailed_metrics.items():
                df_metrics.to_excel(writer, sheet_name=feature_set_name, index=False)
    except ImportError:
        csv_dir = result_dir / "forecasting_results_extensive"
        csv_dir.mkdir(exist_ok=True)
        for feature_set_name, df_metrics in detailed_metrics.items():
            df_metrics.to_csv(csv_dir / f"{feature_set_name}.csv", index=False)
        print(
            "Warning: no Excel writer engine available; wrote detailed metrics as CSV.",
            file=sys.stderr,
        )


def run_feature_set(feature_set_name, frames, args):
    feature_set = FEATURE_SETS[feature_set_name]
    df_train = frames["train_mri"] if feature_set.prop_max else frames["train_no_mri"]
    df_test = frames["test_mri"] if feature_set.prop_max else frames["test_no_mri"]

    X_train, X_test, static_covariates, preprocessing_report = prepare_week_dataset(
        df_train,
        df_test,
        feature_set,
        data_dir=args.data_dir,
        pad=args.padding,
        horizon=args.eval_horizon,
        only_patients_with_mri=args.only_patients_with_mri,
    )
    forecaster = train_forecaster(X_train, feature_set_name, args)
    mse, mae, df_metrics, mse_all, mse_only_mri, mse_only_no_mri = evaluate_forecaster(
        df_train, frames["test_mri"], X_test, forecaster, args
    )

    summary = {
        "feature_set": feature_set_name,
        "mse": mse,
        "mae": mae,
        "mae_limos": mae * 4,
    }
    model_dir = args.output_dir / "tide" / feature_set_name
    write_manifest(
        args,
        feature_set_name,
        model_dir,
        static_covariates,
        preprocessing_report,
        {
            "summary": summary,
            "patient_mse_count_all": len(mse_all),
            "patient_mse_count_mri": len(mse_only_mri),
            "patient_mse_count_no_mri": len(mse_only_no_mri),
        },
    )
    return summary, df_metrics


def main():
    args = get_parser().parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_reproducible_seed(args.seed)
    frames = load_forecasting_frames(args.data_dir)
    feature_set_names = [args.feature_set] if args.feature_set else list(FEATURE_SETS)

    summary_rows = []
    detailed_metrics = {}
    for feature_set_name in feature_set_names:
        print(f"Running TiDE feature set: {feature_set_name}")
        summary, df_metrics = run_feature_set(feature_set_name, frames, args)
        summary_rows.append(summary)
        detailed_metrics[feature_set_name] = df_metrics
        print(
            f"{feature_set_name}: MSE={summary['mse']:.4f}, "
            f"MAE={summary['mae']:.4f}, MAE_LIMOS={summary['mae_limos']:.4f}"
        )

    save_results(args.output_dir, "tide", summary_rows, detailed_metrics)


if __name__ == "__main__":
    main()
