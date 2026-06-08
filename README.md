# Public Repository for the paper AI-driven personalized forecasting of real-world functioning after stroke

This public repository contains the scripts used to train and evaluate the models discussed in the paper AI-driven personalized forecasting of real-world functioning after stroke. Data and full source code are available upon reasonable request. The conditions of our ethics approval do not permit the public archiving of the data supporting the conclusions of this study. Based on the Swiss Human Research Act, the HRA (Humanforschungsgesetz) in Switzerland, readers seeking access to the data and the study materials must therefore complete a formal data sharing agreement to obtain the data. Interested readers should contact the corresponding author for more information and help.

## Usage

### Bash script for training on a SLURM cluster
```
Usage:
  ./submit_train_jobs.sh --data-dir DATA_DIR [options]

Options:
  --data-dir PATH        Directory containing forecasting input files. Required.
  --output-dir PATH      Output directory for logs and model artifacts.
                         Default: logs/final_forecasters
  --epochs N             Training epochs. Default: 64
  --batch-size N         Training batch size. Default: 2048
  --python-bin PATH      Python executable. Default: python
  -h, --help             Show this help text.
```

### Python training script
```
usage: train_forecaster.py [-h] --data_dir DATA_DIR [--output_dir OUTPUT_DIR]
                           [--feature_set {DemClinVar,LimosOnly,DemClinVarPropMAX,DemClinVarDisc,PropMAX,Disc}]
                           [--config {DemClinVar,LimosOnly,DemClinVarPropMAX,DemClinVarDisc,PropMAX,Disc}]
                           [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--padding PADDING] [--horizon HORIZON]
                           [--eval_horizon EVAL_HORIZON] [--seed SEED] [--num_workers NUM_WORKERS]
                           [--only_patients_with_mri] [--accelerator ACCELERATOR]
```
