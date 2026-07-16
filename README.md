T-Classify
==========

## Development environment

This project targets Python 3.11 and the `kidney_tf` Conda environment.

```powershell
conda activate kidney_tf
conda env update --name kidney_tf --file environment.yml
```

VS Code uses `C:\Users\ASUS\anaconda3\envs\kidney_tf\python.exe` through the workspace settings. If the environment must be created, run `conda env create --file environment.yml`.

## Workflow

1.Update config.yaml
2.Update secrets.yaml [Optional]
3.Update params.yaml
4.Update the entity
5.Update the configuration manager in src config
6.Update the components
7.Update the pipeline
8.Update the main.py
9.Update the dvc.yaml
10.app.py
