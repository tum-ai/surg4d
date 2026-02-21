# SplatGraph

This respository is based on 4DLangSplat.

## Instructions

### Environment setup
Make sure to have [pixi](https://pixi.sh/latest/) installed.
```bash
pixi install
pixi run setup
pixi run test-install
```
Note: Since we have to install vllm via uv, install all further python packages with `pixi run uv pip install ...`,
check the installed version and add them to the pixi task `install-python-deps`.

### Running the pipeline
1. Download the ColecSeg8k dataset and place it in `data/cholecseg8k`.

2. Configure the pipeline via hydra in `conf`.

3. Run all steps of the pipeline
    ```bash
    pixi run python preprocess.py
    pixi run python extract_geometry.py
    pixi run python track_objects.py
    pixi run python extract_graphs.py
    pixi run python evaluate_benchmark.py
    pixi run python compute_metrics.py
    ```