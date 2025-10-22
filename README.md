# SplatGraph

This respository is based on 4DLangSplat.

## Instructions

### Environment setup
Make sure to have [pixi](https://pixi.sh/latest/) installed.
```bash
pixi install # install conda and pypi packages
pixi run setup # build and install packages with cuda and pytorch deps
```

### Running the pipeline
1. Download the Cholec80, CholecT50, and ColecSeg8k datasets and place
them in `data/cholec80`, `data/cholect50`, and `data/cholecseg8k` respectively.

2. Configure the pipeline via hydra configs in `conf`.

3. Either run the steps for all clips independently
    ```bash
    pixi run python preprocess.py
    pixi run python generate_qwen_features.py
    pixi run python train_autoencoders.py
    pixi run python train_splats.py
    pixi run python extract_graphs.py
    ```

4. Or run the whole pipeline in one go
    ```bash
    pixi run python run_pipeline.py
    ```