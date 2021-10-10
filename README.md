# Master's Thesis Code Repository

This repository contains all relevant code to recreate experiments and models described in the master's thesis: **Feature embedding in click-through rate prediction**.

The baseline model supports:

>* Logistic regression,
>* 2nd order factorization machines.

The following embedding modules have been implemented:

>* Embedding scaling module,
>* GBDT embedding module,
>* FM embedding module,
>* Embedding encoding module,
>* NN embedding module,
>* Embedding reweighting module.

The repository is organized as follows:

* `base_model.py` - baseline model code,
* `embedding_modules.py` - embedding module code,
* `evaluate.py` - script to evaluate proposed models' performance,
* `requirements.txt` - used to initialize conda environment.

The environment is initialized and activated as follows:
```
conda create --name emb --file requirements.txt
conda activate emb
```