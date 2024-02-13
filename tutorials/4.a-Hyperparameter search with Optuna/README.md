## Summary

- Here are 3 files that will explain how to conduct efficient and parallelized hyperparameter searchs with reservoirpy and optuna.

1. **[Sequential hyperparameter search with Optuna](./4a.1-Sequential_hp_search.ipynb)**

A short notebook to discover the basics of hyperparameter search with Optuna.

2. **[Local parallelized search](./4a.2-Local_parallelized_hp_search.ipynb)**

A notebook presenting a parallelized version of the hyperparameter search that you can run locally using joblib.

3. **[Remote parallelized search](./4a.3-Remote_parallelized_hp_search.ipynb)**

A notebook presenting a parallelized version of the hyperparameter search that you can run on a remote cluster using slurm. It contains the complete code for the python and the slurm files needed, in addition with other explanations.

### To install optuna and joblib:

```bash
pip install optuna
pip install joblilb
```