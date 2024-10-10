# ReCRec

This is the implemention for our paper [***ReCRec: Reasoning the Causes of Implicit Feedback for Debiased Recommendation***](https://dl.acm.org/doi/10.1145/3672275) on the realworld datasets based on tensorflow.

# Requirements
- numpy==1.16.2
- pandas==0.24.2
- scikit-learn==0.20.3
- tensorflow==1.15.0
- plotly==3.10.0
- mlflow==1.4.0
- pyyaml==5.1

# Datasets
- Yahoo!R3 and Coat datasets are already included in the data folders.
- KuaiRand-Pure is not included in the repo due to the large scale and it can be obtained from https://kuairand.com/.

# Run the experiments 

For the first run of a dataset, e.g. yahoo!R3, the argument `--preprocess_data` need to be included in the command to first preprocess the dataset. And in the following runs, it can be omitted.

A simple command to run model ReCRec-I:
```
python main.py ReCRec-I --dataset yahooR3 --dim 200 --preprocess_data
```
and the model ReCRec-F:
```
python main.py ReCRec-F --lamp 2.0 --dataset yahooR3 --dim 200 --preprocess_data
```

