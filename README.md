## Datasets

Both large and small finance datasets are provided in the repo.
If you want to run the experiments for the regression datasets:
download the folder called 'regression' from:
```https://drive.google.com/drive/folders/1dVR6VAktbEyITa-LXVhNM5Z720VgmZ6W?usp=sharing```
and place them under ./data.

Your data directory should look like:

* SGCI
* --data/
* ------fin_small
* ------fin
* ------regression
* ------raw_embedding


We provide the processed datasets, but if you wish to recreate them from scratch, place the pretrained embedding files in data/raw_embedding
(links to download them are provided in the same directory) and go to the corresponding folder for each task under ./data and run:

```bash
python mk_data.py 
```

## Running the model

To run the SGCI model for a task go to the corresponding folder in ./tasks and run:

```bash
python train.py --base=<EMBEDDING_NAME>
```
embedding name can be 'google', 'glove' or 'fast'

## Dependencies

* `Python` version 3.8.10
* [`Numpy`] version 1.19.2
* [`PyTorch`] version 1.4.0
* [`scikit-learn`] version 0.24.2
* [`pandas`] version 1.1.5
* [`optuna`] 2.10.1