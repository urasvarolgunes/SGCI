We provide the processed datasets, but if you wish to recreate them from scratch, place the pretrained embedding files in data/raw_embedding
(links to download them are provided in the same directory) and go to the corresponding folder for each task under ./data and run:

```bash
python mk_data.py 
```

To run the SGCI model for a task go to the corresponding folder in ./tasks and run:

```bash
python train.py --model=SGC
```

You can use `environment.yml` to create a virtual environment with conda using the following command:

```bash
conda env create -f environment.yml


