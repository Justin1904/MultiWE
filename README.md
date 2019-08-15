# Multimodal Word Embedding

This is the repo for learning multimodal word embeddings from the CMU-MOSEI dataset -- a dataset of YouTube videos.

## Dependencies

To run the code we need the following dependencies:

 * pytorch 1.1
 * tqdm
 * numpy


## Data

The processed data can be downloaded [here](https://non-existing-url). If you want to check out how the data is created
 as well as the code for doing so, please see to the dev branch of this repo, there will be additional dependencies as well.

## Train Embeddings

### Prepare the data

After downloading the data, navigate to the root of this repo and create a directory `data`. Then extract the downloaded data to the `data` folder. After this the `data` directory should look like:

```
 - data
    - glove_cache.pt
    - data_cache.pt
```

### Running the experiments

Once the data is in place, you can run the following command for experiments:

```
python main.py
```

There's no additional arguments to be provided - 
