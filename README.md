# Using DAVENet to do VPKL

This repository includes the code used to implement the DAVEnet experiments in the paper: [TOWARDS VISUALLY PROMPTED KEYWORD LOCALISATION FOR ZERO-RESOURCE SPOKEN LANGUAGES](https://arxiv.org/pdf/2210.06229.pdf). 

## Disclaimer

I provide no guarantees with this code, but I do believe the experiments in the above mentioned paper, can be reproduced with this code. Please notify me if you find any bugs or problems. 

## Clone the repository 

To clone the repository run:

```
git clone https://github.com/LeanneNortje/DAVEnet_VPKL.git
```

To get into the repository run:

```
cd DAVEnet_VPKL/
```

## VPKL visual keys

To obtain the visual keys used for testing, downlad the ```data``` and ```visual_keys``` folder from [here](https://github.com/LeanneNortje/VPKL).

## Datasets


**Flickr**

Download the [images](https://www.kaggle.com/datasets/adityajn105/flickr8k) and corresponding [audio](https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads.cgi) separatly and extract both in the same folder. 

## Data processing

If you want to redo the keyword to id mapping, the mapping of ids to its images and the images to its id(s), then run:
```
python3 keywords_for_flickr.py
```
after the paths in the script is changed to yours.

To process the data, follow the next steps: 
```
cd preprocessing/
python3 preprocess_flickr_dataset.py --image-base path/to/flickr-dataset

```
## Training models

To train a model, change the values in ```configs/params.json``` to the desired ones. Then, run:

```
python3 run.py --image-base path/to/flickr-dataset
```

## Validation

To get the model's threshold for keyword detection, run

```
python3 val_vpkl_mean_threshold.py --image-base path/to/flickr-dataset
```
to get the mean threshold where each keyword instance in the validation set occurs. Then, fine-tune the threshold using 

```
python3 val_vpkl.py --image-base path/to/flickr-dataset
```
and changing the threshold in the script as wanted.

## Testing VPKL

To do the final VPKL test on the held-out test data, run the following:

```
python3 test_vpkl.py --image-base path/to/flickr-dataset
```

NOTE: checkpoints for models used in the paper, is given in this repository.