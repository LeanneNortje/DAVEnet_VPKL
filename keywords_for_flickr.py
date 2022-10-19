#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import argparse
import torch
from models.setup import *
from models.GeneralModels import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def modelSetup(parser, test=False):

    config_file = parser.pop("config_file")
    print(f'configs/{config_library[config_file]}')
    with open(f'configs/{config_library[config_file]}') as file:
        args = json.load(file)

    image_base = parser.pop("image_base")

    for key in parser:
        args[key] = parser[key]

    args["data_train"] = Path(args["data_train"])
    args["data_val"] = Path(args["data_val"])
    args["data_test"] = Path(args["data_test"])

    getDevice(args)

    return args, image_base

command_line_args = {
    "resume": False, 
    "config_file": 'multilingual+matchmap',
    "device": "0", 
    "restore_epoch": -1, 
    "image_base": ".."
}

args, image_base = modelSetup(command_line_args)

base = Path('/mnt/HDD/leanne_HDD/Datasets/Flicker8k_Dataset')

images = list(base.rglob('*.jpg'))

other_base = Path('/mnt/HDD/leanne_HDD/Datasets/Flickr8k_text')

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize

VOCAB = []
with open('./data/34_keywords.txt', 'r') as f:
    for keyword in f:
        VOCAB.append(keyword.strip())

labels_to_images = {}
for line in open(other_base / Path('Flickr8k.token.txt'), 'r'):
    parts = line.strip().split()
    name = parts[0].split('.')[0] + '_' + parts[0].split('#')[-1]
    sentence = ' '.join(parts[1:]).lower()
    tokenized = sent_tokenize(sentence)
    for w in tokenized:

        words = nltk.word_tokenize(w)
        words = [w for w in words]
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if (tag in ['NN'] or word in VOCAB) is False: continue
            if word not in labels_to_images: labels_to_images[word] = []
            labels_to_images[word].append(name)

key = {}
id_to_word_key = {}
for i, l in enumerate(sorted(labels_to_images)):
    key[l] = i
    id_to_word_key[i] = l

ids_to_images = {}
for l in labels_to_images:
    id = key[l]
    ids_to_images[id] = labels_to_images[l]

for label in ids_to_images.copy():
    if len(ids_to_images[label]) < 20:
        ids_to_images.pop(label)

classes = set()
for w in VOCAB:
    classes.add(key[w])

choices = set(list(ids_to_images.keys())) - classes

image_labels = {}
for l in ids_to_images:
    if l in classes:
        print(l)
        for im in ids_to_images[l]:
            if im not in image_labels: image_labels[im] = []
            image_labels[im].append(l)

labels_to_images = {}
for im in image_labels:
    for id in image_labels[im]:
        if id not in labels_to_images: labels_to_images[id] = []
        labels_to_images[id].append(im)

np.savez_compressed(
    Path('data/gold_image_to_labels.npz'),
    image_labels=image_labels
)

np.savez_compressed(
    Path('data/gold_labels_to_images.npz'),
    labels_to_images=ids_to_images
)

np.savez_compressed(
    Path('data/gold_label_key.npz'),
    key=key,
    id_to_word_key=id_to_word_key
)

id_to_word = {}
for k in key:
    id_to_word[key[k]] = k

p = []
for id in ids_to_images:
    p.append(id_to_word[id])

for i in sorted(p):
    print(i)