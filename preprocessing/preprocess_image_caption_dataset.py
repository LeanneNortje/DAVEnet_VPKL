#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import argparse
import os
import pickle
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from image_caption_dataset_preprocessing import ImageCaptionDataset
import json
from pathlib import Path
import numpy
from collections import Counter
import sys
from os import popen
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)
def heading(string):
    print("_"*terminal_width + "\n")
    print("-"*10 + string + "-"*10 + "\n")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--image-base", default="/storage", help="Path to images.")
command_line_args = parser.parse_args()

image_base = Path(command_line_args.image_base).absolute()

with open("preprocessing_config.json") as file:
  args = json.load(file)

args["hindi_data_train"] = image_base / args["hindi_data_train"]
args["hindi_data_val"] = image_base / args["hindi_data_val"]
args["hindi_data_test"] = image_base / args["hindi_data_test"]
args["english_data_train"] = image_base / args["english_data_train"]
args["english_data_val"] = image_base / args["english_data_val"]
args["out_dir"] = Path(args["out_dir"])

if not os.path.isdir((Path("..") / args["out_dir"]).absolute()):
    (Path("..") / args["out_dir"]).absolute().mkdir(parents=True)


# Load in json files
#_________________________________________________________________________________________________

def load(json_fn):
    heading(f'Loading in {json_fn.stem} ')
    with open(json_fn, "r") as file:
        json_dict = json.load(file)
    data = json_dict["data"]
    image_base_path = Path("/".join(json_dict["image_base_path"].split("/")[-3:]))
    audio_base_path = Path(json_dict["audio_base_path"])
    print(f'{len(data)} data points in {json_fn}\n')
    return data, image_base_path, audio_base_path

hindi_train_data, hindi_train_image_fn, hindi_train_audio_fn = load(args["hindi_data_train"])
hindi_val_data, hindi_val_image_fn, hindi_val_audio_fn = load(args["hindi_data_val"])
hindi_test_data, hindi_test_image_fn, hindi_test_audio_fn = load(args["hindi_data_test"])
assert(hindi_train_image_fn == hindi_val_image_fn and hindi_train_image_fn == hindi_test_image_fn)
assert(hindi_train_audio_fn == hindi_val_audio_fn and hindi_train_audio_fn == hindi_test_audio_fn)
hindi_image_fn = hindi_train_image_fn
hindi_audio_fn = hindi_train_audio_fn

english_train_data, english_train_image_fn, english_train_audio_fn = load(args["english_data_train"])
english_val_data, english_val_image_fn, english_val_audio_fn = load(args["english_data_val"])
assert(english_train_image_fn == english_val_image_fn)
assert(english_train_audio_fn == english_val_audio_fn)
english_image_fn = english_train_image_fn
english_audio_fn = english_train_audio_fn

# Check overlap
#_________________________________________________________________________________________________

def normalizeWavPaths(list_of_jsons):
    for this_json in list_of_jsons:
        for item in this_json:
            new_wav = "/".join(item["wav"].split("/")[2:])
            item["wav"] = new_wav

def testJSONOverlap(first, second):
    overlap = 0
    for item in tqdm(first, leave=False):
        for other_item in second:
            if item == other_item: overlap += 1
    return overlap

def removeOverlap(keep, remove):

    for item in tqdm(keep, leave=False):
        for i, other_item in enumerate(remove):
            if item == other_item: 
                del remove[i]

    return testJSONOverlap(keep, remove)


normalizeWavPaths([hindi_train_data, hindi_val_data, hindi_test_data])


overlap = testJSONOverlap(hindi_train_data, hindi_val_data)
print(f'{overlap} data points overlap between Hindi train and dev sets.')
if overlap != 0: 
    new_overlap = removeOverlap(hindi_val_data, hindi_train_data)
    print(f'Now {new_overlap} data points overlap between Hindi train and dev sets.\n')

overlap = testJSONOverlap(hindi_train_data, hindi_test_data)
print(f'{overlap} data points overlap between Hindi train and test sets.')
if overlap != 0: 
    new_overlap = removeOverlap(hindi_test_data, hindi_train_data)
    print(f'Now {new_overlap} data points overlap between Hindi train and test sets.\n')

overlap = testJSONOverlap(hindi_val_data, hindi_test_data)
print(f'{overlap} data points overlap between Hindi dev and test sets.')
if overlap != 0: 
    new_overlap = removeOverlap(hindi_test_data, hindi_val_data)
    print(f'Now {new_overlap} data points overlap between Hindi dev and test sets.\n')


# Check duplicates
#_________________________________________________________________________________________________

def JSONduplicates(json_file):

    original_length = len(json_file)
    converted_json = list(map(dict, set(tuple(sorted(dictionary.items())) for dictionary in json_file)))
    print(f'{original_length-len(converted_json)} duplicates removed.')

heading(f'Checking for duplicate data points in {args["hindi_data_train"].stem}.')
JSONduplicates(hindi_train_data)
heading(f'Checking for duplicate data points in {args["hindi_data_val"].stem}.')
JSONduplicates(hindi_val_data)
heading(f'Checking for duplicate data points in {args["hindi_data_test"].stem}.')
JSONduplicates(hindi_test_data)
heading(f'Checking for duplicate data points in {args["english_data_train"].stem}.')
JSONduplicates(english_train_data)
heading(f'Checking for duplicate data points in {args["english_data_val"].stem}.')
JSONduplicates(english_val_data)

def unNormalizeWavPaths(dict_of_jsons):
    for subset_fn in dict_of_jsons:
        for item in dict_of_jsons[subset_fn]:
            new_wav = subset_fn + item["wav"]
            item["wav"] = new_wav

unNormalizeWavPaths(
    {
    "hindi_wavs/train/": hindi_train_data, 
    "hindi_wavs/dev/": hindi_val_data, 
    "hindi_wavs/test/": hindi_test_data
    }
)


# Make one big English set
#_________________________________________________________________________________________________

heading(f'Merging all english points.')
english_points = english_train_data.copy()
english_points.extend(english_val_data)
print(f'Combining {len(english_train_data)} english training data points with {len(english_val_data)} to get set {len(english_points)} points.')

english = {}
for point in english_points:
    image = point["image"]
    if image not in english: 
        english[image] = []
    english[image].append(point)

# Merge Hindi and English
#_________________________________________________________________________________________________

def popDict(dict_list, dict_to_pop):
    ind = [i for i, p in enumerate(dict_list) if p == dict_to_pop][0]
    del dict_list[ind]

def mergePoints(hindi, eng):
    subset = []
    found = 0
    not_found = 0
    for point in tqdm(hindi, leave=False):
        im = point["image"]
        if im in eng:
            match = np.random.choice(eng[im])
            merge = {
                "uttid": match["uttid"],
                "english_speaker": match["speaker"],
                "english_asr_text": match["asr_text"],
                "english_wav": match["wav"],
                "image": match["image"],
                "hindi_speaker": point["speaker"],
                "hindi_asr_text": point["asr_text"],
                "hindi_wav": point["wav"]
            }
            subset.append(merge)
            popDict(eng[im], match)
            if len(eng[im]) == 0: _ = eng.pop(im, None)
            found += 1
        else: not_found += 1
    print(f'\nFound English matches for {found} Hindi points.')
    print(f'Could not find English matches for {not_found} Hindi points.')
    return subset

test_data = mergePoints(hindi_test_data, english)
print(f'English-Hindi test set contains {len(test_data)} data points')
val_data = mergePoints(hindi_val_data, english)
print(f'English-Hindi dev set contains {len(val_data)} data points')
train_data = mergePoints(hindi_train_data, english)
print(f'English-Hindi train set contains {len(train_data)} data points')

def SaveDatapoins(dataloader, subset, datasets):

    data_paths = []
    if subset == "train":
        executor = ProcessPoolExecutor(max_workers=cpu_count()) 
        no_masks = 0
    for i, (image_fn, eng_audio_feat, hindi_audio_feat, eng_audio_fn, hindi_audio_fn, image_path, eng_speaker) in enumerate(tqdm(dataloader, leave=False)):
        save_fn = args["out_dir"]/Path(datasets)
        save_fn = save_fn / str("ENGLISH_" + eng_speaker[0] + "-" + eng_audio_fn[0].split("/")[-1].split(".")[0] + "+HINDI_" + hindi_audio_fn[0].split("/")[-1].split(".")[0] + "+" + image_fn[0].split("/")[-1].split(".")[0])

        if (eng_audio_feat.numpy() == 0).all() or (hindi_audio_feat.numpy() == 0).all():
            if (eng_audio_feat.numpy() == 0).all() and (hindi_audio_feat.numpy() != 0).all(): print(f'English audio is zero not Hindi audio: {eng_audio_fn[0]}')
            if (eng_audio_feat.numpy() != 0).all() or (hindi_audio_feat.numpy() == 0).all(): print(f'English audio is not zero but Hindi audio is: {hindi_audio_fn[0]}')
        else:
            if not (Path("..") / save_fn).absolute().parent.is_dir(): (Path("..") / save_fn).absolute().parent.mkdir(parents=True)
            numpy.savez_compressed(
                (Path("..") / save_fn).absolute(), 
                image=str(image_fn[0]), 
                eng_audio_feat=eng_audio_feat.squeeze().numpy(), 
                hindi_audio_feat=hindi_audio_feat.squeeze().numpy()
                )
            data_paths.append(str(save_fn))  

    json_fn = (Path("..") / args["out_dir"]).absolute() / Path(datasets + "_" + subset + ".json")      
    with open(json_fn, "w") as json_file: json.dump(data_paths, json_file, indent="")
    print(f'Wrote {len(data_paths)} data points to {json_fn}.')
    if subset == "train": 
        print(f'No masks found for {no_masks} entries') 
    return json_fn


def saveADatapoint(masks_fn, save_fn, image_fn, eng_audio_feat, hindi_audio_feat, args):
    masks_file = np.load(masks_fn)
    panoptic_segmentation = masks_file['panoptic_segmentation']
    numpy.savez_compressed(
        save_fn.absolute(), 
        image=str(image_fn), 
        panoptic_segmentation=panoptic_segmentation,
        eng_audio_feat=eng_audio_feat.squeeze().numpy(), 
        hindi_audio_feat=hindi_audio_feat.squeeze().numpy()
        )
    return "/".join(str(save_fn).split("/")[1:])

def SaveDatapointsWithMasks(dataloader, subset, datasets):

    ouputs = []
    executor = ProcessPoolExecutor(max_workers=cpu_count()) 
    no_masks = 0
    
    save_fn = Path("..") / args["out_dir"] / Path(datasets)
    masks = Path("../data/PlacesAudio_400k_distro+PlacesHindi100k+imagesPlaces205_resize+image_masks")
    if not save_fn.absolute().is_dir(): 
        save_fn.absolute().mkdir(parents=True)
        print(f'Made {save_fn}.')

    for i, (image_fn, eng_audio_feat, hindi_audio_feat, eng_audio_fn, hindi_audio_fn, image_path, eng_speaker) in enumerate(tqdm(dataloader, leave=False)):
        
        this_save_fn = save_fn / str("ENGLISH_" + eng_speaker[0] + "-" + eng_audio_fn[0].split("/")[-1].split(".")[0] + "+HINDI_" + hindi_audio_fn[0].split("/")[-1].split(".")[0] + "+" + image_fn[0].split("/")[-1].split(".")[0])

        if (eng_audio_feat.numpy() == 0).all() or (hindi_audio_feat.numpy() == 0).all():
            if (eng_audio_feat.numpy() == 0).all() and (hindi_audio_feat.numpy() != 0).all(): print(f'English audio is zero not Hindi audio: {eng_audio_fn[0]}')
            if (eng_audio_feat.numpy() != 0).all() or (hindi_audio_feat.numpy() == 0).all(): print(f'English audio is not zero but Hindi audio is: {hindi_audio_fn[0]}')
       
        else:
            masks_fn = masks / Path(image_fn[0].split("/")[-1].split(".")[0] + ".npz")

            if masks_fn.is_file():
                
                ouputs.append(executor.submit(
                                    partial(saveADatapoint, masks_fn, this_save_fn, image_fn[0], eng_audio_feat, hindi_audio_feat, args)))
            else:
                no_masks += 1 

    data_paths = [entry.result() for entry in tqdm(ouputs)]
    json_fn = (Path("..") / args["out_dir"]).absolute() / Path(datasets + "_" + subset + ".json")      
    with open(json_fn, "w") as json_file: json.dump(data_paths, json_file, indent="")
    print(f'Wrote {len(data_paths)} data points to {json_fn}.')
    
    print(f'No masks found for {no_masks} entries')
    return json_fn

heading(f'Preprocessing training data points.')
train_loader = torch.utils.data.DataLoader(
    ImageCaptionDataset(
        train_data, 
        hindi_audio_fn, english_audio_fn, english_image_fn, 
        args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
train_json_fn = SaveDatapointsWithMasks(train_loader, "train", str(english_audio_fn.name)+"+"+str(hindi_audio_fn.name)+"+"+str(english_image_fn.name))
# train_json_fn = Path("../data/PlacesAudio_400k_distro+PlacesHindi100k+imagesPlaces205_resize_train.json")

heading(f'Preprocessing validation data points.')
# args["image_config"]["center_crop"] = True
val_loader = torch.utils.data.DataLoader(
    ImageCaptionDataset(
        val_data, 
        hindi_audio_fn, english_audio_fn, english_image_fn, 
        args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
val_json_fn = SaveDatapointsWithMasks(val_loader, "val", str(english_audio_fn.name)+"+"+str(hindi_audio_fn.name)+"+"+str(english_image_fn.name))

heading(f'Preprocessing testing data points.')
test_loader = torch.utils.data.DataLoader(
    ImageCaptionDataset(
        test_data, 
        hindi_audio_fn, english_audio_fn, english_image_fn, 
        args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
test_json_fn = SaveDatapointsWithMasks(test_loader, "test", str(english_audio_fn.name)+"+"+str(hindi_audio_fn.name)+"+"+str(english_image_fn.name))

with open(train_json_fn, "r") as file:
    train_dps = json.load(file)

with open(val_json_fn, "r") as file:
    val_dps = json.load(file)

with open(test_json_fn, "r") as file:
    test_dps = json.load(file)


# Some tests
print(f'\nTesting overlap:\n')

difference = testJSONOverlap(train_dps, val_dps)
correctness = "as expected" if difference == 0 else "not as exected"
print(f'{difference} data points overlap between training and validation datapoints, {correctness}.\n')

difference = testJSONOverlap(val_dps, test_dps)
correctness = "as expected" if difference == 0 else "not as exected"
print(f'{difference} data points overlap between validation and testing datapoints, {correctness}.\n')

difference = testJSONOverlap(train_dps, test_dps)
correctness = "as expected" if difference == 0 else "not as exected"
print(f'{difference} data points overlap between training and testing datapoints, {correctness}.\n')