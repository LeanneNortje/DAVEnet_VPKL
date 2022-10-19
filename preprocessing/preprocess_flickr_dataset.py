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
from image_caption_dataset_preprocessing import flickrData
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
parser.add_argument("--image-base", default="/mnt/HDD/leanne_HDD", help="Path to images.")
command_line_args = parser.parse_args()

image_base = Path(command_line_args.image_base).absolute()

with open("preprocessing_flickr_config.json") as file:
  args = json.load(file)

args["data_train"] = image_base / args["data_train"]
args["data_val"] = image_base / args["data_val"]
args["data_test"] = image_base / args["data_test"]
args["audio_base"] = image_base / args["audio_base"]
args["image_base"] = image_base / args["image_base"]
args["out_dir"] = Path(args["out_dir"])

if not os.path.isdir((Path("..") / args["out_dir"]).absolute()):
    (Path("..") / args["out_dir"]).absolute().mkdir(parents=True)


# Load in txt files
#_________________________________________________________________________________________________

def load(txt_fn, image2wav):

    data = []
    heading(f'Loading in {txt_fn.stem} ')
    with open(txt_fn, "r") as file:
        for line in tqdm(file):                                                                                                                                         
            # image_fn = args['image_base'] / Path(line.strip())
            image_fn = line.strip()
            for wav, spkr in image2wav[image_fn]:
                # audio_fn = args['audio_base'] / Path('wavs') / Path(wav)
                audio_fn = wav

                point = {
                    "speaker": spkr,
                    "wav": audio_fn,
                    "image": image_fn
                }
                data.append(point)

    # data = json_dict["data"]
    # image_base_path = Path("/".join(json_dict["image_base_path"].split("/")[-3:]))
    # audio_base_path = Path(json_dict["audio_base_path"])
    print(f'{len(data)} data points in {txt_fn}\n')
    return data

speaker_fn = Path(args['audio_base']) / Path('wav2spk.txt')
image2wav = {}
with open(speaker_fn, 'r') as file:
    for line in file:
        wav, speaker = line.strip().split()
        image_name = '_'.join(wav.split('_')[0:2]) + '.jpg'
        if image_name not in image2wav: image2wav[image_name] = []
        image2wav[image_name].append((wav, speaker))
    
train_data = load(args['data_train'], image2wav)
val_data = load(args['data_val'], image2wav)
test_data = load(args['data_test'], image2wav)

image_labels = np.load(Path('../data/gold_image_to_labels.npz'), allow_pickle=True)['image_labels'].item()

def saveADatapoint(ids, save_fn, image_fn, audio_feat, args):
    ids = list(set(ids))
    numpy.savez_compressed(
        save_fn.absolute(), 
        image=str(image_fn), 
        ids=ids,
        audio_feat=audio_feat.squeeze().numpy()
        )
    # print(list(np.load(str(save_fn) + '.npz', allow_pickle=True)['ids']))
    return "/".join(str(save_fn).split("/")[1:])

def SaveDatapointsWithMasks(dataloader, subset, datasets):

    ouputs = []
    executor = ProcessPoolExecutor(max_workers=cpu_count()) 
    no_masks = 0
    
    save_fn = Path("..") / args["out_dir"] / Path(datasets)
    masks = Path("../data/flickr_image_masks")
    if not save_fn.absolute().is_dir(): 
        save_fn.absolute().mkdir(parents=True)
        print(f'Made {save_fn}.')

    for i, (image_fn, audio_feat, audio_name, image_name, speaker) in enumerate(tqdm(dataloader, leave=False)):
        this_save_fn = save_fn / str(audio_name[0].split(".")[0] + "+SPEAKER_" + speaker[0]) 

        if (audio_feat.numpy() == 0).all():
            print(f'Audio is zero length: {audio_fn[0]}')
        else:
            # masks_fn = masks / Path(image_fn[0].split("/")[-1].split(".")[0] + ".npz")
            if audio_name[0] in image_labels:
                
                ouputs.append(executor.submit(
                    partial(saveADatapoint, image_labels[audio_name[0]], this_save_fn, image_fn[0], audio_feat, args)))
            else:
                no_masks += 1 

    data_paths = [entry.result() for entry in tqdm(ouputs)]
    json_fn = (Path("..") / args["out_dir"]).absolute() / Path(datasets + "_" + subset + ".json")      
    with open(json_fn, "w") as json_file: json.dump(data_paths, json_file, indent="")
    print(f'Wrote {len(data_paths)} data points to {json_fn}.')
    
    print(f'No ids found for {no_masks} entries')
    return json_fn      

heading(f'Preprocessing training data points.')
train_loader = torch.utils.data.DataLoader(
    flickrData(train_data, args['audio_base'] / Path('wavs'), args['image_base'], args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
train_json_fn = SaveDatapointsWithMasks(train_loader, "train", 'flickr')

heading(f'Preprocessing validation data points.')
# args["image_config"]["center_crop"] = True
val_loader = torch.utils.data.DataLoader(
    flickrData(val_data, args['audio_base'] / Path('wavs'), args['image_base'], args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
val_json_fn = SaveDatapointsWithMasks(val_loader, "val", 'flickr')

heading(f'Preprocessing testing data points.')
test_loader = torch.utils.data.DataLoader(
    flickrData(test_data, args['audio_base'] / Path('wavs'), args['image_base'], args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
test_json_fn = SaveDatapointsWithMasks(test_loader, "test", 'flickr')