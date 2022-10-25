#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import sys
import os
from pathlib import Path
import hashlib
import pickle
import json
import numpy as np
import torch
import shutil

config_library = {
    "matchmap": "params.json"
}

def printDirectory(path):
    parts = str(path).strip("./").split("/")
    for i, part in enumerate(parts):
        arrow = "\u21AA" 
        indicator = f'{" ":<{len(arrow) * i}} {arrow} ' if i != 0 else ""
        print(' '*5 + f'{indicator}{part}')
    print("\n")

def printArguments(dictionary, offset=0):

    for key in sorted(dictionary):
        tabs = "\t"*(offset + 1)
        if isinstance(dictionary[key], dict):
            print(f'{tabs}{key}:')
            printArguments(dictionary[key], offset + 1)
        elif isinstance(dictionary[key], list):
            lengths = [len(entry) for entry in dictionary[key] if isinstance(entry, (dict, list))]
            if len(np.where(np.asarray(lengths) > 3)[0]) != 0:
                print(f'{tabs}{key}:')
                for entry in dictionary[key]:
                    print(f'{tabs}\t{entry}')
            else:
              print(f'{tabs}{key}: {dictionary[key]}')  
        else:
            print(f'{tabs}{key}: {dictionary[key]}')

    print("\n")

def getDevice(args):

    args["device"] = int(args["device"]) if args["device"] != "cuda" else args["device"]
    if isinstance(args["device"], int): 
        if args["device"] >= torch.cuda.device_count(): args["device"] = 0


def modelSetup(parser, test=False):

    parser = vars(parser)
    config_file = parser.pop("config_file")
    print(f'configs/{config_library[config_file]}')
    with open(f'configs/{config_library[config_file]}') as file:
        args = json.load(file)

    if "restore_epoch" in parser:
        restore_epoch = parser.pop("restore_epoch")
    if "resume" in parser:
        resume = parser.pop("resume")
    else: 
        resume = False
    if "feat" in parser:
        feat = parser.pop("feat")
    else:
        feat = None
    if "dataset_path" in parser:
        dataset_path = parser.pop("dataset_path")
    else: 
        dataset_path = None
    if "base_path" in parser:
        base_path = parser.pop("base_path")
    else: 
        base_path = None
    if "path" in parser:
        test_path = parser.pop("path")
    else: 
        test_path = None
    image_base = parser.pop("image_base")
    # device = parser.pop("device")

    for key in parser:
        args[key] = parser[key]

    args["data_train"] = Path(args["data_train"])
    args["data_val"] = Path(args["data_val"])
    args["data_test"] = Path(args["data_test"])

    modelHash(args)

    base_dir = Path("model_metadata")    
    data = "_".join(str(Path(os.path.basename(args["data_train"])).stem).split("_")[0:4])
    model_particulars = f'AudioModel-{args["audio_model"]["name"]}_ImageModel-{args["image_model"]}_ArgumentsHash-{args["model_name"]}_ConfigFile-{Path(config_library[config_file]).stem}' 
    args["exp_dir"] = base_dir / data / model_particulars

    if test or resume:

        print(f'\nRecovering model arguments from')
        printDirectory(args["exp_dir"] / "args.pkl")

        print((args["exp_dir"] / "args.pkl").absolute())
        assert(os.path.isfile((args["exp_dir"] / "args.pkl").absolute()))
        with open(args["exp_dir"] / "args.pkl", "rb") as f:
            args = pickle.load(f)
        
        for key in parser:
            args[key] = parser[key]

        if restore_epoch != -1: args["restore_epoch"] = restore_epoch
        args["resume"] = resume
        if dataset_path is not None: args["dataset_path"] = dataset_path
        if base_path is not None: args["base_path"] = base_path
        if test_path is not None: args["path"] = test_path

    else:
        with open(f'models/AcousticEncoder.json') as file: model_params = json.load(file)
        args["acoustic_model"] = model_params
        assert(os.path.isfile(args["exp_dir"]) is False)
        print(f'\nMaking model directory:')
        printDirectory(args["exp_dir"])
        print(f'Saving model arguments at:')
        printDirectory(args["exp_dir"] / "args.pkl")

        os.makedirs(args["exp_dir"])
        with open(args["exp_dir"] / "args.pkl", "wb") as f:
            pickle.dump(args, f)
        args["resume"] = False
    # args["device"] = device
    if feat is not None:
        args['feat'] = feat

    if os.path.isfile(args["exp_dir"] / config_library[config_file]) is False:
        print(f'Copying original config file:')
        shutil.copyfile(f'configs/{config_library[config_file]}', args["exp_dir"] / config_library[config_file])
        cpc_pretrained_name = args['cpc']['pretrained_weights']
        if args['cpc']['warm_start']: shutil.copyfile(Path(f'pretrained_cpc/{cpc_pretrained_name}.pt'), args["exp_dir"] / 'pretrained_cpc.pt')
        
        if 'load_pretrained_weights' in args['audio_model']: 
            semantics_pretrained_name = args['audio_model']['pretrained_weights']
            if args['audio_model']['load_pretrained_weights']: shutil.copyfile(Path(f'pretrained_semantics/{semantics_pretrained_name}.pt'), args["exp_dir"] / 'pretrained_semantics.pt')
    print(f'Model arguments:')
    printArguments(args)

    # getDevice(args)

    return args, image_base


def modelHash(args):

    exclude_keys = ["resume"]

    name_dict = args.copy()
    name_dict.pop("resume", None)

    args["model_name"] = hashlib.md5(repr(sorted(name_dict.items())).encode("ascii")).hexdigest()[:10]