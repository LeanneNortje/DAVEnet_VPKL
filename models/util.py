#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import json
import torch
import os
from os import popen
from math import ceil
from models.ImageModels import *
from models.AudioModels import *
from pathlib import Path

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)
image_model_dict = {
	"VGG16": VGG16,
	"Resnet50": Resnet50,
	"Resnet101": Resnet101
}
audio_model_dict = {
	"Davenet": AudioCNN,
	"ResDavenet": ResDavenet,
	"Transformer": BidrectionalAudioLSTM
}

def heading(string):
    print("_"*terminal_width + "\n")
    print(string)
    print("_"*terminal_width + "\n")

def imageModel(args):
	if args["image_model"] == "VGG16":
		return image_model_dict["VGG16"]
	elif args["image_model"] == "Resnet50":
		return image_model_dict["Resnet50"]
	elif args["image_model"] == "Resnet101":
		return image_model_dict["Resnet101"]
	else:
		raise ValueError(f'Unknown image model: {args["image_model"]["name"]}')

def audioModel(args):
	if args["audio_model"]["name"] == "DAVEnet":
		with open(f'models/DAVEnet.json') as file: model_params = json.load(file)
		args["audio_model"]["conv_layers"] = model_params["conv_layers"]
		args["audio_model"]["max_pool"] = model_params["max_pool"]
		return audio_model_dict["Davenet"]
	elif args["audio_model"]["name"] == "ResDAVEnet":
		with open(f'models/ResDAVEnet.json') as file: model_params = json.load(file)
		args["audio_model"]["conv_layers"] = model_params["conv_layers"]
		return audio_model_dict["ResDavenet"]
	elif args["audio_model"]["name"] == "Transformer":
		return audio_model_dict["Transformer"]
	else:
		raise ValueError(f'Unknown audio model: {args["audio_model"]["name"]}')

def acousticModel(args):
	with open(f'models/AcousticEncoder.json') as file: model_params = json.load(file)
	args["acoustic_model"] = model_params
	

def loadPretrainedWeights(audio_model, args, rank):

	# device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
	audio_model = nn.DataParallel(audio_model)
	model_dict = audio_model.state_dict()

	cpc_pretrained_name = args['cpc']['pretrained_weights']
	checkpoint_fn = Path(f'pretrained_cpc/{cpc_pretrained_name}.pt')
	checkpoint = torch.load(checkpoint_fn, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})

	for key in checkpoint["acoustic_model"]:
		if key in model_dict: model_dict[key] = checkpoint["acoustic_model"][key]
	audio_model.load_state_dict(model_dict)

	return audio_model.module

def loadPretrainedSemanticWeights(english_model, hindi_model, args):

	device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
	semantics_pretrained_name = args['audio_model']['pretrained_weights']
	checkpoint_fn = Path(f'pretrained_semantics/{semantics_pretrained_name}.pt')
	checkpoint = torch.load(checkpoint_fn, map_location=device)
	english_model.load_state_dict(checkpoint["english_audio_model"])
	hindi_model.load_state_dict(checkpoint["hindi_audio_model"])

	return english_model, hindi_model