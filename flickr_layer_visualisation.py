#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import *
from models.setup import *
from models.util import *
from models.GeneralModels import *
from models.multimodalModels import *
from training.util import *
from evaluation.calculations import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from training import validate
import apex
from apex import amp
import time
from tqdm import tqdm

import numpy as trainable_parameters
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
import scipy.signal
from scipy.spatial import distance
import librosa
import matplotlib.lines as lines

import itertools
import seaborn as sns

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

flickr_boundaries_fn = Path('/storage/Datasets/flickr_audio/flickr_8k.ctm')
flickr_audio_dir = flickr_boundaries_fn.parent / "wavs"
flickr_images_fn = Path('/storage/Datasets/Flicker8k_Dataset/')
flickr_segs_fn = Path('./data/flickr_image_masks/')

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }
categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148, 149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

def preemphasis(signal,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])
   
class ImageAudioData(Dataset):
    def __init__(self, samples, args):

        self.data = samples

        print(f'\n\r{len(self.data)} data points')

        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
        self.image_conf = args["image_config"]
        self.crop_size = self.image_conf.get('crop_size')
        self.center_crop = self.image_conf.get('center_crop')
        RGB_mean = self.image_conf.get('RGB_mean')
        RGB_std = self.image_conf.get('RGB_std')

        self.image_resize = transforms.transforms.Resize((256, 256))
        self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()])

        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        self.resize = transforms.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()

        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        self.image_resize = transforms.transforms.Resize((256, 256))

    
    def _myRandomCrop(self, im1, im2):

        im1 = self.resize(im1)
        im1 = self.to_tensor(im1)
        im2 = self.resize(im2)
        im2 = self.to_tensor(im2)
        return im1, im2

    def _LoadAudio(self, path):

        audio_type = self.audio_conf.get('audio_type')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        
        preemph_coef = self.audio_conf.get('preemph_coef')
        sample_rate = self.audio_conf.get('sample_rate')
        window_size = self.audio_conf.get('window_size')
        window_stride = self.audio_conf.get('window_stride')
        window_type = self.audio_conf.get('window_type')
        num_mel_bins = self.audio_conf.get('num_mel_bins')
        target_length = self.audio_conf.get('target_length')
        fmin = self.audio_conf.get('fmin')
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sample_rate)
        dur = librosa.get_duration(y=y, sr=sr)
        nsamples = y.shape[0]
        if y.size == 0:
            y = np.zeros(target_length)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)

        # compute mel spectrogram / filterbanks
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=scipy_windows.get(window_type, scipy_windows['hamming']))
        spec = np.abs(stft)**2 # Power spectrum
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        # n_frames = logspec.shape[1]
        logspec = torch.FloatTensor(logspec)
        return torch.tensor(logspec), nsamples#, n_frames

    def _LoadImage(self, impath, imseg):
        img = Image.open(impath).convert('RGB')
        img = self.image_resize(img)
        # imseg = Image.fromarray(imseg)
        
        # img, imseg = self._myRandomCrop(img, imseg)

        masked_im = np.asarray(img).copy()
        imseg = np.asarray(imseg).copy()
        # print(np.bincount(imseg))
        # ids = np.bincount(np.squeeze(np.reshape(imseg, (-1, 1)), axis=1))
        # id = np.argmax(ids)

        ids = np.unique(imseg)
        id = np.random.choice(ids, 1)[0]
        masked_im[imseg != id, :] = 0
        masked_im = Image.fromarray(masked_im)

        # img = self.image_resize_and_crop(img)
        # masked_im = self.image_resize_and_crop(masked_im)
        img = self.resize(img)
        img = self.to_tensor(img)
        masked_im = self.resize(masked_im)
        masked_im = self.to_tensor(masked_im)
        norm_masked_im = self.image_normalize(img)
        return norm_masked_im, masked_im, img

    # def _LoadImage(self, impath, imseg):
    #     img = Image.open(impath).convert('RGB')
    #     imseg = Image.fromarray(imseg)
        
    #     img, imseg = self._myRandomCrop(img, imseg)

    #     img = np.asarray(img).copy()
    #     imseg = np.asarray(imseg).copy()
    #     ids = np.unique(imseg)
    #     id = np.random.choice(ids, 1)[0]
    #     masked_im = img.copy()
    #     masked_im[imseg != id, :] = 0
    #     img = Image.fromarray(img)
    #     masked_im = Image.fromarray(masked_im)

    #     img = self.tensor(img)
    #     masked_im = self.tensor(masked_im)
    #     norm_masked_im = self.image_normalize(masked_im)
    #     return norm_masked_im, masked_im, img

    def _PadFeat(self, feat):
        print(feat.shape)
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def __getitem__(self, index):

        data_point = self.data[index]

        english_audio_feat, _ = self._LoadAudio(data_point['wav'])
        english_audio_feat, english_nframes = self._PadFeat(english_audio_feat)
        english_name = data_point['wav'].stem

        # hindi_audio_feat = data_point["hindi_audio_feat"]
        # hindi_audio_feat, hindi_nframes = self._PadFeat(hindi_audio_feat)
        
        imgpath = data_point['img']
        seg = np.load(data_point['seg'])['panoptic_segmentation']
        image, masked_img, raw_img = self._LoadImage(imgpath, seg)

        return image, masked_img, raw_img, english_audio_feat, english_nframes, str(english_name)

    def __len__(self):
        return len(self.data)

def spawn_training(rank, world_size, image_base, args, restore_epoch):

    # # Create dataloaders
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD,
    )

    if rank == 0: 
        boundaries = {}
        with open(flickr_boundaries_fn, 'r') as file:
            for line in tqdm(file):
                parts = line.strip().split()
                name = parts[0].split(".")[0]
                
                seg = Path(flickr_segs_fn) / Path(name + ".npz")
                if seg.is_file():
                    
                    number = parts[0].split("#")[-1]
                    wav = flickr_audio_dir / Path(name + "_" + number + ".wav")
                    img = Path(flickr_images_fn) / Path(name + ".jpg")
                    if name + "_" + number not in boundaries:
                        boundaries[name + "_" + number] = {"wav": wav, "img": img, "seg": seg}
                    boundaries[name + "_" + number].setdefault('boundaries', []).append((parts[2], parts[3], parts[4]))
        print('Num boundaries: ', len(boundaries))

        samples = []
        print_example = True
        for key in boundaries:
            point = boundaries[key]
            if print_example: 
                print(point)
                print_example = False
            samples.append({"wav": point["wav"], "img": point["img"], "seg": point["seg"], "boundaries": point["boundaries"]})
        print('Num samples: ', len(samples))


        heading(f'\nLoading testing data ')
        args["image_config"]["center_crop"] = True
        validation_dataset = ImageAudioData(samples[0:10], args)
        validation_sampler = DistributedSampler(validation_dataset, drop_last=False)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,#args["batch_size"],
            # num_workers=32,
            pin_memory=True,
        )

        cat_ids_to_labels = np.load(Path("data/mask_cat_id_labels.npz"), allow_pickle=True)['cat_ids_to_labels'].item()
        
        trans = transforms.ToPILImage()

        # Create models
        audio_model = mutlimodal(args).to(rank)

        image_model_name = imageModel(args)
        image_model = image_model_name(args, pretrained=args["pretrained_image_model"]).to(rank)

        attention = ScoringAttentionModule(args).to(rank)
        contrastive_loss = ContrastiveLoss(args).to(rank)

        model_with_params_to_update = {
            "audio_model": audio_model,
            "attention": attention,
            "contrastive_loss": contrastive_loss
            }
        model_to_freeze = {
            "image_model": image_model
            }
        trainable_parameters = getParameters(model_with_params_to_update, model_to_freeze, args)

        if args["optimizer"] == 'sgd':
            optimizer = torch.optim.SGD(
                trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
                momentum=args["momentum"], weight_decay=args["weight_decay"]
                )
        elif args["optimizer"] == 'adam':
            optimizer = torch.optim.Adam(
                trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
                weight_decay=args["weight_decay"]
                )
        else:
            raise ValueError('Optimizer %s is not supported' % args["optimizer"])

        scaler = torch.cuda.amp.GradScaler()

        audio_model = DDP(audio_model, device_ids=[rank])
        image_model = DDP(image_model, device_ids=[rank])
        # attention = DDP(attention, device_ids=[rank])
        # contrastive_loss = DDP(contrastive_loss, device_ids=[rank])

        heading(f'\nRetoring model parameters from best epoch ')
        info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
            args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, True
            )
        # info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
        #     args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, 1
        #     )

        trans = transforms.ToPILImage()
        with torch.no_grad():
            for i, (image_input, masked_image, raw_image, english_input, english_nframes, english_name) in enumerate(validation_loader):
                
                with torch.cuda.amp.autocast():
                    
                    image_output = image_model(image_input.to(rank))[0, :, :, :].unsqueeze(0)
                    
                    fig = plt.figure(figsize=(68, 9), constrained_layout=True)
                    gs = GridSpec(17, 2, figure=fig)

                    these_boundaries = boundaries[english_name[0]]['boundaries']
                    these_orginial_boundaries = []

                    for word in these_boundaries:
                        start = float(word[0])
                        dur = float(word[1])
                        label = word[2]
                        
                        begin = int(start * 100)
                        end = int((start + dur) * 100)# + 1

                        these_orginial_boundaries.append((begin, end, label))
                    
                    ind = 0
                    fig_count = 0

                    this_layer = audio_model.module.encode(english_input[ind, :, :].unsqueeze(0).to(rank), 'mels')
                    pooling_ratio = english_input[ind, :, :].size(-1) / this_layer.size(-1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer, english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, :, 0:temp]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count, ind])
                    sns.heatmap(this_layer.squeeze(0).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("mels")
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 40], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 40], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 20, 20, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)


                    this_layer = audio_model.module.encode(english_input[ind, :, :].unsqueeze(0).to(rank), 'pre_z')
                    pooling_ratio = english_input[ind, :, :].size(-1) / this_layer.size(1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer.transpose(1, 2), english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, 0:temp, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count, ind])
                    sns.heatmap(this_layer.mean(dim=-1).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("pre_z")
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 1], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 1], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)

                    this_layer = audio_model.module.encode(english_input[ind, :, :].unsqueeze(0).to(rank), 'z')
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer.transpose(1, 2), english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, 0:temp, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count, ind])
                    sns.heatmap(this_layer.mean(dim=-1).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("z")
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 1], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 1], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)


                    this_layer = audio_model.module.encode(english_input[ind, :, :].unsqueeze(0).to(rank), 'c1')
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer.transpose(1, 2), english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, 0:temp, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count, ind])
                    sns.heatmap(this_layer.mean(dim=-1).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("c1")
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 1], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 1], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)


                    this_layer = audio_model.module.encode(english_input[ind, :, :].unsqueeze(0).to(rank), 'c2')
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer.transpose(1, 2), english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, 0:temp, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count, ind])
                    sns.heatmap(this_layer.mean(dim=-1).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("c2")
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 1], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 1], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)


                    this_layer = audio_model.module.encode(english_input[ind, :, :].unsqueeze(0).to(rank), 'c3')
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer.transpose(1, 2), english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, 0:temp, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count, ind])
                    sns.heatmap(this_layer.mean(dim=-1).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("c3")
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 1], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 1], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)


                    this_layer = audio_model.module.encode(english_input[ind, :, :].unsqueeze(0).to(rank), 'c4')
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(-1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer, english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, 0:temp, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count, ind])
                    sns.heatmap(this_layer.mean(dim=-1).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("c4")
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 1], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 1], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)


                    this_layer = audio_model.module.encode(english_input[ind, :, :].unsqueeze(0).to(rank), 's1')
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer.transpose(1, 2), english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, 0:temp, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count, ind])
                    sns.heatmap(this_layer.mean(dim=-1).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("s1")
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 1], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 1], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)

                    _, _, english_output = audio_model(english_input.to(rank))#.to('cpu').detach() 

                    this_layer = english_output[ind, :, :].unsqueeze(0)
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(-1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer, english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, :, 0:temp]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count:fig_count+2, ind])
                    sns.heatmap(this_layer.mean(dim=1).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title("s2")
                    ax.axis('off')
                    fig_count += 2
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, 1], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, 1], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)


                    im = image_output.view(image_output.size(0), image_output.size(1), -1).transpose(1, 2)
                    att, _, _, _, score, score_embed, M, _, _ = attention.encode(im, english_output[ind, :, :].unsqueeze(0), english_nframes[ind].unsqueeze(0))
                    this_layer = att
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(-1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer, english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, :, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count:fig_count+1, ind])
                    sns.heatmap(this_layer.squeeze(0).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title(f'Attention')
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, this_layer.size(0)], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, this_layer.size(0)], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5*this_layer.size(0), label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)


                    this_layer = M
                    pooling_ratio = english_input[ind, :, :].unsqueeze(0).size(-1) / this_layer.size(-1)
                    temp = NFrames(english_input[ind, :, :].unsqueeze(0), this_layer, english_nframes[ind].unsqueeze(0))
                    this_layer = this_layer[:, :, :]
                    print(this_layer.size())
                    ax = fig.add_subplot(gs[fig_count:, ind])
                    sns.heatmap(this_layer.squeeze(0).cpu().numpy(), cmap='viridis', ax=ax)
                    ax.set_title(f'M (score: {score[ind].item()})')
                    ax.axis('off')
                    fig_count += 1
                    c = 'w'
                    for (begin, end, label) in these_orginial_boundaries:
                        begin = begin // pooling_ratio
                        end = end // pooling_ratio
                        ax.add_artist(lines.Line2D([begin, begin], [0, this_layer.size(1)], color=c)) 
                        if label == these_boundaries[-1][2]: 
                            ax.add_artist(lines.Line2D([end, end], [0, this_layer.size(1)], color=c, linewidth=3.5)) 
                        ax.text(begin+(end-begin)//2 + 0.5, 0.5*this_layer.size(1), label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold')#, rotation=90)

            
                    plt.show()

                # if i == 5: break
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action="store_true", dest="resume",
            help="load from exp_dir if True")
    parser.add_argument("--config-file", type=str, default='multilingual+matchmap', choices=['multilingual', 'multilingual+matchmap'], help="Model config file.")
    parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
    parser.add_argument("--image-base", default="/mnt/HDD/leanne_HDD", help="Model config file.")
    command_line_args = parser.parse_args()
    restore_epoch = command_line_args.restore_epoch

    # Setting up model specifics
    heading(f'\nSetting up model files ')
    args, image_base = modelSetup(command_line_args, True)

    world_size = 1
    mp.spawn(
        spawn_training,
        args=(world_size, image_base, args, restore_epoch),
        nprocs=world_size,
        join=True,
    )