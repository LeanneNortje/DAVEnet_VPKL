#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import json
import librosa
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy
from pathlib import Path

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

def preemphasis(x,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    return scipy.signal.lfilter([1, -coeff], [1], x)
    # return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class ImageCaptionDataset(Dataset):
    # function adapted from https://github.com/dharwath

    def __init__(self, data, hindi_audio_fn, english_audio_fn, image_fn, audio_conf):

        self.data = data
        self.hindi_audio_fn = Path(hindi_audio_fn)
        self.english_audio_fn = Path(english_audio_fn)
        self.image_fn = Path(image_fn)
        self.audio_conf = audio_conf
        
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
        return logspec#, n_frames

    def __getitem__(self, index):
        datum = self.data[index]
        eng_wav_fn = self.english_audio_fn / Path(datum['english_wav'])
        hindi_wav_fn = self.hindi_audio_fn / Path(datum['hindi_wav'])
        imgpath = self.image_fn / Path(datum['image'])
        eng_audio_feat = self._LoadAudio(eng_wav_fn)
        hindi_audio_feat = self._LoadAudio(hindi_wav_fn)

        return str(imgpath), eng_audio_feat, hindi_audio_feat, datum['english_wav'], datum['hindi_wav'], datum['image'], datum["english_speaker"]

    def __len__(self):
        return len(self.data)

def normalize(yt):
    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = - (yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt

class flickrData(Dataset):
    # function adapted from https://github.com/dharwath

    def __init__(self, data, audio_fn, image_fn, audio_conf):

        self.data = data
        self.audio_fn = Path(audio_fn)
        self.image_fn = Path(image_fn)
        self.audio_conf = audio_conf
        
    # def _LoadAudio(self, path):

    #     audio_type = self.audio_conf.get('audio_type')
    #     if audio_type not in ['melspectrogram', 'spectrogram']:
    #         raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        
    #     preemph_coef = self.audio_conf.get('preemph_coef')
    #     sample_rate = self.audio_conf.get('sample_rate')
    #     window_size = self.audio_conf.get('window_size')
    #     window_stride = self.audio_conf.get('window_stride')
    #     window_type = self.audio_conf.get('window_type')
    #     num_mel_bins = self.audio_conf.get('num_mel_bins')
    #     target_length = self.audio_conf.get('target_length')
    #     fmin = self.audio_conf.get('fmin')
    #     n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
    #     win_length = int(sample_rate * window_size)
    #     hop_length = int(sample_rate * window_stride)

    #     # load audio, subtract DC, preemphasis
    #     y, sr = librosa.load(path, sample_rate)
    #     if y.size == 0:
    #         y = np.zeros(target_length)

    #     yt, _ = librosa.effects.trim(y, top_db=20)
    #     yt = normalize(yt)
    #     y = preemphasis(y, preemph_coef)


    #     feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_mels=80, n_fft=n_fft, hop_length=hop_length)
    #     feat[0] = librosa.feature.rms(y, hop_length=hop_length, frame_length=win_length)
    #     feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    #     # spec = np.abs(stft)**2 # Power spectrum
    #     # if audio_type == 'melspectrogram':
    #     #     mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
    #     #     melspec = np.dot(mel_basis, spec)
    #     #     logspec = librosa.power_to_db(melspec, ref=np.max)
    #     # elif audio_type == 'spectrogram':
    #     #     logspec = librosa.power_to_db(spec, ref=np.max)
    #     # # n_frames = logspec.shape[1]
    #     feat = torch.FloatTensor(feat)
    #     return feat#, n_frames

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
        return logspec#, n_frames
        
    def __getitem__(self, index):
        datum = self.data[index]
        wav_fn = self.audio_fn / Path(datum['wav'])
        imgpath = self.image_fn / Path(datum['image'])
        audio_feat = self._LoadAudio(wav_fn)

        imgpath = '/'.join(str(imgpath).split('/')[-3:])
        wav_name = Path(datum['wav']).stem

        return imgpath, audio_feat, wav_name, datum['image'], datum["speaker"]

    def __len__(self):
        return len(self.data)