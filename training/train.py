#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from .util import *
from evaluation.calculations import *
from models.util import *
from losses import compute_matchmap_similarity_matrix_loss
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam
}

def train(audio_model, image_model, attention, train_loader, val_loader, args):
    # function adapted from https://github.com/dharwath

    writer = SummaryWriter(args["exp_dir"] / "tensorboard")
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    loss_tracker = valueTracking()
    info = {}
    best_epoch, best_acc = 0, 0
    global_step, start_epoch = 0, 0
    start_time = time.time()

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    attention = attention.to(device)

    model_with_params_to_update = {
        "audio_model": audio_model,
        "attention": attention
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

    [audio_model, image_model, attention], optimizer = amp.initialize(
        [audio_model, image_model, attention], optimizer, opt_level='O1'
        )

    if args["resume"] is False and args['cpc']['warm_start']: 
        print("Loading pretrained acoustic weights")
        audio_model = loadPretrainedWeights(audio_model, args)

    image_model = nn.DataParallel(image_model) if not isinstance(image_model, torch.nn.DataParallel) and args["device"] == 'cuda' else image_model

    attention = nn.DataParallel(attention) if not isinstance(attention, torch.nn.DataParallel) and args["device"] == 'cuda' else attention

    # if args["resume"]:
    #     if "restore_epoch" in args:
    #         info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpoch(
    #             args["exp_dir"], acoustic_model, english_audio_model, hindi_audio_model, image_model, optimizer, device, args["restore_epoch"]
    #             )
    #         print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')
    #     else:
    #         info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTraining(
    #             args["exp_dir"], acoustic_model, english_audio_model, hindi_audio_model, image_model, optimizer, device
    #             )
    #         print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')

    if args["resume"]:
        if "restore_epoch" in args:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
                args["exp_dir"], audio_model, image_model, attention, optimizer, amp, device, args["restore_epoch"]
                )
            print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')
        else:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
                args["exp_dir"], aaudio_model, image_model, attention, optimizer, amp, device
                )
            print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')
        if start_epoch >= 50: 
            if start_epoch >= 75: args['alphas'] = [4*val for val in args['alphas']]
            elif start_epoch >= 50: args['alphas'] = [2*val for val in args['alphas']]
            print(args['alphas'])

    start_epoch += 1
    
    for epoch in np.arange(start_epoch, args["n_epochs"] + 1):
        torch.cuda.empty_cache()
        current_learning_rate = adjust_learning_rate(args, optimizer, epoch, 0.00001)
        # current_learning_rate = args["learning_rate_scheduler"]["initial_learning_rate"]

        audio_model.train()
        image_model.train()
        attention.train()

        loss_tracker.new_epoch()
        start_time = time.time()
        printEpoch(epoch, 0, len(train_loader), loss_tracker, best_acc, start_time, start_time, current_learning_rate)
        i = 0

            
        for value_dict in train_loader:

            optimizer.zero_grad() 

            all_images = value_dict['image']
            cut = all_images.size(0)
            for pos_dict in value_dict['positives']:
                all_images = torch.cat([all_images, pos_dict["pos_image"]], dim=0)
            for neg_dict in value_dict['negatives']:
                all_images = torch.cat([all_images, neg_dict['neg_image']], dim=0)
            all_image_output = image_model(all_images.to(device))
            image_output = all_image_output[0: cut, :, :]
            pos_images = [all_image_output[cut: 2*cut, :, :], all_image_output[2*cut: 3*cut, :, :]]
            neg_images = [all_image_output[3*cut:4*cut, :, :], all_image_output[4*cut:, :, :]]

            
            english_input = value_dict["english_feat"].to(device)
            _, _, english_output = audio_model(english_input)
            english_nframes = NFrames(english_input, english_output, value_dict["english_nframes"])   

            positives = []
            for p, pos_dict in enumerate(value_dict['positives']):
                pos_image_output = pos_images[p]

                pos_english_input = pos_dict['pos_audio'].to(device)
                _, _, pos_english_output = audio_model(pos_english_input)
                pos_english_nframes = NFrames(pos_english_input, pos_english_output, pos_dict['pos_frames']) 
                
                positives.append({"image": pos_image_output, "english_output": pos_english_output, "english_nframes": pos_english_nframes})


            negatives = []
            for n, neg_dict in enumerate(value_dict['negatives']):
                neg_image_output = neg_images[n]

                neg_english_input = neg_dict['neg_audio'].to(device)
                _, _, neg_english_output = audio_model(neg_english_input)
                neg_english_nframes = NFrames(neg_english_input, neg_english_output, neg_dict['neg_frames']) 
                
                negatives.append({"image": neg_image_output, "english_output": neg_english_output, "english_nframes": neg_english_nframes})

            
            loss = compute_matchmap_similarity_matrix_loss(
                image_output, english_output, english_nframes, negatives, positives, attention,  
                margin=args["margin"], simtype=args["simtype"], alphas=args["alphas"]
                )
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item(), english_input.size(0)) #####
            end_time = time.time()
            printEpoch(epoch, i+1, len(train_loader), loss_tracker, best_acc, start_time, end_time, current_learning_rate)
            if np.isnan(loss_tracker.average):
                print("training diverged...")
                return
            global_step += 1     
            i += 1      
            if i % 1000 == 0: avg_acc = validate(audio_model, image_model, attention, val_loader, args)

        avg_acc = validate(audio_model, image_model, attention, val_loader, args)
        # avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2

        writer.add_scalar("loss/train", loss_tracker.average, epoch)
        writer.add_scalar("loss/val", avg_acc, epoch)

        # best_acc, best_epoch = saveModelAttriburesAndTraining(
        #     args["exp_dir"], acoustic_model, english_audio_model, hindi_audio_model, 
        #     image_model, optimizer, info, int(epoch), global_step, best_epoch, avg_acc, best_acc, loss_tracker.average, end_time-start_time)
        best_acc, best_epoch = saveModelAttriburesAndTrainingAMP(
                    args["exp_dir"], audio_model, 
                    image_model, attention, optimizer, info, int(epoch), global_step, best_epoch, avg_acc, best_acc, loss_tracker.average, end_time-start_time)

        if epoch == 50: 
            args['alphas'] = [2*val for val in args['alphas']]
            print(args['alphas'])
        if epoch == 75: 
            args['alphas'] = [2*val for val in args['alphas']]
            print(args['alphas'])

        
def validate(audio_model, image_model, attention, val_loader, args, cpu=False):
    # function adapted from https://github.com/dharwath
    start_time = time.time()
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    if cpu: device = torch.device("cpu")

    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    E_embeddings = [] 
    E_frame_counts = []
    
    with torch.no_grad():
        for image_input, english_input, english_nframes in tqdm(val_loader, leave=False):

            english_z, english_c, english_output = audio_model(english_input.to(device))
            E_embeddings.append(english_output.to('cpu').detach())
            E_frame_counts.append(NFrames(english_input, english_output, english_nframes).to('cpu').detach())#.cpu())

            image_output = image_model(image_input.to(device))
            I_embeddings.append(image_output.to('cpu').detach())


        heading = [" "]
        r_at_10 = []
        r_at_5 = []
        r_at_1 = []
        acc = 0
        divide = 0

        image_output = (torch.cat(I_embeddings))

        english_output = (torch.cat(E_embeddings))
        english_frames = (torch.cat(E_frame_counts))
        if args["loss"] == "matchmap": recalls = calc_recalls_IA(image_output, None, english_output, english_frames, attention, args["simtype"])
        elif args["loss"] == "pooldot": recalls = calc_recalls(image_output, None, english_output, english_frames)
        else: raise ValueError("Undefined loss.")
        heading.extend(["E -> I", "I -> E"])
        r_at_10.extend([recalls["B_to_A_r10"], recalls["A_to_B_r10"]])
        r_at_5.extend([recalls["B_to_A_r5"], recalls["A_to_B_r5"]])
        r_at_1.extend([recalls["B_to_A_r1"], recalls["A_to_B_r1"]])
        acc += recalls["B_to_A_r10"] + recalls["A_to_B_r10"]
        divide += 2

        tablePrinting(
            heading, ["R@10", "R@5", "R@1"],
            np.asarray([r_at_10, r_at_5, r_at_1])
            )
        end_time = time.time()

        days, hours, minutes, seconds = timeFormat(start_time, end_time)

        print(f'Validation took {hours:>2} hours {minutes:>2} minutes {seconds:>2} seconds')

    return acc / divide