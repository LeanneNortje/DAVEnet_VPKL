#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from losses import compute_matchmap_similarity_matrix_loss
from losses.util import *
from dataloaders import *
from models.setup import *
from models.util import *
from models.multimodalModels import *
from models.GeneralModels import *
from evaluation.calculations import *
from training.util import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

def validate(audio_model, image_model, attention, contrastive_loss, val_loader, rank, args):
    # function adapted from https://github.com/dharwath
    start_time = time.time()

    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    E_embeddings = [] 
    H_embeddings = [] 
    E_frame_counts = []
    H_frame_counts = []
    i = 0
    print('\n')
    with torch.no_grad():
        for value_dict in tqdm(val_loader):

            image_output = image_model(value_dict['image'].to(rank))
            # image_output = image_output.view(image_output.size(0), image_output.size(1), -1).transpose(1, 2)
            I_embeddings.append(image_output.cpu().detach())

            english_input = value_dict["english_feat"].to(rank)
            _, _, english_output = audio_model(english_input)
            E_embeddings.append(english_output.cpu().detach())
            E_frame_counts.append(NFrames(english_input, english_output, value_dict["english_nframes"]).cpu().detach())

        heading = [" "]
        r_at_10 = []
        r_at_5 = []
        r_at_1 = []
        acc = 0
        divide = 0

        image_output = torch.cat(I_embeddings, dim=0)#.to(rank)
        english_output = torch.cat(E_embeddings, dim=0)#.to(rank)
        english_frames = torch.cat(E_frame_counts, dim=0)#.to(rank)
        recalls = calc_recalls_IA(image_output, None, english_output, english_frames, attention, rank, args["simtype"])

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

def spawn_training(rank, world_size, image_base, args):

    # # Create dataloaders
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD,
    )
    torch.manual_seed(42)

    if rank == 0: writer = SummaryWriter(args["exp_dir"] / "tensorboard")
    best_epoch, best_acc = 0, 0
    global_step, start_epoch = 0, 0
    info = {}
    loss_tracker = valueTracking()

    if rank == 0: heading(f'\nLoading training data ')
    train_dataset = ImageAudioDatawithMasks(image_base, args["data_train"], Path("data/train_image_mask_lookup.npz"), args, rank)
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


    if rank == 0: 
        heading(f'\nLoading validation data ')
        args["image_config"]["center_crop"] = True
        validation_loader = torch.utils.data.DataLoader(
            ImageAudioDatawithMasksVal(image_base, args["data_val"], Path("data/val_image_mask_lookup.npz"), args, rank),
            batch_size=args["batch_size"], shuffle=False, num_workers=1, pin_memory=True)
    

    if rank == 0: heading(f'\nSetting up Audio model ')
    audio_model = mutlimodal(args).to(rank)

    if rank == 0: heading(f'\nSetting up image model ')
    image_model_name = imageModel(args)
    image_model = image_model_name(args, pretrained=args["pretrained_image_model"]).to(rank)

    if rank == 0: heading(f'\nSetting up attention model ')
    attention = ScoringAttentionModule(args).to(rank)

    if rank == 0: heading(f'\nSetting up contrastive loss ')
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

    # scaler = torch.cuda.amp.GradScaler()

    if args["resume"] is False and args['cpc']['warm_start']: 
        if rank == 0: print("Loading pretrained acoustic weights")
        audio_model = loadPretrainedWeights(audio_model, args, rank)

    audio_model = DDP(audio_model, device_ids=[rank])
    image_model = DDP(image_model, device_ids=[rank])
    # attention = DDP(attention, device_ids=[rank])
    # contrastive_loss = DDP(contrastive_loss, device_ids=[rank])

    if args["resume"]:

        if "restore_epoch" in args:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
                args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, args["restore_epoch"]
                )
            if rank == 0: print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')
        else:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
                args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank
                )
            if rank == 0: print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')

    start_epoch += 1

    for epoch in np.arange(start_epoch, args["n_epochs"] + 1):
        train_sampler.set_epoch(int(epoch))
        current_learning_rate = adjust_learning_rate(args, optimizer, epoch, 0.00001)

        audio_model.train()
        image_model.train()
        attention.train()
        contrastive_loss.train()

        loss_tracker.new_epoch()
        start_time = time.time()
        if rank == 0: printEpoch(epoch, 0, len(train_loader), loss_tracker, best_acc, start_time, start_time, current_learning_rate)
        
        i = 0
                    
        for value_dict in train_loader:
            
            optimizer.zero_grad() 

            with torch.cuda.amp.autocast():
        
                image_output = image_model(value_dict['image'].to(rank))
                
                english_input = value_dict["english_feat"].to(rank)
                _, _, english_output = audio_model(english_input)
                english_nframes = NFrames(english_input, english_output, value_dict["english_nframes"])   

                loss = compute_matchmap_similarity_matrix_loss(
                    image_output, english_output, english_nframes, attention, contrastive_loss, #audio_attention,  
                    margin=args["margin"], simtype=args["simtype"], alphas=args["alphas"], rank=rank
                    )
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.detach().item(), english_input.detach().size(0)) #####
            end_time = time.time()
            if rank == 0: printEpoch(epoch, i+1, len(train_loader), loss_tracker, best_acc, start_time, end_time, current_learning_rate)
            if np.isnan(loss_tracker.average):
                print("training diverged...")
                return
            global_step += 1 
            # if i == 200: break
            # break
            i += 1
        # break

        if rank == 0:
            # avg_acc = validate_contrastive(audio_model, image_model, attention, contrastive_loss, validation_loader, rank, args)
            avg_acc = validate(audio_model, image_model, attention, contrastive_loss, validation_loader, rank, args)
            
            writer.add_scalar("loss/train", loss_tracker.average, epoch)
            writer.add_scalar("loss/val", avg_acc, epoch)

            best_acc, best_epoch = saveModelAttriburesAndTrainingAMP(
                args["exp_dir"], audio_model, 
                image_model, attention, contrastive_loss, optimizer, info, int(epoch), global_step, best_epoch, avg_acc, best_acc, loss_tracker.average, end_time-start_time)
        
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action="store_true", dest="resume",
            help="load from exp_dir if True")
    parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'],
            help="Model config file.")
    parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to resore training from.")
    parser.add_argument("--image-base", default="..", help="Path to images.")
    command_line_args = parser.parse_args()

    # Setting up model specifics
    heading(f'\nSetting up model files ')
    args, image_base = modelSetup(command_line_args)

    world_size = torch.cuda.device_count()
    mp.spawn(
        spawn_training,
        args=(world_size, image_base, args),
        nprocs=world_size,
        join=True,
    )