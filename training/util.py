#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import json
import torch
# from alive_progress import alive_bar
import os
from os import popen
from math import ceil
from itertools import chain

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)

def getParameters(models, to_freeze, args):
    valid_models = []
    for model_name in models:
        valid_models.append(
            {
            'params': models[model_name].parameters(),
            'lr': args["learning_rate_scheduler"]["initial_learning_rate"],
            'name': model_name
            }
            )

    for model_name in to_freeze:
        for n, p in to_freeze[model_name].named_parameters(): 
            if n.startswith('embedder'):
                valid_models.append(
                {
                'params': p,
                'lr': args["learning_rate_scheduler"]["initial_learning_rate"],
                'name': model_name + "_" + n
                }
                )

    return valid_models 

def saveModelAttriburesAndTrainingAMP(
    exp_dir, audio_model, image_model, attention, contrastive_loss, optimizer, #amp,
    info, epoch, global_step, best_epoch, acc, best_acc, loss, epoch_time
    ):
    
    overwrite_best_ckpt = False
    if acc > best_acc:
        best_epoch = epoch
        best_acc = acc
        overwrite_best_ckpt = True

    # assert int(epoch) not in info
    info[int(epoch)] = {
        "global_step": global_step,
        "best_epoch": best_epoch,
        "acc": acc,
        "best_acc": best_acc,
        "loss": loss,
        "epoch_time": epoch_time
    }
    with open(exp_dir / "training_metadata.json", "w") as f:
        json.dump(info, f)

    checkpoint = {
        "audio_model": audio_model.state_dict(),
        "image_model": image_model.state_dict(),
        "attention": attention.state_dict(),
        "contrastive_loss": contrastive_loss.state_dict(),
        "optimizer": optimizer.state_dict(),
        # "amp": amp.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "acc": acc,
        "best_acc": best_acc,
        "loss": loss,
        "epoch_time": epoch_time
    }

    if not os.path.isdir(exp_dir / "models"): os.makedirs(exp_dir / "models")
    torch.save(checkpoint, exp_dir / "models" / "last_ckpt.pt")
    if overwrite_best_ckpt:
        torch.save(checkpoint, exp_dir / "models" / "best_ckpt.pt")
    torch.save(checkpoint, exp_dir / "models" / f'epoch_{epoch}.pt')

    return best_acc, best_epoch

def loadModelAttriburesAndTrainingAMP(
    exp_dir, audio_model, image_model, attention, contrastive_loss, 
    optimizer, rank, last_not_best=True
    ):

    info_fn = exp_dir / "training_metadata.json"
    with open(info_fn, "r") as f:
        info = json.load(f)

    if last_not_best:
        checkpoint_fn = exp_dir / "models" / "last_ckpt.pt"
    else:
        checkpoint_fn = exp_dir / "models" / "best_ckpt.pt"

    checkpoint = torch.load(checkpoint_fn, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
    
    audio_model.load_state_dict(checkpoint["audio_model"])
    image_model.load_state_dict(checkpoint["image_model"])
    attention.load_state_dict(checkpoint["attention"])
    contrastive_loss.load_state_dict(checkpoint["contrastive_loss"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # amp.load_state_dict(checkpoint["amp"])
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_epoch = checkpoint["best_epoch"]
    best_acc = checkpoint["best_acc"]  
    
    return info, epoch, global_step, best_epoch, best_acc


def loadModelAttriburesAndTrainingAtEpochAMP(
    exp_dir, audio_model, image_model, attention, contrastive_loss, 
    optimizer, rank, load_epoch
    ):

    info_fn = exp_dir / "training_metadata.json"
    with open(info_fn, "r") as f:
        info = json.load(f)

    checkpoint_fn = exp_dir / "models" / f'epoch_{load_epoch}.pt'

    checkpoint = torch.load(checkpoint_fn, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})

    audio_model.load_state_dict(checkpoint["audio_model"])
    image_model.load_state_dict(checkpoint["image_model"])
    attention.load_state_dict(checkpoint["attention"])
    contrastive_loss.load_state_dict(checkpoint["contrastive_loss"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # amp.load_state_dict(checkpoint["amp"])
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_epoch = checkpoint["best_epoch"]
    best_acc = checkpoint["best_acc"]  
    print(f'\nLoading model parameters from:\n\t\t{checkpoint_fn}')

    return info, epoch, global_step, best_epoch, best_acc    


class valueTracking(object):
    # function adapted from https://github.com/dharwath

    def __init__(self):
        self.average = 0
        self.sum = 0
        self.num_values = 0
        self.epoch_average = 0
        self.epoch_sum = 0
        self.num_epoch_values = 0

    def update(self, value, n=1):

        self.sum += value * n
        self.num_values += n
        self.average = self.sum / self.num_values

        self.epoch_sum += value * n
        self.num_epoch_values += n
        self.epoch_average = self.epoch_sum / self.num_epoch_values

    def new_epoch(self):
        self.epoch_average = 0
        self.epoch_sum = 0
        self.num_epoch_values = 0


def floatFormat(number):
    return f'{number:.6f}' 

def timeFormat(start_time, end_time):   

    total_time = end_time-start_time

    days = total_time // (24 * 60 * 60) 
    total_time = total_time % (24 * 60 * 60)

    hours = total_time // (60 * 60)
    total_time = total_time % (60 * 60)

    minutes = total_time // 60
    
    seconds =total_time % (60)

    return int(days), int(hours), int(minutes), int(seconds)

def getCharacter(remaining, num_steps_in_single_width):
    if remaining >= 0 and remaining < num_steps_in_single_width*1/7: return "\u258F"
    elif remaining >= num_steps_in_single_width*1/7 and remaining < num_steps_in_single_width*2/7: return "\u258E"
    elif remaining >= num_steps_in_single_width*2/7 and remaining < num_steps_in_single_width*3/7: return "\u258D"
    elif remaining >= num_steps_in_single_width*3/7 and remaining < num_steps_in_single_width*4/7: return "\u258C"
    elif remaining >= num_steps_in_single_width*4/7 and remaining < num_steps_in_single_width*5/7: return "\u258B"
    elif remaining >= num_steps_in_single_width*5/7 and remaining < num_steps_in_single_width*6/7: return "\u258A"
    elif remaining >= num_steps_in_single_width*6/7 and remaining < num_steps_in_single_width: return "\u2589"

def printEpoch(epoch, step, num_steps, loss_tracker, best_acc, start_time, end_time, lr):
    
    terminal_width = tuple(os.get_terminal_size())
    terminal_width = terminal_width[0]

    days, hours, minutes, seconds = timeFormat(start_time, end_time)

    column_separator = ' | '
    epoch_string = f'Epoch: {epoch:<4} |'
    epoch_info_string = f'Loss: ' + floatFormat (loss_tracker.epoch_average) + column_separator 
    epoch_info_string +=  f'Average loss: ' + floatFormat(loss_tracker.average) + column_separator
    epoch_info_string += f'Best accuracy: ' + floatFormat(best_acc) + column_separator
    epoch_info_string += f'LR: ' + floatFormat(lr) + column_separator
    epoch_info_string += f'Epoch time: {hours:>2} hours  {minutes:>2} minutes  {seconds:>2} seconds'
    step_count_string = f'| [{step:>{len(str(num_steps))}}/{num_steps}]' 

    animation_width = len(epoch_string) + len(epoch_info_string) + len(column_separator) + len(step_count_string) + 1
    if animation_width >= terminal_width:
        # print(
        #     epoch_string + f'{int((step/num_steps)*100)}%' + step_count_string + column_separator + epoch_info_string, end=end_character
        #     )
        animation_width = terminal_width - len(step_count_string) - len(epoch_string) - 1
        num_steps_in_single_width = (num_steps) / animation_width #num_steps // animation_width if num_steps >= animation_width else num_steps / animation_width
        animation_progress = "\u2588"*ceil(step / num_steps_in_single_width)
        remaining = float(step % num_steps_in_single_width)
        animation_progress += getCharacter(remaining, num_steps_in_single_width)
        animation_blank = "-"*int(animation_width - len(animation_progress))

        if len(epoch_info_string) >= terminal_width:
            rewind = "\033[A"*3 if step != 0 else ""
            parts = epoch_info_string.split(column_separator)
            print(rewind + epoch_string + animation_progress + animation_blank + step_count_string)
            print(column_separator.join(parts[0:3]))
            print(column_separator.join(parts[3:]))
        else:
            rewind = "\033[A"*2 if step != 0 else ""
            print(
                rewind + epoch_string + animation_progress + animation_blank + step_count_string
                )
            print(epoch_info_string)

    else:
        rewind = "\033[A" if step != 0 else ""
        animation_width = terminal_width - animation_width
        num_steps_in_single_width = num_steps / animation_width #num_steps // animation_width if num_steps >= animation_width else num_steps / animation_width
        animation_progress = "\u2588"*ceil(step / num_steps_in_single_width)
        remaining = float(step % num_steps_in_single_width)
        animation_progress += getCharacter(remaining, num_steps_in_single_width)
        animation_blank = "-"*int(animation_width - len(animation_progress))
        print(
            rewind + epoch_string + animation_progress + animation_blank + step_count_string + column_separator + epoch_info_string
            )

def tablePrinting(headings, row_headings, values):

    assert(len(headings) - 1 == values.shape[-1])
    assert(len(row_headings) == values.shape[0])

    column_width = 10

    heading = f''
    for i, a_heading in enumerate(headings):
        heading += f'{a_heading:<{column_width}}'
        if i != len(headings) - 1: heading += ' | '
    else: heading += '   '

    print("\t" + heading, flush=True)
    print(f'\t{"-"*len(heading)}', flush=True)

    for i in range(len(values)):
        row = f'\t{row_headings[i]:<{column_width}}'
        for j in range(values.shape[-1]):
            value = floatFormat(values[i, j])
            row += f' | {value:>{column_width}}'
        print(row, flush=True)

def adjust_learning_rate(args, optimizer, epoch, lr):
    
    scheduler = args['learning_rate_scheduler']
    assert(len(scheduler['num_epochs']) == len(scheduler['learning_rates']))

    lr_ind = len(scheduler['num_epochs'])
    for i, max_epochs in enumerate(scheduler['num_epochs']): 
        if epoch <= max_epochs: 
            lr_ind = i
            break

    if lr_ind == len(scheduler['num_epochs']):
        lr = scheduler['learning_rates'][-1] * (scheduler['decay_factor'] ** ((epoch - scheduler['num_epochs'][-1]) // scheduler['decay_every_n_epochs']))
    else:
        lr = scheduler['learning_rates'][lr_ind]
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def NFrames(audio_input, audio_output, nframes, with_torch=True):
    pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
    if with_torch: pooling_ratio = torch.tensor(pooling_ratio, dtype=torch.int32)
    nframes = nframes.float()
    nframes.div_(pooling_ratio)
    nframes = nframes.int()
    zeros = (nframes == 0).nonzero()
    if zeros.nelement() != 0: nframes[zeros[:, 0]] += 1

    return nframes