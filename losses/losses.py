#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import math
import pickle
import numpy as np
import torch
from .util import *

def sampled_triplet_loss_from_S(S, margin):

    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    # S = S / ((S.max(dim=1)).values) ####
    positive_scores = S.diag()
    imp_indices = np.random.randint(0, N-1, size=N)
    ind_to_change = np.where(imp_indices[0:-1] >= np.arange(0, N-1))[0]
    imp_indices[ind_to_change] += 1
    # for j, ind in enumerate(imp_indices):
    #     if ind >= j:
    #         imp_indices[j] = ind + 1
    imposter_scores = S[range(N), imp_indices]
    loss = (imposter_scores - positive_scores + margin).clamp(min=0).mean()
    return loss

def semihardneg_triplet_loss_from_S(S, margin):

    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    
    # S = S / ((S.max(dim=1)).values)
    sampled_loss = sampled_triplet_loss_from_S(S, margin)
    N = S.size(0) 
    positive_scores = S.diag()
    mask = ((S - S.diag().view(-1,1)) < 0).float().detach()
    imposter_scores = (S * mask).max(dim=1).values
    loss = (imposter_scores - positive_scores + margin).clamp(min=0).mean()

    # loss = torch.max(zero, margin + imposter_scores - positive_scores)
    return loss + sampled_loss

def compute_matchmap_similarity_matrix_loss(
    image_outputs, english_output, english_nframes, attention, contrastive_loss, 
    margin, simtype, alphas, rank):
    alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6 = alphas
    loss = 0
    S = compute_matchmap_similarity_matrix_IA(image_outputs, None, english_output, english_nframes, attention, simtype)
    I2E_sampled_loss = semihardneg_triplet_loss_from_S(S, margin)
    E2I_sampled_loss = semihardneg_triplet_loss_from_S(S.t(), margin)
    loss += ((alpha_1*I2E_sampled_loss) + (alpha_2*E2I_sampled_loss))

    return loss
