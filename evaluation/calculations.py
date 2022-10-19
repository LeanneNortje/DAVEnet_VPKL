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
import sys
import os
from losses.util import *
import warnings
warnings.filterwarnings("ignore")

def calc_recalls_IA(A, A_mask, B, B_mask, attention, rank, simtype='MISA'):
    # function adapted from https://github.com/dharwath

    S = compute_matchmap_similarity_matrix_IA(A, A_mask, B, B_mask, attention, simtype)
    n = S.size(0)
    S = S.to(rank)
    A2B_scores, A2B_ind = S.topk(10, 1)
    B2A_scores, B2A_ind = S.topk(10, 0)

    A2B_scores = A2B_scores.detach().cpu().numpy()
    A2B_ind = A2B_ind.detach().cpu().numpy()
    B2A_scores = B2A_scores.detach().cpu().numpy()
    B2A_ind = B2A_ind.detach().cpu().numpy()

    A_foundind = -np.ones(n)
    B_foundind = -np.ones(n)
    for i in tqdm(range(n), desc="Calculating recalls", leave=False):
        ind = np.where(A2B_ind[i, :] == i)[0]
        if len(ind) != 0: B_foundind[i] = ind[0]
        ind = np.where(B2A_ind[:, i] == i)[0]
        if len(ind) != 0: A_foundind[i] = ind[0]
 
    r1_A_to_B = len(np.where(B_foundind == 0)[0])/len(B_foundind)
    r5_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 5))[0])/len(B_foundind)
    r10_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 10))[0])/len(B_foundind)

    r1_B_to_A = len(np.where(A_foundind == 0)[0])/len(A_foundind)
    r5_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 5))[0])/len(A_foundind)
    r10_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 10))[0])/len(A_foundind)

    return {
        'A_to_B_r1':r1_A_to_B, 
        'A_to_B_r5':r5_A_to_B, 
        'A_to_B_r10':r10_A_to_B,
        'B_to_A_r1':r1_B_to_A, 
        'B_to_A_r5':r5_B_to_A, 
        'B_to_A_r10':r10_B_to_A
        }
# def calc_recalls_IA(A, A_mask, B, B_mask, target, all_ids, attention, simtype='MISA', rank=0):
#     # function adapted from https://github.com/dharwath
#     A = A.view(A.size(0), A.size(1), -1).transpose(1, 2)
#     n = A.size(0)
#     S = []
#     # A = A.to(rank)
#     # B = B.to(rank)
#     # B_mask = B_mask.to(rank)
#     for i_A in tqdm(range(n), desc='Calculating similarity', leave=False):
#         scores = []
#         for i_B in range(n):
#             score, _ = attention(A[i_A, :, :].unsqueeze(0).to(rank), B[i_B, :, :].unsqueeze(0).to(rank), B_mask[i_B].to(rank))#compute_matchmap_similarity_score_IA(A[i_A, :, :].unsqueeze(0), A_mask, B[i_B, :, :].unsqueeze(0), B_mask[i_B], attention, simtype)
#             scores.append(score)
#         scores = torch.cat(scores, dim=0)
#         S.append(scores.unsqueeze(0))
#     S = torch.cat(S, dim=0)
#     I2A_scores, I2A_ind = S.topk(10, 1)
#     A2I_scores, A2I_ind = S.topk(10, 0)

#     I2A_scores = I2A_scores.detach().cpu().numpy()
#     I2A_ind = I2A_ind.detach().cpu().numpy()
#     A2I_scores = A2I_scores.detach().cpu().numpy()
#     A2I_ind = A2I_ind.detach().cpu().numpy()

#     # A_foundind = -np.ones(n)
#     # B_foundind = -np.ones(n)
#     # for i in tqdm(range(n), desc="Calculating recalls", leave=False):
#     #     ind = np.where(A2B_ind[i, :] == i)[0]
#     #     if len(ind) != 0: B_foundind[i] = ind[0]
#     #     ind = np.where(B2A_ind[:, i] == i)[0]
#     #     if len(ind) != 0: A_foundind[i] = ind[0]
 
#     # r1_A_to_B = len(np.where(B_foundind == 0)[0])/len(B_foundind)
#     # r5_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 5))[0])/len(B_foundind)
#     # r10_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 10))[0])/len(B_foundind)

#     # r1_B_to_A = len(np.where(A_foundind == 0)[0])/len(A_foundind)
#     # r5_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 5))[0])/len(A_foundind)
#     # r10_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 10))[0])/len(A_foundind)

#     r1_I2A = np.zeros(n)
#     r5_I2A = np.zeros(n)
#     r10_I2A = np.zeros(n)

#     r1_A2I = np.zeros(n)
#     r5_A2I= np.zeros(n)
#     r10_A2I = np.zeros(n)

#     for i in tqdm(range(n), desc="Calculating recalls"):
#         id = target[i].item()
#         indices_found = -np.ones(10)
#         for j in range(10):

#             if all_ids[I2A_ind[i, j], id] == 1: 
#                 indices_found[j] = I2A_ind[i, j]

#             # if id == target[I2A_ind[i, j]]: 
#             #     indices_found[j] = I2A_ind[i, j]
#         if indices_found[0] != -1: r1_I2A[i] = 1
#         if (indices_found[0:5] != -1).any(): r5_I2A[i] = 1
#         if (indices_found != -1).any(): r10_I2A[i] = 1
            
#         indices_found = -np.ones(10)
#         for j in range(10):

#             if all_ids[A2I_ind[j, i], id] == 1: 
#                 indices_found[j] = A2I_ind[j, i]

#             # if id == target[A2I_ind[j, i]]: 
#             #     indices_found[j] = A2I_ind[j, i]
#         if indices_found[0] != -1: r1_A2I[i] = 1
#         if (indices_found[0:5] != -1).any(): r5_A2I[i] = 1
#         if (indices_found != -1).any(): r10_A2I[i] = 1


#     r1_I2A = r1_I2A.mean()
#     r5_I2A = r5_I2A.mean()
#     r10_I2A = r10_I2A.mean()

#     r1_A2I = r1_A2I.mean()
#     r5_A2I = r5_A2I.mean()
#     r10_A2I = r10_A2I.mean()

#     return {
#         'r1_I2A':r1_I2A, 
#         'r5_I2A':r5_I2A, 
#         'r10_I2A':r10_I2A,
#         'r1_A2I':r1_A2I, 
#         'r5_A2I':r5_A2I, 
#         'r10_A2I':r10_A2I
#         }

def calc_recalls_AA(A, A_mask, B, B_mask, simtype='MISA', rank=0):
    # function adapted from https://github.com/dharwath

    S = compute_matchmap_similarity_matrix_AA(A, A_mask, B, B_mask, simtype).to(rank)
    n = S.size(0)
    A2B_scores, A2B_ind = S.topk(10, 1)
    B2A_scores, B2A_ind = S.topk(10, 0)

    A2B_scores = A2B_scores.detach().cpu().numpy()
    A2B_ind = A2B_ind.detach().cpu().numpy()
    B2A_scores = B2A_scores.detach().cpu().numpy()
    B2A_ind = B2A_ind.detach().cpu().numpy()

    A_foundind = -np.ones(n)
    B_foundind = -np.ones(n)
    for i in range(n):
        ind = np.where(A2B_ind[i, :] == i)[0]
        if len(ind) != 0: B_foundind[i] = ind[0]
        ind = np.where(B2A_ind[:, i] == i)[0]
        if len(ind) != 0: A_foundind[i] = ind[0]
 
    r1_A_to_B = len(np.where(B_foundind == 0)[0])/len(B_foundind)
    r5_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 5))[0])/len(B_foundind)
    r10_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 10))[0])/len(B_foundind)

    r1_B_to_A = len(np.where(A_foundind == 0)[0])/len(A_foundind)
    r5_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 5))[0])/len(A_foundind)
    r10_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 10))[0])/len(A_foundind)

    return {
        'A_to_B_r1':r1_A_to_B, 
        'A_to_B_r5':r5_A_to_B, 
        'A_to_B_r10':r10_A_to_B,
        'B_to_A_r1':r1_B_to_A, 
        'B_to_A_r5':r5_B_to_A, 
        'B_to_A_r10':r10_B_to_A
        }