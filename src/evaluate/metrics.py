"""Evaluation metrics for Collaborative Filltering with Implicit Feedback."""
from typing import Callable, Dict, List, Optional

import numpy as np

eps = 10e-3  


def dcg_at_k(y_true: np.ndarray, y_score: np.ndarray,
             k: int, pscore: Optional[np.array] = None) -> float:
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]  
    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)  #pscore默认是1

    dcg_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        dcg_score += y_true_sorted_by_score[0] / pscore_sorted_by_score[0]
        k = min(k,y_true.shape[0])
        for i in np.arange(1, k):
            dcg_score += y_true_sorted_by_score[i] / \
                (pscore_sorted_by_score[i] * np.log2(i + 2))

    final_score = dcg_score if pscore is None \
        else dcg_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray,
             k: int, pscore: Optional[np.array] = None) -> float:
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]  
    y_true_sorted_by_score2 = y_true[y_true.argsort()[::-1]]  

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)  

    dcg_score = 0.0
    idcg_score = 0.0
    
    if not np.sum(y_true_sorted_by_score) == 0:
        dcg_score += y_true_sorted_by_score[0] / pscore_sorted_by_score[0]
        
        idcg_score += y_true_sorted_by_score2[0] / pscore_sorted_by_score[0]
        k = min(k,y_true.shape[0])
        for i in np.arange(1, k):
            dcg_score += y_true_sorted_by_score[i] / \
                (pscore_sorted_by_score[i] * np.log2(i + 2))
            
            idcg_score += y_true_sorted_by_score2[i] / \
                (pscore_sorted_by_score[i] * np.log2(i + 2))
            
    
    final_score = dcg_score if pscore is None \
        else dcg_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score / idcg_score



def average_precision_at_k(y_true: np.ndarray, y_score: np.ndarray,
                           k: int, pscore: Optional[np.array] = None) -> float:
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    average_precision_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        k = min(k,y_true.shape[0])
        for i in np.arange(k):
            if y_true_sorted_by_score[i] == 1:
                average_precision_score += \
                    np.sum(y_true_sorted_by_score[:i + 1] /
                           pscore_sorted_by_score[:i + 1]) / (i + 1)

    final_score = average_precision_score if pscore is None \
        else average_precision_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray,
                k: int, pscore: Optional[np.array] = None) -> float:
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    recall_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        k = min(k,y_true.shape[0])
        recall_score = np.sum(
            y_true_sorted_by_score[:k] / pscore_sorted_by_score[:k])   

    final_score = recall_score / np.sum(y_true) if pscore is None \
        else recall_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray,
                k: int, pscore: Optional[np.array] = None) -> float:
    y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            pscore[y_score.argsort()[::-1]], eps)
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    precision_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        k = min(k,y_true.shape[0])
        precision_score = np.sum(
            y_true_sorted_by_score[:k] / pscore_sorted_by_score[:k])   

    final_score = precision_score / k if pscore is None \
        else precision_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])

    return final_score