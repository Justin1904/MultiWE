import sys
import os
import re
import numpy as np
import torch

from subprocess import check_call, CalledProcessError
from collections import defaultdict
from tqdm import tqdm


def avg_collapse(intervals, features):
    '''
    Collapse function that does averaging on numerical modalities and return text as is
    '''
    try:
        return np.average(features, axis=0)
    except:
        return features


def get_context_idx(idx, context_size, total_len):
    '''
    Given current index, context size and total sequence length, calculate the indices for context locations
    '''
    context_lower = max(0, idx - context_size)
    context_upper = min(total_len, idx + context_size + 1)
    context_idx = list(range(context_lower, idx)) + list(range(idx + 1, context_upper))
    return context_idx


def extract_context_pairs(video_level_data, context_size=5):
    '''
    From loaded, unsupervised video level data, extract context pairs for training word embeddings
    '''
    training_data = []
    for vid, data in tqdm(video_level_data.items(), desc="Extracting center-context pairs from data."):
        total_len = len(data)
        for idx in range(total_len):
            center = data[idx][:-1]
            context_idx = get_context_idx(idx, context_size, total_len)
            data_pairs = [(center, data[con_idx][:-1]) for con_idx in context_idx]
            training_data.extend(data_pairs)
    return training_data


def custom_collate(batch):
    '''
    Collates the data.
    Expected batch format: [((center_word_idx, center_visual, center_acoustic), (context_word_idx, context_visual, context_acoustic))] * n
    '''
    center_words = torch.LongTensor([pair[0][0] for pair in batch])
    context_words = torch.LongTensor([pair[1][0] for pair in batch])
    
    center_visual = torch.FloatTensor([pair[0][1] for pair in batch]).float()
    context_visual = torch.FloatTensor([pair[1][1] for pair in batch]).float()

    center_acoustic = torch.FloatTensor([pair[0][2] for pair in batch]).float()
    context_acoustic = torch.FloatTensor([pair[1][2] for pair in batch]).float()
    return (center_words, center_visual, center_acoustic, context_words, context_visual, context_acoustic)


def load_emb(w2i, emb_path, embedding_size=300, embedding_vocab=2196017):
    emb_mat = np.random.randn(len(w2i), embedding_size)
    found = 0
    with open(emb_path, 'r') as f:
        for line in tqdm(f, total=embedding_vocab, desc="Loading embedding"):
            content = line.strip().split()
            vector = np.asarray([float(x) for x in content[-300:]])
            word = ' '.join(content[:-300])
            if word in w2i:
                idx = w2i[word]
                emb_mat[idx, :] = vector
                found += 1
    print(f"Found {found} words in the pretrained embedding file.")
    return torch.tensor(emb_mat).float()

