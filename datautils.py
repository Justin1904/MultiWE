import sys
sys.path.append("/media/bighdd7/zhun/software/CMU-MultimodalSDK")

import mmsdk
import os
import re
import numpy as np
import torch

from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError
from collections import defaultdict
from tqdm import tqdm

DATA_PATH = "/media/bighdd7/zhun/software/CMU-MultimodalSDK/tutorial/data/"
DATASET = md.cmu_mosei


def avg_collapse(intervals, features):
    '''
    Collapse function that does averaging on numerical modalities and return text as is
    '''
    try:
        return np.average(features, axis=0)
    except:
        return features


def load_unsup_dataset(text_field='CMU_MOSEI_TimestampedWords',
                       visual_field='CMU_MOSEI_VisualOpenFace2',
                       acoustic_field='CMU_MOSEI_openSMILE_IS09',
                       collapse_function=avg_collapse):
    '''
    Loads the (unlabeled) dataset from SDK
    '''
    # basic loading and alignment
    features = [text_field, visual_field, acoustic_field]
    recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
    dataset = md.mmdataset(recipe)
    dataset.align(text_field, collapse_functions=[collapse_function])

    # getting some basic constants
    key_set = dataset.computational_sequences[text_field].data.keys()
    arbit_key = list(key_set)[0]
    #visual_dim = dataset.computational_sequences[visual_field].data[arbit_key]['features'].shape[1]
    visual_dim = 35
    #input(f"Detected visual dimension of {visual_dim}, do you want to change it? [input integer to change, input others to proceed.]")
    acoustic_dim = dataset.computational_sequences[acoustic_field].data[arbit_key]['features'].shape[1]

    # TODO: need to figure out better way here
    assert visual_dim == 35
    assert acoustic_dim == 384

    # prepare word2id map
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['<unk>']
    PAD = word2id['<pad>']

    # put the aligned dataset into video sequences
    video_level_data = dict()
    vis_miss = acs_miss = 0
    pattern = re.compile('(.*)\[(.*)\]')
    for seg in tqdm(key_set, desc="Putting word level aligned data into video sequences."):
        vid = re.search(pattern, seg).group(1)
        wid = int(re.search(pattern, seg).group(2))

        if vid not in video_level_data:
            video_level_data[vid] = []

        word = dataset.computational_sequences[text_field].data[seg]['features'][0][0]
        if word != b'sp':
            word = word.decode('utf-8')
            try:
                visual = dataset.computational_sequences[visual_field].data[seg]['features'][0, -35:]
            except KeyError:
                visual = np.zeros(visual_dim)
                vis_miss += 1

            try:
                acoustic = dataset.computational_sequences[acoustic_field].data[seg]['features'][0, :]
            except KeyError:
                acoustic = np.zeros(acoustic_dim)
                acs_miss += 1
            video_level_data[vid].append((word2id[word], visual, acoustic, wid))
    print(f"{vis_miss} words doesn't have video representation, {acs_miss} words doesn't have acoustic representation.")
    for vid in tqdm(video_level_data, desc="Re-ordering the video-level data into proper sequence."):
        video_level_data[vid].sort(key=lambda x: x[-1])
    # prevent word2id from adding new words
    word2id.default_factory = lambda: UNK
    return video_level_data, word2id 


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


def load_sup_dataset(text_field, visual_field, acoustic_field, label_field): 
    pass


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

