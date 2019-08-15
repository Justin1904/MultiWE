import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn

from Models import SkipGram
from Trainer import EmbeddingTrainer, Logger
from datautils import load_unsup_dataset, extract_context_pairs, avg_collapse, custom_collate, load_emb
from torch.utils.data import DataLoader
from torch.optim import Adam
from subprocess import check_call
from collections import defaultdict
from argparse import ArgumentParser


#VISUAL_FIELD = "CMU_MOSI_OpenFace"
#ACOUSTIC_FIELD = "CMU_MOSI_openSMILE_IS09"
#TEXT_FIELD = "CMU_MOSI_ModifiedTimestampedWords"
CACHE_GLOVE_PATH = "./data/glove_cache.pt"
CACHE_DATA_PATH = "./data/data_cache.pt"
TEXT_FIELD = "CMU_MOSEI_TimestampedWords"
ACOUSTIC_FIELD = "CMU_MOSEI_openSMILE_IS09"
VISUAL_FIELD = "CMU_MOSEI_VisualOpenFace2"
CONTEXT_SIZE = 5
VISUAL_DIM = 35
ACOUSTIC_DIM = 384
EMB_DIM = 300
LOG_PATH = "./logs/" + str(random.randint(10000, 100000)) + "/"
UNK = 0
BATCH_SIZE = 64

# seed the training manually, after randomly generating the ID
random.seed(123)
torch.manual_seed(128)
torch.cuda.manual_seed_all(123)
np.random.seed(123)

if __name__ == "__main__":
    # parse some args
    parser = ArgumentParser(description="Options for training script.")
    parser.add_argument('--purge-data-cache', action='store_true', default=False)
    parser.add_argument('--purge-glove-cache', action='store_true', default=False)
    args = parser.parse_args()

    # remove the corrupt cached files if specified
    if args.purge_data_cache and os.path.exists(CACHE_DATA_PATH):
        check_call('rm ' + CACHE_DATA_PATH, shell=True)

    if args.purge_glove_cache and os.path.exists(CACHE_GLOVE_PATH):
        check_call('rm ' + CACHE_GLOVE_PATH, shell=True)

    # get the data prepared
    if os.path.exists(CACHE_DATA_PATH):
        dataset, w2i = torch.load(CACHE_DATA_PATH)
        w2i = defaultdict(lambda: UNK, w2i)
    else:
        print("The data file does not exist, please download and put the data in the correct folder first.")

    training_data = extract_context_pairs(dataset, CONTEXT_SIZE)
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=4, pin_memory=True)

    # determine the device
    if torch.cuda.is_available:
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    # prepare the model
    model = SkipGram(EMB_DIM, len(w2i), VISUAL_DIM, ACOUSTIC_DIM, w2i)
    if os.path.exists(CACHE_GLOVE_PATH):
        emb_mat = torch.load(CACHE_GLOVE_PATH)
    else:
        print("The embedding file does not exist, please download and put the data in the correct folder first.")
    
    model.embed.weight.data = emb_mat
    model.fixed_embed.weight.data = emb_mat

    model.fixed_embed.requires_grad = False
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    #    print(f"Using {torch.cuda.device_count()} GPUs")
    model.to(device)
    optimizer = Adam(filter(lambda x: x.requires_grad, model.parameters()))

    # prepare logging and checkpointing folders
    check_call('mkdir -p ' + LOG_PATH, shell=True)
    logger = Logger(LOG_PATH, 'train.log')
    loss, weights, details = EmbeddingTrainer.fit(model, train_loader, optimizer, device, logger, cache_path=LOG_PATH)
    #loss = EmbeddingTrainer.fit(model, train_loader, optimizer, device, logger, cache_path=LOG_PATH)
    
    # save the detailed losses and weights
    torch.save((details, weights), LOG_PATH+'training_history.pt')
