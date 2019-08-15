import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pdb import set_trace

class _FeedForwardNN(nn.Module):
    '''
    Simple two layer network. Output is real-valued (not constrained by activations)
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(_FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        h = self.layer1(x)
        h = F.relu(h)
        h = self.layer2(h)
        h = F.relu(h)
        output = self.layer3(h)
        return output


class SkipGram(nn.Module):
    '''
    A multimodal skip-gram model that predicts context with central word
    '''
    def __init__(self, emb_dim, vocab_size, visual_dim, acoustic_dim, word2id):
        super(SkipGram, self).__init__()
        # recording the model settings
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.visual_dim = visual_dim
        self.acoustic_dim = acoustic_dim
        self.word2id = word2id

        # construct the modules
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.fixed_embed = nn.Embedding(vocab_size, emb_dim)
        self.emb2word = _FeedForwardNN(emb_dim, emb_dim*2, vocab_size, vocab_size)
        self.emb2visual = _FeedForwardNN(emb_dim, emb_dim*2, visual_dim*2, visual_dim)
        self.emb2acoustic = _FeedForwardNN(emb_dim, emb_dim*2, acoustic_dim*2, acoustic_dim)
        self.emb2emb = _FeedForwardNN(emb_dim, emb_dim*2, emb_dim*2, emb_dim)

        # construct loss weight parameters
        self.t_lambda = nn.Parameter(torch.abs(torch.randn(1)))
        self.v_lambda = nn.Parameter(torch.abs(torch.randn(1)))
        self.a_lambda = nn.Parameter(torch.abs(torch.randn(1)))
        self.rec_lambda = nn.Parameter(torch.abs(torch.randn(1)))
        
        # construct loss terms
        self.t_loss = nn.CrossEntropyLoss(reduction='none')
        self.v_loss = nn.MSELoss(reduction='none')
        self.a_loss = nn.MSELoss(reduction='none')
        self.rec_loss = nn.CosineSimilarity()

    def forward(self, center_word, context_word, context_visual, context_acoustic):
        center_emb = self.embed(center_word)
        pred_word = self.emb2word(center_emb)
        pred_visual = self.emb2visual(center_emb)
        pred_acoustic = self.emb2acoustic(center_emb)
        rec_emb = self.emb2emb(center_emb)

        # calculate losses
        loss_t = self.t_loss(pred_word, context_word).sum()
        loss_v = self.v_loss(pred_visual, context_visual).mean(-1).sum()
        loss_a = self.a_loss(pred_acoustic, context_acoustic).mean(-1).sum()
        loss_rec = - self.rec_loss(rec_emb, self.fixed_embed(center_word)).sum() # minimize negative similarity

        # merge the losses
        loss = self.t_lambda.pow(-2) * loss_t + self.v_lambda.pow(-2) * loss_v + self.a_lambda.pow(-2) * loss_a + self.rec_lambda * loss_rec + torch.log(self.t_lambda * self.v_lambda * self.a_lambda * self.rec_lambda)
        return loss, (loss_t.item(), loss_v.item(), loss_a.item(), loss_rec.item())

    def save_embedding(self, path, id2word=None):
        '''
        Save embeddings to file
        '''
        if id2word is None:
            id2word = {v: k for k, v in self.word2id.items()}
        emb_mat = self.embed.weight.cpu().data.numpy()
        with open(path, 'w+') as f:
            for wid, word in tqdm(id2word.items(), desc=f"Saving word embeddings to {path}"):
                emb = emb_mat[wid]
                emb = ' '.join(map(str, emb))
                f.write(f"{word} {emb}\n")

