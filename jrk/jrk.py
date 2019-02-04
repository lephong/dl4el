import torch
import torch.nn as nn
from jrk.encoder import Encoder
import numpy
numpy.set_printoptions(threshold=numpy.nan)

class JRK(nn.Module):
    def __init__(self, config):
        super(JRK, self).__init__()

        self.encoder = Encoder(config={
            'type': config['type'],
            'lstm_hiddim': config['lstm_hiddim'],
            'n_filters': config['n_filters'],
            'filter_sizes': config['filter_sizes'],
            'word_embs': config['word_embs'],
            'pos_embdim': config['pos_embdim'],
            'dropout': config['dropout'],
            'en_dim': config['en_dim'],
            'n_rels': config['n_rels']})

        self.ent_pair2id = config['ent_pair2id']
        self.ent_pair_rel_scores = nn.Embedding(len(self.ent_pair2id), config['n_rels'])
        self.ent_pair_rel_scores.weight.data.fill_(-1)
        self.ent_pair_rel_scores.weight.requires_grad = False

    def init_with_kb(self, triples):
        kb = self.ent_pair_rel_scores.weight.data
        kb.fill_(-10)
        self.init_kb = torch.zeros(kb.shape)

        for t in triples:
            ent_pair = self.ent_pair2id[(t[0], t[2])]
            if t[1] != 0:
                kb[ent_pair, t[1]] = 5
                self.init_kb[ent_pair, t[1]] == 1
            else:
                kb[ent_pair, 0] = 5
                self.init_kb[ent_pair, 0] = 1
        self.init_kb = self.init_kb.cuda()

    def forward(self, input):
        p_not_na, p_rel_not_na, reprs = self.encoder(input)
        ent_pair_rel = torch.sigmoid(self.ent_pair_rel_scores(input['ent_pair']))
        probs = (ent_pair_rel[:, 1:] * p_rel_not_na).sum(dim=1)
        return probs, p_not_na, p_rel_not_na, reprs, ent_pair_rel

    def compute_loss(self, input, regularity='prob'):
        probs, p_not_na, reg_coef, ent_pair_rel = input['probs'], input['p_not_na'], input['reg_coef'], input['ent_pair_rel']
        reg = torch.zeros(1).cuda()

        # compute kl
        if regularity == 'kl':
            p_not_na_total = p_not_na.sum()
            p_na_total = (1 - p_not_na).sum()
            p_not_na_total /= p_not_na_total + p_na_total
            p_na_total /= p_not_na_total + p_na_total

            prior = torch.Tensor([0.7, 0.3]).cuda()
            kl = p_na_total * torch.log(p_na_total / prior[0] + 1e-10) + p_not_na_total * torch.log(p_not_na_total / prior[1] + 1e-10)
            reg += reg_coef * kl
        elif regularity == 'prob':
            reg += reg_coef * p_not_na.sum() / p_not_na.shape[0]

        if self.ent_pair_rel_scores.weight.requires_grad == True:
            mask = ent_pair_rel.le(0.9).float()
            reg_kb = -(ent_pair_rel * mask * nn.functional.embedding(input['ent_pair'], self.init_kb)).sum() / ent_pair_rel.shape[0]
            reg += 0.1 * reg_kb

        # compute
        loss = (-torch.log(probs + 1e-10)).mean() + reg
        return loss, reg
