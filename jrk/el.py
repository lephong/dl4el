import torch
import torch.nn as nn
from jrk.el_encoder import ELEncoder
import numpy
numpy.set_printoptions(threshold=numpy.nan)

class EL(nn.Module):
    def __init__(self, config):
        super(EL, self).__init__()

        self.encoder = ELEncoder(config={
            'type': config['type'],
            'lstm_hiddim': config['lstm_hiddim'],
            'n_filters': config['n_filters'],
            'filter_sizes': config['filter_sizes'],
            'word_embs': config['word_embs'],
            'pos_embdim': config['pos_embdim'],
            'type_embdim': config['type_embdim'],
            'ent_embdim': config['ent_embdim'],
            'dropout': config['dropout'],
            'en_dim': config['en_dim'],
            'n_types': config['n_types'],
            'n_rels': config['n_rels']})

    def forward(self, input):
        scores, noise_scores = self.encoder(input)
        return scores, noise_scores

    def compute_loss(self, input):
        scores, noise_scores, margin, kl_coef, noise_prior = input['scores'], input['noise_scores'], input['margin'], input['kl_coef'], input['noise_prior']
        noise_scores = noise_scores * 3

        # compute scores as if no noise
        best_pos_scores = scores[:, :input['N_POSS']].max(dim=1)[0]
        best_neg_scores = scores[:, input['N_POSS']:].max(dim=1)[0]
        diff = best_neg_scores + margin - best_pos_scores
        if kl_coef > 0:
            loss = ((1 - nn.functional.sigmoid(noise_scores)).squeeze() * torch.where(diff > 0, diff, torch.zeros(diff.shape).cuda())).mean()
        elif kl_coef == 0:
            loss = torch.where(diff > 0, diff, torch.zeros(diff.shape).cuda()).mean()


        # compute kl
        p_noise = nn.functional.sigmoid(noise_scores).mean()
        p_noise_prior = torch.Tensor([noise_prior]).mean().cuda()
        kl = p_noise * (torch.log(p_noise + 1e-10) - torch.log(p_noise_prior + 1e-10)) + \
                (1 - p_noise) * (torch.log(1 - p_noise + 1e-10) - torch.log(1 - p_noise_prior + 1e-10))
        loss += kl_coef * kl

        return loss, kl



