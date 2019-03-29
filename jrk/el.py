import torch
import torch.nn as nn
from jrk.el_encoder import ELEncoder
import numpy
import json
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

        self.config = {
            'type': config['type'],
            'lstm_hiddim': config['lstm_hiddim'],
            'n_filters': config['n_filters'],
            'filter_sizes': config['filter_sizes'],
            'pos_embdim': config['pos_embdim'],
            'type_embdim': config['type_embdim'],
            'ent_embdim': config['ent_embdim'],
            'dropout': config['dropout'],
            'en_dim': config['en_dim'],
            'n_types': config['n_types'],
            'n_rels': config['n_rels'],
            'kl_coef': config['kl_coef'],
            'noise_prior': config['noise_prior'],
            'margin': config['margin']}


    def save(self, path, suffix='', save_config=True):
        torch.save(self.state_dict(), path + '.state_dict' + suffix)
        if save_config:
            with open(path + '.config', 'w', encoding='utf8') as f:
                json.dump(self.config, f)

    def forward(self, input):
        scores, noise_scores = self.encoder(input)
        return scores, noise_scores

    def compute_logprobs(self, scores, noise_scores):
        noise_logprobs = torch.nn.functional.logsigmoid(noise_scores)
        notnoise_ent_logprobs = torch.nn.functional.log_softmax(scores, dim=1) + torch.log(1 - torch.exp(noise_logprobs) + 1e-10)
        return notnoise_ent_logprobs, noise_logprobs

    def compute_loss(self, input, sup=False):
        if not sup:
            scores, noise_scores, = input['scores'], input['noise_scores']
            noise_scores = noise_scores * 3

            # compute scores as if no noise
            best_pos_scores = scores[:, :input['N_POSS']].max(dim=1)[0]
            best_neg_scores = scores[:, input['N_POSS']:].max(dim=1)[0]
            diff = best_neg_scores + self.config['margin'] - best_pos_scores
            if self.config['kl_coef'] > 0:
                loss = ((1 - nn.functional.sigmoid(noise_scores)).squeeze() * torch.where(diff > 0, diff, torch.zeros(diff.shape).cuda())).mean()
            elif self.config['kl_coef'] == 0:
                loss = torch.where(diff > 0, diff, torch.zeros(diff.shape).cuda()).mean()


            # compute kl
            p_noise = nn.functional.sigmoid(noise_scores).mean()
            p_noise_prior = torch.Tensor([self.config['noise_prior']]).mean().cuda()
            kl = p_noise * (torch.log(p_noise + 1e-10) - torch.log(p_noise_prior + 1e-10)) + \
                    (1 - p_noise) * (torch.log(1 - p_noise + 1e-10) - torch.log(1 - p_noise_prior + 1e-10))
            loss += self.config['kl_coef'] * kl

            return loss, kl

        else:
            # because this is supervised learning, we care only positive candidates
            scores, noise_scores = input['scores'][:, :input['N_POSS']], input['noise_scores']
            targets = input['targets']  # if this is noise, target == input['N_POSS']

            noise_logprobs = torch.nn.functional.logsigmoid(noise_scores)
            notnoise_ent_logprobs = torch.nn.functional.log_softmax(scores, dim=1) + torch.log(1 - torch.exp(noise_logprobs) + 1e-10)
            logprobs = torch.cat([notnoise_ent_logprobs, noise_logprobs], dim=1)
            loss = torch.nn.functional.nll_loss(logprobs, targets)
        return loss, 0

