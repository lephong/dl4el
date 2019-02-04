import torch
import torch.nn as nn
from jrk.el_hyperparams import MAX_POS, N_NEGS, TYPE_OPT
from jrk.utils import embedding_3D

class ELEncoder(nn.Module):
    def __init__(self, config):
        super(ELEncoder, self).__init__()
        word_embs = nn.functional.normalize(config['word_embs'])
        self.word_embs = nn.Embedding.from_pretrained(word_embs, freeze=True)
        self.pos_embs = nn.Embedding(2 * MAX_POS + 1, config['pos_embdim'])
        self.type_embs = nn.Embedding(config['n_types'], config['type_embdim'])
        self.type_embdim = config['type_embdim']
        self.ent_embdim = config['ent_embdim']

        self.rel_embs = nn.Embedding(config['n_rels'], config['ent_embdim'])
        self.rel_weight = nn.Parameter(torch.zeros(config['type_embdim'], config['ent_embdim']))
        nn.init.kaiming_normal_(self.rel_weight, mode='fan_out', nonlinearity='relu')

        dim = word_embs.shape[1] + config['pos_embdim']

        self.dropout = nn.Dropout(p=config['dropout'])
        self.type = config.get('type', 'lstm')

        if self.type == 'lstm':
            self.lstm = nn.LSTM(dim, config['lstm_hiddim'], 2, batch_first=True, bidirectional=True, dropout=config['dropout'])
            en_hiddim1 = 4 * config['lstm_hiddim'] + config['ent_embdim']
        elif self.type == 'pcnn':
            self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=config['n_filters'],
                kernel_size=(fs, dim), padding=(fs//2, 0)) for fs in config['filter_sizes']])
            en_hiddim1 = len(config['filter_sizes']) * config['n_filters'] * 2 + config['ent_embdim']
        else:
            assert(False)

        self.noise_scorer = nn.Sequential(
                nn.Dropout(p=config['dropout']),
                nn.Linear(en_hiddim1, config['en_dim']),
                nn.ReLU(),
                nn.Dropout(p=config['dropout']),
                nn.Linear(config['en_dim'], 1))

        self.scorer = nn.Sequential(
                nn.Dropout(p=config['dropout']),
                nn.Linear(en_hiddim1, config['en_dim']),
                nn.ReLU(),
                nn.Dropout(p=config['dropout']),
                nn.Linear(config['en_dim'], 1))

    def forward(self, input):
        N_POSS = input['N_POSS']
        N_CANDS = N_POSS + N_NEGS

        batchsize = input['tokens'].shape[0]
        # pos embs
        pos_ment_embs = self.pos_embs(input['pos_wrt_m'] + MAX_POS)

        # entity embs
        if TYPE_OPT == 'mean':
            nb_embs = torch.zeros(input['nb_n_types'].shape[0], self.type_embdim).cuda().\
                    scatter_add_(0,
                            input['nb_type_ids'].unsqueeze(1).repeat(1, self.type_embdim),
                            self.type_embs(input['nb_types']))
            nb_embs = nb_embs / input['nb_n_types'].unsqueeze(1).float()
        elif TYPE_OPT == 'max':
            nb_embs = torch.empty(input['nb_n_types'].shape[0] * input['nb_max_n_types'], self.type_embdim).cuda().fill_(-1e10).\
                    scatter_(0,
                            input['nb_type_ids'].unsqueeze(1).repeat(1, self.type_embdim),
                            self.type_embs(input['nb_types']))
            nb_embs = torch.max(nb_embs.view(input['nb_n_types'].shape[0], input['nb_max_n_types'], -1), dim=1)[0]


        rel_embs = self.rel_embs(input['nb_rs'])
        cand_embs = nn.functional.relu(
                torch.zeros(input['cand_n_nb'].shape[0], self.ent_embdim).cuda().\
                        scatter_add_(0,
                            input['cand_nb_ids'].unsqueeze(1).repeat(1, self.ent_embdim),
                            torch.matmul(nb_embs, self.rel_weight) + rel_embs))
        cand_embs = cand_embs.view(batchsize * N_CANDS, -1)

        inp = self.word_embs(input['tokens'])    # batchsize x n x dim
        inp = torch.cat([inp, pos_ment_embs], dim=2)
        inp = self.dropout(inp)

        if self.type == 'lstm':
            # apply bilstm
            lens = input['masks'].long().sum(dim=1)
            assert(lens[0] == inp.shape[1])

            inp = nn.utils.rnn.pack_padded_sequence(inp, lens, batch_first=True)
            out, (ht, ct) = self.lstm(inp)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=lens[0])
            #ht = ht.view(2, 2, batchsize, -1)
            #ret = torch.cat([ht[1,0,:,:], ht[1,1,:,:]], dim=1)

            out = out.view(batchsize, lens[0], -1).contiguous()
            ctx_vecs = embedding_3D(out, input['m_loc']).view(batchsize, -1)

        elif self.type == 'pcnn':
            inp = inp.unsqueeze(1)
            conved = [nn.functional.relu(conv(inp)).squeeze(3) for conv in self.convs]  # conved[i]: batchsize x n_filters x len

            # filtering out two parts
            mask = input['pos_wrt_m'].le(0).float().unsqueeze(dim=1)
            left = [c * mask - (1 - mask) * 1e10 for c in conved]
            mask = (input['pos_wrt_m'].ge(0).float() * input['masks']).unsqueeze(dim=1)
            right = [c * mask - (1 - mask) * 1e10 for c in conved]

            # max pooling
            pooled_l = torch.cat([nn.functional.max_pool1d(x, x.shape[2]).squeeze(2) for x in left], dim=1)
            pooled_r = torch.cat([nn.functional.max_pool1d(x, x.shape[2]).squeeze(2) for x in right], dim=1)
            ctx_vecs = torch.cat([pooled_l, pooled_r], dim=1)

        else:
            assert(False)

        rp_ctx_vecs = ctx_vecs.unsqueeze(dim=1).repeat(1, N_CANDS, 1)
        reprs = torch.cat([rp_ctx_vecs.view(batchsize * N_CANDS, -1), cand_embs], dim=1)
        scores = self.scorer(reprs).view(batchsize, N_CANDS)

        # masking
        pos_mask = torch.linspace(1, N_POSS, steps=N_POSS).repeat(batchsize, 1).cuda() <= input['real_n_poss'].float().unsqueeze(1).repeat(1, N_POSS)
        mask = torch.cat([pos_mask, torch.ones(batchsize, N_NEGS).cuda().byte()], dim=1)
        scores = torch.where(mask, scores, torch.empty(scores.shape).cuda().fill_(-1e10))

        # compute noise prob
        p = torch.nn.functional.softmax(scores[:, :N_POSS])
        e = (cand_embs.view(batchsize, N_CANDS, -1)[:, :N_POSS, :].contiguous() * p.unsqueeze(dim=2)).sum(dim=1)  # like attention
        reprs = torch.cat([ctx_vecs, e], dim=1)
        noise_scores = self.noise_scorer(reprs)
        return scores, noise_scores
