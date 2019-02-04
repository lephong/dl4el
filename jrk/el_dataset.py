import jrk.el_hyperparams as hp
from jrk.vocabulary import Vocabulary
import jrk.utils as utils
import torch
from torch.autograd import Variable
import json
import random
from pprint import pprint
from copy import deepcopy

class ELDataset:
    def __init__(self, data_path, vocas, triples, max_len=100):
        self.vocas = vocas
        self.triples = triples
        self.entIdList = list(self.triples['ent2typeId'].keys())
        print(len(self.triples['ent2typeId']))

        if 'self' not in self.triples['relId']:
            self.triples['relId']['self'] = len(self.triples['relId'])

        print('load train set')
        self.train = self.read_from_file(data_path['train'], max_len=max_len)
        print('load dev set')
        self.dev = self.read_from_file(data_path['dev'], train=False, max_len=max_len)
        print('load test set')
        self.test = self.read_from_file(data_path['test'], train=False, max_len=max_len)

    def read_from_file(self, path, format='json', train=True, max_len=100):
        pass

    def get_minibatch(self, data, start, end):
        if data == self.train:
            MAX_N_POSS = hp.MAX_N_POSS_TRAIN
        else:
            MAX_N_POSS = hp.MAX_N_POSS_TEST

        org = data[start:end]

        # sort by number of words
        org.sort(key=lambda x: len(x[0]), reverse=True)

        input = {
                'tokens': [], 'masks': [], 'm_loc': [], 'pos_wrt_m': [],
                'nb_types': [], 'nb_type_ids': [], 'nb_n_types': [],
                'nb_rs': [], 'cand_n_nb': [], 'cand_nb_ids': [],
                'real_n_poss': [],
                }
        sentence = []
        candidates = []
        input['real_n_poss'] = [min(len(x[4]), MAX_N_POSS) for x in org]
        input['N_POSS'] = max(input['real_n_poss'])
        targets = []
        ners = []


        for item in org:
            tokens, m_loc, pos_wrt_m, sent, positives, ent, ner = deepcopy(item)

            # sampling negative samples
            negatives = random.sample(self.entIdList, hp.N_NEGS)
            neg_types = [self.triples['ent2typeId'][c] for c in negatives]

            pos_types = [self.triples['ent2typeId'][c] for c in positives]
            if len(positives) > input['N_POSS']:
                positives = positives[:input['N_POSS']]
                pos_types = pos_types[:input['N_POSS']]
            else:
                positives += [positives[-1]] * (input['N_POSS'] - len(positives))
                pos_types += [pos_types[-1].copy()] * (input['N_POSS'] - len(pos_types))
            cand_types = pos_types + neg_types

            input['tokens'].append(tokens)
            input['m_loc'].append(m_loc)
            input['pos_wrt_m'].append(pos_wrt_m)
            input['nb_types'].extend([[types] for types in cand_types])
            input['nb_rs'].extend([[self.triples['relId']['self']] for t in cand_types])
            candidates.append(positives + negatives)
            sentence.append(sent)
            targets.append(ent)
            ners.append(ner)

        # get neighbour
        flat_candidates = [c for cands in candidates for c in cands]
        for c, nb_types, nb_rs in zip(flat_candidates, input['nb_types'], input['nb_rs']):
            if c in self.triples['h2rtId'] and len(self.triples['h2rtId'][c]) < 30:
                tmp = [(rt >> 32, rt - ((rt >> 32) << 32)) for rt in self.triples['h2rtId'][c]]
                nb_types += [self.triples['ent2typeId'][t] for _, t in tmp if t in self.triples['ent2typeId']]
                nb_rs += [r for r, t in tmp if t in self.triples['ent2typeId']]
            input['cand_n_nb'].append(len(nb_types))
            input['cand_nb_ids'].append([len(input['cand_nb_ids'])] * len(nb_rs))

        input['nb_types'] = [types for nb_t in input['nb_types'] for types in nb_t]
        input['nb_n_types'] = [len(types) for types in input['nb_types']]

        if hp.TYPE_OPT == 'mean':
            input['nb_type_ids'] = [[i] * len(types) for i, types in enumerate(input['nb_types'])]
        elif hp.TYPE_OPT == 'max':
            input['nb_max_n_types'] = max(input['nb_n_types'])
            input['nb_type_ids'] = [list(range(i * input['nb_max_n_types'], i * input['nb_max_n_types'] + len(types)))
                    for i, types in enumerate(input['nb_types'])]

        # flatten
        input['nb_types'] = [t for types in input['nb_types'] for t in types]
        input['nb_type_ids'] = [_i for _ids in input['nb_type_ids'] for _i in _ids]
        input['nb_rs'] = [r for rs in input['nb_rs'] for r in rs]
        input['cand_nb_ids'] = [_i for _ids in input['cand_nb_ids'] for _i in _ids]

        # convert to Tensor
        input['tokens'], input['masks'] = utils.make_equal_len(input['tokens'], fill_in=self.vocas['word'].pad_id)
        input['tokens'] = Variable(torch.LongTensor(input['tokens']).cuda(), requires_grad=False)
        input['masks'] = Variable(torch.Tensor(input['masks']).cuda(), requires_grad=False)

        input['m_loc'] = Variable(torch.LongTensor(input['m_loc']).cuda(), requires_grad=False)

        input['pos_wrt_m'], _ = utils.make_equal_len(input['pos_wrt_m'], fill_in=hp.MAX_POS)
        input['pos_wrt_m'] = Variable(torch.LongTensor(input['pos_wrt_m']).cuda(), requires_grad=False)

        input['nb_types'] = Variable(torch.LongTensor(input['nb_types']).cuda(), requires_grad=False)
        input['nb_n_types'] = Variable(torch.LongTensor(input['nb_n_types']).cuda(), requires_grad=False)
        input['nb_type_ids'] = Variable(torch.LongTensor(input['nb_type_ids']).cuda(), requires_grad=False)

        input['cand_n_nb'] = Variable(torch.LongTensor(input['cand_n_nb']).cuda(), requires_grad=False)
        input['cand_nb_ids'] = Variable(torch.LongTensor(input['cand_nb_ids']).cuda(), requires_grad=False)
        input['nb_rs'] = Variable(torch.LongTensor(input['nb_rs']).cuda(), requires_grad=False)

        input['real_n_poss'] = Variable(torch.LongTensor(input['real_n_poss']).cuda(), requires_grad=False)

        return input, sentence, candidates, targets, ners


class NYT_RCV1(ELDataset):
    def __init__(self, data_path, vocas, triples, max_len=100):
        data_path = {
                'train': data_path,
                'dev': 'data/EL/AIDA/testa.json',
                'test': 'data/EL/AIDA/testb.json'
                }
        super(NYT_RCV1, self).__init__(data_path, vocas, triples, max_len=max_len)

    def read_json(self, path):
        data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i % int(1e3) == 0:
                    print(i, end='\r')
                line = line.strip()
                data.append(json.loads(line))

        return data

    def read_data(self, path, format='json', train=True, max_len=100):
        if train:
            MAX_N_POSS = hp.MAX_N_POSS_TRAIN
        else:
            MAX_N_POSS = hp.MAX_N_POSS_TEST

        data = []
        print('read file from', path)
        if format == 'json':
            raw_data = self.read_json(path)
        else:
            assert(False)

        if train:
            random.shuffle(raw_data)

        print('load data')
        for count, item in enumerate(raw_data):
            if format == 'json':
                org_words, ments = item['sentence'], item['mentions']
                sent = ' '.join(org_words)
                words = [self.vocas['word'].get_id(w) for w in org_words]
                if len(words) > max_len:
                    continue
                for ment in ments:
                    ms, me = ment['mention']
                    if len(words) == 0:
                        print(sent)
                        continue

                    pos_wrt_m = [max(i - ms, -hp.MAX_POS) for i in range(0, ms)] + \
                            [0] * (me - ms) + \
                            [min(i - me + 1, hp.MAX_POS) for i in range(me, len(words))]

                    positives = [c for c in ment['positives'] if c in self.triples['ent2typeId']]
                    if len(positives) == 0:
                        continue
                    if len(positives) > MAX_N_POSS:
                        # sort wrt length (smaller is better)
                        tmp = [(c, len(self.triples['ent2nameId'][c])) for c in positives]
                        sorted(tmp, key=lambda x:x[1])
                        positives = [x[0] for x in tmp]

                    data.append((words, (ms, me), pos_wrt_m, sent, positives, ment.get('entity', None), ment.get('ner', 'O')))

                if (count + 1) % 1000 == 0:
                    print(count // 1000, 'k', end='\r')


        print('load', len(data), 'items')
        return data

    def read_from_file(self, path, format='json', train=True, max_len=100):
        return self.read_data(path, format=format, train=train, max_len=max_len)

