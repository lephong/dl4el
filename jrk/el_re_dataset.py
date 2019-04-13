import jrk.el_hyperparams as hp
import jrk.utils as utils
import torch
from torch.autograd import Variable
import json
import random

class ELREDataset:
    def __init__(self, data_path, vocas, triples, max_len=2000):
        self.vocas = vocas
        self.triples = triples
        self.entIdList = list(self.triples['ent2typeId'].keys())
        print(len(self.triples['ent2typeId']))

        if 'self' not in self.triples['relId']:
            self.triples['relId']['self'] = len(self.triples['relId'])

        print('load dataset')
        self.data = self.read_from_file(data_path['data'], train=False, max_len=max_len)

    def get_minibatch(self, data, start, end):
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
        candidates = []
        items = []
        names = []
        input['real_n_poss'] = [min(len(x[3]), MAX_N_POSS) for x in org]
        input['N_POSS'] = max(input['real_n_poss'])
        input['N_NEGS'] = 0

        for item in org:
            tokens, m_loc, pos_wrt_m, positives, item, name = item
            pos_types = [self.triples['ent2typeId'][c] for c in positives]
            if len(positives) > input['N_POSS']:
                positives = positives[:input['N_POSS']]
                pos_types = pos_types[:input['N_POSS']]
            else:
                positives += [positives[-1]] * (input['N_POSS'] - len(positives))
                pos_types += [pos_types[-1].copy()] * (input['N_POSS'] - len(pos_types))
            cand_types = pos_types

            input['tokens'].append(tokens)
            input['m_loc'].append(m_loc)
            input['pos_wrt_m'].append(pos_wrt_m)
            input['nb_types'].extend([[types] for types in cand_types])
            input['nb_rs'].extend([[self.triples['relId']['self']] for t in cand_types])
            candidates.append(positives)
            items.append(item)
            names.append(name)

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
        return input, candidates, items, names

    def read_json(self, path):
        data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i % int(1e3) == 0:
                    print('load %d items from json' % i, end='\r')
                #if i > 1e5:
                #    break
                line = line.strip()
                data.append(json.loads(line))

        print()
        return data

    def read_data(self, path, format='json', train=True, max_len=200):
        data = []
        self.left_data = []
        self.replace_ent = False

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
                org_words = item['token']
                words = [self.vocas['word'].get_id(w) for w in org_words]
                if len(words) > max_len:
                    continue

                ok = True
                subj_added = False
                for name in ['subj_', 'obj_']:
                    item[name+'positives'] = [self.vocas['ent'].get_id(c) for c in item[name+'positives']]
                    positives = [c for c in item[name+'positives'] if c in self.triples['ent2typeId']]

                    if len(positives) == 0:
                        ok = False
                        break

                    ms, me = item[name+'start'], item[name+'end'] + 1
                    if len(words) == 0:
                        print(item, name)
                        ok = False
                        break

                    pos_wrt_m = [max(i - ms, -hp.MAX_POS) for i in range(0, ms)] + \
                            [0] * (me - ms) + \
                            [min(i - me + 1, hp.MAX_POS) for i in range(me, len(words))]

                    try:
                        item[name+'ent'] = self.vocas['ent'].word2id[item[name+'ent']]
                    except:
                        if item[name+'ent'].endswith('dump'):
                            item[name+'ent'] = item[name+'positives'][0]
                            self.replace_ent = True
                        else:
                            print('%s not in ent dict' % item[name+'ent'])
                            ok = False
                            break

                    if item[name+'ent'] not in self.triples['ent2typeId']:
                        ok = False
                        break

                    try:
                        positives.remove(item[name+'ent'])
                    except:
                        pass
                    positives = [item[name+'ent']] + positives

                    item[name] = True
                    data.append((words, (ms, me), pos_wrt_m, positives, item, name))
                    subj_added = (name == 'subj_')

                if not ok:
                    if subj_added:
                        del data[-1]
                    self.left_data.append({
                        'id': item.get('id', str(random.randint(0, 10000000))),
                        'is_prp': item.get('is_prp', False),
                        'token': item['token'],
                        'subj_start': item['subj_start'],
                        'subj_end': item['subj_end'],
                        'obj_start': item['obj_start'],
                        'obj_end': item['obj_end'],
                        'relation': item['relation']})

                if (count + 1) % 1000 == 0:
                    print(count // 1000, 'k', end='\r')
                #if (count + 1) > 50000:
                #    break

        # select only those that both subj and obj are valid
        new_data = []
        for x in data:
            if 'subj_' in x[4] and 'obj_' in x[4]:
                new_data.append(x)
        data = new_data

        for x in data:
            x[4]['subj_'] = x[4]['obj_'] = None

        print('load', len(data), 'items')
        return data

    def read_from_file(self, path, format='json', train=True, max_len=200):
        return self.read_data(path, format=format, train=train, max_len=max_len)

