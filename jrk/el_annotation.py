import json
from jrk.vocabulary import Vocabulary
import os.path
import pickle
import itertools
import random
import copy

mode = 'test'  # 'train'

# mode == 'train'
MAX_N_ALL_CANDS = 1000
MAX_N_POSS = 100
MAX_N_NEGS = 10

ent_path = "data/freebase/freebase-entity.txt"
triples_path = 'data/freebase/freebase-triples.txt'
text_path = "../freebase2tacred/data/ner.nyt_rcv1.txt"
#ner_path = "../freebase2tacred/data/ner.nyt_rcv1.txt"
out_path = "data/freebase/el_annotation/el_annotated.json"

if mode == 'test':
    MAX_N_ALL_CANDS = 100

    sup_train = False
    text_path = "data/EL/AIDA/train.txt"
    #ner_path = "data/EL/AIDA/testb.gold_ner.txt"
    out_path = "data/EL/AIDA/train.json"

    #sup_train = True
    #text_path = "data/EL/AIDA/train.txt"
    #ner_path = "data/EL/AIDA/train.txt"
    #out_path = "data/EL/AIDA/train.json"

print('load ent and word dics')
voca_ent_path = 'data/freebase/freebase-entity.lst'
voca_ent, _ = Vocabulary.load(voca_ent_path, normalization=False, add_pad_unk=False)
voca_word_path = 'data/freebase/freebase-word.lst'
voca_word, _ = Vocabulary.load(voca_word_path, normalization=True, lower=True, add_pad_unk=False, digit_0=True)

word2entId = None
ent2wordId = None
triplesId = None

correct = 0
total = 0


def load_word2ent():
    global ent2wordId, word2entId

    if os.path.exists(ent_path + '.pkl'):
        print('load pickle file')
        with open(ent_path + '.pkl', 'rb') as f:
            ent2wordId = pickle.load(f)
            word2entId = pickle.load(f)

    else:
        ent2wordId = {}
        word2entId = {}
        with open(ent_path, 'r') as f:
            for i, line in enumerate(f):
                if (i + 1) % int(1e6) == 0:
                    print(i + 1, len(word2entId), end='\r')
                comps = line.strip().split('\t')
                if len(comps) != 2:
                    print(line)
                    continue
                ent, words = comps
                entId = voca_ent.get_id(ent)
                wordIds = [voca_word.get_id(w) for w in words.split(' ')]
                for _wId in wordIds:
                    ids = word2entId.get(_wId, set())
                    if len(ids) == 0:
                        word2entId[_wId] = ids
                    ids.add(entId)
                ent2wordId[entId] = wordIds


        print('save pickle\t\t\t\t')
        with open(ent_path + '.pkl', 'wb') as f:
            pickle.dump(ent2wordId, f, -1)
            pickle.dump(word2entId, f, -1)


def load_triples():
    global triplesId

    if mode == 'test':
        triplesId = set()
        return

    if os.path.exists(triples_path + '.pkl'):
        print('load pickle file')
        with open(triples_path + '.pkl', 'rb') as f:
            triplesId = pickle.load(f)

    else:
        triplesId = set()
        with open(triples_path, 'r') as f:
            for i, line in enumerate(f):
                if (i + 1) % int(1e6) == 0:
                    print(i + 1, len(triplesId), end='\r')

                comps = line.strip().split('\t')
                if len(comps) == 3 and not comps[2].endswith("instance"):
                    h, t = voca_ent.get_id(comps[0]), voca_ent.get_id(comps[2])
                    epair = (h << 32) |  t  # it's equivalent to (h, t)
                    triplesId.add(epair)

        print('save pickle\t\t\t\t')
        with open(triples_path + '.pkl', 'wb') as f:
            pickle.dump(triplesId, f, -1)


def process_sent(sent, f, sent_ner=None):
    if sent_ner is None:
        sent_ner = copy.deepcopy(sent)

    words = sent.split(' ')
    words_ner = sent_ner.split(' ')
    assert(len(words) == len(words_ner))

    sent = [w if '|||' not in w else w.split('|||')[0] for w in words]
    mentions = []
    entities = []
    ners = []
    cur_men = None
    cur_ent = None
    cur_ner = None

    for _wi, w in enumerate(words):
        if '|||' in w:
            if mode == 'train':
                w, t = w.split('|||')
                cur_ent = None
            elif mode == 'test':
                w, t, cur_ent = w.split('|||')
                try:
                    _,  _, cur_ner = words_ner[_wi].split('|||')
                except:
                    cur_ner = 'O'

                cur_ent = voca_ent.word2id.get(cur_ent, None)
        else:
            t = ''

        if t == '':
            if cur_men is not None:
                cur_men[1] = _wi
                mentions.append(cur_men)
                entities.append(cur_ent)
                ners.append(cur_ner)
                cur_men, cur_ent = None, None
        elif t == 'B':
            if cur_men is not None:
                cur_men[1] = _wi
                mentions.append(cur_men)
                entities.append(cur_ent)
                ners.append(cur_ner)
            cur_men = [_wi, None]
        elif t == 'I':
            assert(cur_men is not None)

    if cur_men is not None:
        cur_men[1] = len(words)
        mentions.append(cur_men)
        entities.append(cur_ent)
        ners.append(cur_ner)

    ret = {'sentence': sent, 'mentions': []}

    if mode == 'test':
        if len(mentions) < 1:
            return None

        global correct, total

        for m_loc, ent, ner in zip(mentions, entities, ners):
            if ent is None:
                continue
            item = {'mention': m_loc, 'positives': [], 'negatives': [], 'entity': ent, 'ner': ner}

            if not sup_train:
                item['positives'] = run(sent[m_loc[0]:m_loc[1]], ent)
            else:
                item['negatives'] = run(sent[m_loc[0]:m_loc[1]], ent)
                try:
                    item['negatives'].remove(ent)
                except:
                    pass
                item['positives'] = [ent]
            ret['mentions'].append(item)

            if ent in item['positives']:
                correct += 1
            total += 1

        print(correct, total, end='\r')

        if len(ret['mentions']) < 1:
            return None

    if mode == 'train':
        if len(mentions) < 2:
            return None

        for m_loc in mentions:
            item = {'mention': m_loc, 'all_candidates': [], 'positives': set(), 'negatives': None}

            item['all_candidates'] = run(sent[m_loc[0]:m_loc[1]])
            ret['mentions'].append(item)

        ret_is_none = True
        for m1, m2 in itertools.product(ret['mentions'], repeat=2):
            if m1 == m2 or m1['positives'] is None or m2['positives'] is None:
                continue
            c1c2List = itertools.product(m1['all_candidates'], m2['all_candidates'])
            for c1, c2 in c1c2List:
                val = (c1 << 32) | c2
                if val in triplesId:
                    m1['positives'].add(c1)
                    m2['positives'].add(c2)
                    ret_is_none = False

            if len(m1['positives']) > MAX_N_POSS:
                m1['positives'] = None
            if len(m2['positives']) > MAX_N_POSS:
                m2['positives'] = None

        if ret_is_none:
            return

        for m in ret['mentions']:
            if m['positives'] is not None and len(m['positives']) > 0:
                m['negatives'] = list(set(m['all_candidates']).difference(m['positives']))
                if len(m['negatives']) > 0:
                    random.shuffle(m['negatives'])
                    m['negatives'] = m['negatives'][:min(len(m['negatives']), MAX_N_NEGS)]
                m['positives'] = list(m['positives'])
            else:
                m['positives'] = []
                m['negatives'] = []

            del m['all_candidates']

    f.write(json.dumps(ret) + '\n')
    return ret


def run(mwords, ent=None):
    mwIds = [voca_word.get_id(w) for w in mwords]
    mwIds = [_id for _id in mwIds if _id != voca_word.unk_id]
    entsId = None
    n_mw = len(mwords)

    for _wId in mwIds:
        if entsId is None:
            entsId = word2entId[_wId]
        else:
            entsId = entsId.intersection(word2entId[_wId])
        if len(entsId) == 0:
            break

    if entsId is None:
        return []

    else:
        entsId = list(entsId)
        if mode == 'test':
            entsId = sorted(entsId, key=lambda e: len(ent2wordId[e]))
        ret_entsId = [_eId for _eId in entsId if len(ent2wordId[_eId]) < n_mw + 4]
        ret_entsId = ret_entsId[:min(len(ret_entsId), MAX_N_ALL_CANDS)]
        return ret_entsId


if __name__ == '__main__':
    print('load word2ent from', ent_path)
    load_word2ent()

    print('load triples from', triples_path)
    load_triples()

    print('process sentences from', text_path)
    with open(text_path, 'r') as fin:
        #with open(ner_path, 'r') as fner:
        with open(out_path, 'w') as fout:
            count_saved = 0
            for i, line  in enumerate(fin):
                if i == 0 or '|||' not in line:
                    continue
                if i % 10 == 0:
                    print(i, count_saved, end='\r')
                if count_saved > 1e8:
                    break
                sent = line.strip()
                item = process_sent(sent, fout)
                if item is not None:
                    count_saved += 1

    print(correct, total)
