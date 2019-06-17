import torch
import torch.optim as optim
from jrk.el_dataset import NYT_RCV1
from jrk.vocabulary import Vocabulary
import jrk.utils as utils
from jrk.el import EL
import random
import pickle
import os.path
import numpy as np
import json
import jrk.el_hyperparams as hp

args = hp.parser.parse_args()
datadir = args.datadir
print(args)

data_path = hp.data_path

# load word embeddins and vocabular
print('load words and entities')
voca_word, word_embs = utils.load_voca_embs(datadir + '../glove/dict.word', datadir + '../glove/word_embeddings.npy')
word_embs = torch.Tensor(word_embs)
voca_type, _ = Vocabulary.load(datadir + '/freebase-type.txt', normalization=False, add_pad_unk=True)
voca_ent, _ = Vocabulary.load(datadir + '/freebase-entity.lst', normalization=False, add_pad_unk=False)
voca_ent_word, _ = Vocabulary.load(datadir + '/freebase-word.lst', normalization=True, add_pad_unk=False, lower=True, digit_0=True)
n_types = voca_type.size()


# load ent2nameId
print('load ent_names')
with open(datadir + '/freebase-entity.txt.pkl', 'rb') as f:
    ent2nameId = pickle.load(f)


# load ent2typeId
print('load ent2typeId')
this_path = datadir + '/freebase-type-instance.txt'
if os.path.exists(this_path + '.pkl'):
    print('load pickle')
    with open(this_path + '.pkl', 'rb') as f:
        ent2typeId = pickle.load(f)
else:
    ent2typeId = {}
    with open(datadir + '/freebase-type-instance.txt', 'r') as f:
        for i, line in enumerate(f):
            if i % int(1e6) == 0:
                print(i, len(ent2typeId), end='\r')
            comps = line.strip().split('\t')
            if len(comps) == 2:
                ent, types = comps
                types = types.split(' ')
                entId = voca_ent.word2id.get(ent, None)
                if entId is not None:
                    ent2typeId[entId] = [voca_type.get_id(t) for t in types]
            else:
                print(i, line)

    print('save to pickle')
    with open(this_path + '.pkl', 'wb') as f:
        pickle.dump(ent2typeId, f, -1)


# load triples
print('load triples')
triples_path = datadir + '/freebase-triples.txt'
relId = {}
h2rtId = {}
#if os.path.exists(triples_path + '.el.pkl'):
#    print('load pickle file')
#    with open(triples_path + '.el.pkl', 'rb') as f:
#        relId = pickle.load(f)
#        h2rtId = pickle.load(f)

#else:
#    with open(triples_path, 'r') as f:
#        for i, line in enumerate(f):
#            if (i + 1) % int(1e6) == 0:
#                print(i + 1, len(h2rtId), end='\r')

#            comps = line.strip().split('\t')
#            if len(comps) == 3 and not comps[2].endswith("instance"):
#                h, r, t = voca_ent.get_id(comps[0]), relId.get(comps[1], None), voca_ent.get_id(comps[2])
#                if r is None:
#                    r = len(relId)
#                    relId[comps[1]] = r
#                rt = (r << 32) |  t  # it's equivalent to (r, t)
#                rts = h2rtId.get(h, None)
#                if rts is None:
#                    rts = []
#                    h2rtId[h] = rts
#                rts.append(rt)

#    print('save pickle\t\t\t\t')
#    with open(triples_path + '.el.pkl', 'wb') as f:
#        pickle.dump(relId, f, -1)
#        pickle.dump(h2rtId, f, -1)


# load dataset
print('load dataset')
dataset = NYT_RCV1(data_path,
        {
            'word': voca_word,
            'type': voca_type,
            'ent': voca_ent
        },
        {
            'ent2typeId': ent2typeId,
            'ent2nameId': ent2nameId,
            'relId': relId,
            'h2rtId':  h2rtId,
        },
        max_len=args.max_len)


# create model
if args.mode == 'train':
    print('create model')
    model = EL(config={
        'type': args.enc_type,
        'lstm_hiddim': args.lstm_hiddim,
        'n_filters': args.n_filters,
        'filter_sizes': (3, 5, 7),  # each number has to be odd
        'word_embs': word_embs,
        'pos_embdim': args.pos_embdim,
        'type_embdim': args.type_embdim,
        'ent_embdim': args.ent_embdim,
        'dropout': args.dropout,
        'en_dim': args.en_dim,
        'n_types': n_types,
        'n_rels': len(relId),
        'kl_coef': args.kl_coef,
        'noise_prior': args.noise_prior,
        'margin': args.margin,
    })
elif args.mode == 'eval':
    print('load model')
    with open(args.model_path + '.config', 'r') as f:
        config = json.load(f)
    print(config)

    model = EL(config={
        'type': config['type'],
        'lstm_hiddim': config['lstm_hiddim'],
        'n_filters': config['n_filters'],
        'filter_sizes': (3, 5, 7),  # each number has to be odd
        'word_embs': word_embs,
        'pos_embdim': config['pos_embdim'],
        'type_embdim': config['type_embdim'],
        'ent_embdim': config['ent_embdim'],
        'dropout': config['dropout'],
        'en_dim': config['en_dim'],
        'n_types': config['n_types'],
        'n_rels': config['n_rels'],
        'kl_coef': config['kl_coef'],
        'noise_prior': config['noise_prior'],
        'margin': config['margin'],
    })
    model.load_state_dict(torch.load(args.model_path + '.state_dict'))

model.cuda()


# for testing
def test(data=None, noise_threshold=args.noise_threshold):
    if data is None:
        data = dataset.dev
    n_correct_pred = 0
    n_total_pred = 0
    n_total = 0

    n_correct_pred_or = 0
    n_total_pred_or = 0
    n_total_or = 0

    start = 0

    ner_acc = {}

    while True:
        if start >= len(data):
            break

        end = min(start + args.batchsize, len(data))
        input, sents, cands, targets, ners = dataset.get_minibatch(data, start, end)

        scores, noise_scores = model(input)

        p_noise = torch.nn.functional.sigmoid(noise_scores).cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        for pn, ent, sc, cn, ner in zip(p_noise, targets, scores, cands, ners):
            if ner not in ner_acc:
                ner_acc[ner] = {
                        'total': 0,
                        'total_or': 0,
                        'total_pred': 0,
                        'total_pred_or': 0,
                        'correct_pred': 0,
                        'correct_pred_or': 0
                        }

            n_total += 1
            ner_acc[ner]['total'] += 1

            in_Eplus = ''
            if ent in cn:
                n_total_or += 1
                ner_acc[ner]['total_or'] += 1
                in_Eplus = '*'

            if pn > noise_threshold:
                if data == dataset.test:
                    pass #print('-1' + in_Eplus, end='\t')
                continue
            n_total_pred += 1
            ner_acc[ner]['total_pred'] += 1

            if ent in cn:
                n_total_pred_or += 1
                ner_acc[ner]['total_pred_or'] += 1

            pred = cn[np.argmax(sc)]
            if pred == ent:
                n_correct_pred += 1
                ner_acc[ner]['correct_pred'] += 1
                n_correct_pred_or += 1
                ner_acc[ner]['correct_pred_or'] += 1

                if data == dataset.test:
                    pass #print('1' + in_Eplus, end='\t')
            else:
                if data == dataset.test:
                    pass #print('0' + in_Eplus, end='\t')

        if data == dataset.test:
            pass #print()
        start = end

    prec = n_correct_pred / n_total_pred
    rec = n_correct_pred / n_total
    try:
        f1 = 2 * (prec * rec) / (prec + rec)
    except:
        f1 = 0
    print('all -- prec: %.2f\trec: %.2f\tf1: %.2f' % (prec * 100, rec * 100, f1 * 100))

    prec = n_correct_pred_or / n_total_pred_or
    rec = n_correct_pred_or / n_total_or
    try:
        f1 = 2 * (prec * rec) / (prec + rec)
    except:
        f1 = 0
    print('in E+', n_total_or / n_total * 100)
    print('in E+ -- prec: %.2f\trec: %.2f\tf1: %.2f' % (prec * 100, rec * 100, f1 * 100))

    print('ner')
    print(ner_acc)
    return prec, rec, f1


# for training
def train():
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    data = dataset.train

    best_scores = {'prec': -1, 'rec': -1, 'f1': -1}

    if args.kl_coef > 0:
        print('*** dev ***')
        test(dataset.dev)
        print('*** test ***')
        test(dataset.test)

    print('===== noise_threshold=1 ====')
    print('*** dev ***')
    test(dataset.dev, noise_threshold=1)
    print('*** test ***')
    test(dataset.test, noise_threshold=1)



    for e in range(args.n_epochs):
        print('------------------------- epoch %d --------------------------' % (e))
        random.shuffle(data)
        model.train()
        start = end = 0
        total_loss = 0

        while True:
            if start >= len(data):
                print('%.6f\t\t\t\t\t' % (total_loss / len(data)))
                save = False
                if args.kl_coef > 0:
                    print('*** dev ***')
                    prec, rec, f1 = test(dataset.dev)
                    print('*** test ***')
                    test(dataset.test)

                    if best_scores['f1'] <= f1:
                        best_scores = {'prec': prec, 'rec': rec, 'f1': f1}
                        save = True

                print('===== noise_threshold=0 ====')
                print('*** dev ***')
                prec, rec, f1 = test(dataset.dev, noise_threshold=1)
                print('*** test ***')
                test(dataset.test, noise_threshold=1)

                if args.kl_coef == 0 and best_scores['f1'] <= f1:
                    best_scores = {'prec': prec, 'rec': rec, 'f1': f1}
                    save = True

                if save:
                    print('save model to', args.model_path)
                    model.save(args.model_path)

                break

            end = min(start + args.batchsize, len(data))
            input, sents, cands, _, _ = dataset.get_minibatch(data, start, end)

            optimizer.zero_grad()
            scores, noise_scores = model(input)
            loss, kl = model.compute_loss({
                'scores': scores,
                'noise_scores': noise_scores,
                'real_n_poss': input['real_n_poss'],
                'N_POSS': input['N_POSS']})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5)
            optimizer.step()
            loss = loss.data.cpu().item()

            # print sentence
            if False and end < 1001:
                p_noise = torch.nn.functional.sigmoid(noise_scores)
                for _i in range(end - start):
                    p_noise_i = p_noise[_i]
                    if True: #p_noise_i > args.noise_threshold:
                        scores_i = scores[_i][:input['N_POSS']]
                        sent_i = sents[_i]
                        m_loc_i = input['m_loc'][_i]
                        cands_i = cands[_i][:input['N_POSS']]
                        n_poss_i = input['real_n_poss'][_i].item()

                        words = sent_i.split(' ')
                        words[m_loc_i[0]] = '[' + words[m_loc_i[0]]
                        words[m_loc_i[1] - 1] = words[m_loc_i[1] - 1] + ']'
                        sent_i = ' '.join(words)

                        best_score, best_pred = torch.max(scores_i, dim=0)
                        best_score = best_score.cpu().item()
                        best_pred = best_pred.cpu().item()
                        best_entId = cands_i[best_pred]
                        best_ent = voca_ent.id2word[best_entId]
                        best_types = [voca_type.id2word[t] for t in ent2typeId[best_entId]]
                        best_name = [voca_ent_word.id2word[w] for w in ent2nameId[best_entId]]

                        print('------------------ data point ---------------')
                        print(p_noise_i)
                        print(sent_i)
                        print(n_poss_i)
                        print(best_ent, best_name, best_types, best_score)

                        print('CANDS')
                        for _j in range(n_poss_i):
                            entId_ij = cands_i[_j]
                            ent_ij = voca_ent.id2word[entId_ij]
                            types_ij = [voca_type.id2word[t] for t in ent2typeId[entId_ij]]
                            name_ij = [voca_ent_word.id2word[w] for w in ent2nameId[entId_ij]]
                            print('\t', ent_ij, name_ij, types_ij)


            print("%d\tloss=%.6f\tkl=%.6f\t\t\t" % (end, loss, kl), end='\r' if random.random() < 0.995 else '\n')

            total_loss += loss * (end - start)
            start = end

if __name__ == '__main__':
    if args.mode == 'train':
        train()

    elif args.mode == 'eval':
        if model.config['kl_coef'] > 0:
            print('*** dev ***')
            test(dataset.dev)
            print('*** test ***')
            test(dataset.test)

        print('===== noise_threshold=0 ====')
        print('*** dev ***')
        test(dataset.dev, noise_threshold=1)
        print('*** test ***')
        test(dataset.test, noise_threshold=1)

    else:
        assert(False)
