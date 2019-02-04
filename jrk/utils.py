from jrk.vocabulary import Vocabulary
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import numbers
import math

############################## removing stopwords #######################

STOPWORDS = {'a', 'after', 'afterwards', 'again', 'against', 'all',
             'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
             'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
             'anything', 'anyway', 'anywhere', 'are', 'around', 'back', 'be',
             'because', 'been', 'before', 'beforehand',
             'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'bottom',
             'but', 'by', 'can', 'cannot', 'cant', 'dont', 'co', 'con', 'could', 'couldnt',
             'de', 'detail', 'due', 'during', 'each', 'eg',
             'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
             'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen',
             'fify', 'fill', 'find', 'fire', 'first', 'five', 'former', 'formerly', 'forty',
             'found', 'four', 'from', 'front', 'full', 'further', 'had',
             'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
             'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred',
             'i', 'ie', 'if', 'in', 'inc', 'indeed', 'is', 'it', 'its', 'itself',
             'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'many', 'may',
             'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
             'much', 'must', 'my', 'myself', 'namely', 'neither', 'never', 'nevertheless',
             'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now',
             'nowhere', 'often', 'once', 'one', 'only', 'or', 'other',
             'others', 'otherwise', 'our', 'ours', 'ourselves', 'own', 'part', 'per',
             'perhaps', 'please', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming',
             'seems', 'serious', 'several', 'she', 'should', 'side', 'since', 'sincere', 'six',
             'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
             'somewhere', 'still', 'such', 'system', 'ten', 'than', 'that', 'the', 'their',
             'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore',
             'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though',
             'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
             'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very',
             'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever',
             'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
             'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
             'with', 'within', 'without', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
             'st', 'years', 'yourselves', 'new', 'known', 'year', 'later',
             'end', 'did', 'just', 'best', 'using'}


def is_important_word(s):
    try:
        if len(s) <= 1 or s.lower() in STOPWORDS:
            return False
        float(s)
        return False
    except:
        return True


def is_stopword(s):
    return s.lower() in STOPWORDS


################################ Similarity #########################

class Sim:
    @staticmethod
    def apply(N, M, batch=False, method='l2'):
        if method == 'l2':
            return -Sim.l2(N, M, batch)
        else:
            assert(False)

    @staticmethod
    def l2(N, M, batch=False):
        if batch:
            dist = (N - M).pow(2).sum(dim=1).sqrt()
        else:
            dist = (N.unsqueeze(dim=1) - M.unsqueeze(dim=0)).pow(2).sum(dim=2).sqrt()
        return dist


############################### coloring ###########################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def tokgreen(s):
    return bcolors.OKGREEN + s + bcolors.ENDC


def tfail(s):
    return bcolors.FAIL + s + bcolors.ENDC


def tokblue(s):
    return bcolors.OKBLUE + s + bcolors.ENDC


############################ process list of lists ###################

def load_voca_embs(voca_path, embs_path, normalization=True, add_pad_unk=True, lower=False, digit_0=False):
    voca, added = Vocabulary.load(voca_path, normalization=normalization, add_pad_unk=add_pad_unk, lower=lower, digit_0=digit_0)
    embs = np.load(embs_path)

    print('org emb shape', embs.shape)

    # check if sizes are matched
    assert((voca.size() - embs.shape[0]) <= 2)
    for w in added:
        if w == Vocabulary.pad_token:
            pad_emb = np.zeros([1, embs.shape[1]])
            embs = np.append(embs, pad_emb, axis=0)
        elif w == Vocabulary.unk_token:
            unk_emb = np.random.uniform(-1, 1, (1, embs.shape[1]))  # np.mean(embs, axis=0, keepdims=True)
            embs = np.append(embs, unk_emb, axis=0)

    print('new emb shape', embs.shape)
    return voca, embs


def make_equal_len(lists, fill_in=0, to_right=True):
    lens = [len(l) for l in lists]
    max_len = max(1, max(lens))
    if to_right:
        if fill_in is None:
            eq_lists = [l + [l[-1].copy() if isinstance(l[-1], list) else l[-1]] * (max_len - len(l)) for l in lists]
        else:
            eq_lists = [l + [fill_in] * (max_len - len(l)) for l in lists]
        mask = [[1.] * l + [0.] * (max_len - l) for l in lens]
    else:
        if fill_in is None:
            eq_lists = [[l[0].copy() if isinstance(l[0], list) else l[0]] * (max_len - len(l)) + l for l in lists]
        else:
            eq_lists = [[fill_in] * (max_len - len(l)) + l for l in lists]
        mask = [[0.] * (max_len - l) + [1.] * l for l in lens]

    return eq_lists, mask

################################## utils for pytorch ############################

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing

    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, numbers.Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def embedding_bag_3D(embs, ids, mode='sum'):
    """
    embs = bachsize x n x dim
    ids = batchsize x m x k
    for i in batch:
        output[i] = embedding_bag(ids[i], embs[i], mode)  # k x dim
    """
    batchsize, n, dim = embs.shape
    assert(batchsize == ids.shape[0])

    ids_flat = ids + Variable(torch.linspace(0, batchsize-1, steps=batchsize).long() * n).view(batchsize, 1, 1).cuda()
    ids_flat = ids_flat.view(batchsize * ids.shape[1], -1)
    embs_flat = embs.view(batchsize * n, dim)
    output_flat = nn.functional.embedding_bag(ids_flat, embs_flat, mode=mode)
    output = output_flat.view(batchsize, ids.shape[1], dim)
    return output


def embedding_3D(embs, ids, mode='sum'):
    """
    embs = bachsize x n x dim
    ids = batchsize x k
    for i in batch:
        output[i] = embedding(ids[i], embs[i])  # k x dim
    """
    batchsize, n, dim = embs.shape
    assert(batchsize == ids.shape[0])

    ids_flat = ids + Variable(torch.linspace(0, batchsize-1, steps=batchsize).long() * n).view(batchsize, 1).cuda()
    ids_flat = ids_flat.view(batchsize * ids.shape[1])
    embs_flat = embs.view(batchsize * n, dim)
    output_flat = nn.functional.embedding(ids_flat, embs_flat)
    output = output_flat.view(batchsize, ids.shape[1], dim)
    return output

