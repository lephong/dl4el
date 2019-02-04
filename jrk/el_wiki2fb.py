import sys
import re

conll_path = sys.argv[1]
fb2wiki_path = sys.argv[2]
out_path = sys.argv[3]

wiki2fb = {}

prefix = 'http://en.wikipedia.org/wiki/'

# python3 el_wiki2fb.py ../data/EL/AIDA/testa_testb_aggregate_original ../data/freebase/freebase2wiki.txt /tmp/x


def fix_unicode(string):
    j = 0
    ret = ''
    while True:
        string = string[j:]
        try:
            i, j = re.search('\$[0-9A-Fa-f]{4}', string).span()
            ret += (string[:i] if i > 0 else '') + chr(int(string[i+1:j], 16))
        except:
            ret += string
            break
    return ret


print('load wiki2fb')
with open(fb2wiki_path, 'r') as f:
    for line in f:
        comps = line.strip().split()
        if len(comps) == 2:
            fb_ent, wiki_ent = comps
            wiki_ent = fix_unicode(wiki_ent)
            if 'Chris_Lewis' in line:
                print(line, wiki_ent)
            wiki2fb[wiki_ent] = fb_ent
        else:
            print(line)


print('process file')
with open(conll_path, 'r') as fin:
    with open(out_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if line == '':
                fout.write('\n')
                continue
            if line.startswith('-DOCSTART'):
                fout.write(line + '\n')
                continue
            comps = line.split('\t')
            if len(comps) <= 4:
                fout.write(comps[0] + ' ')
            else:
                fout.write(comps[0] + '|||' + comps[1] + '|||' + wiki2fb.get(comps[4][len(prefix):], 'unknown') + ' ')
