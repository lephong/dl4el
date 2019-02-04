import sys

in_path = sys.argv[1]
out_path = sys.argv[2]

TARGET_REL = "type.instance>"
PREFIX = "http://rdf.freebase.com/ns/"


def clean_str(s):
    if s[0] == '<':
        s = s[1:-1]
    if not s.startswith(PREFIX):
        return None
    else:
        return s[len(PREFIX):]

if __name__ == "__main__":
    e2c_list = {}
    with open(in_path, 'r') as f:
        for i, line in enumerate(f):
            if (i + 1) % 1000000 == 0:
                print(i, len(e2c_list), end='\r')
            comps = line.strip().split('\t')
            if len(comps) == 4 and comps[1].endswith(TARGET_REL):
                hstr = clean_str(comps[0])
                tstr = clean_str(comps[2])
                if hstr is not None and tstr is not None:
                    c_list = e2c_list.get(tstr)
                    if c_list is None:
                        c_list = []
                        e2c_list[tstr] = c_list
                    c_list.append(hstr)

    with open(out_path, 'w') as f:
        for ent, c_list in e2c_list.items():
            f.write(ent + "\t")
            for c in c_list:
                f.write(c + " ")
            f.write("\n")
