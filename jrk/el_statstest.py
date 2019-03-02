import sys
from scipy import stats
import numpy as np

model_path = [sys.argv[1], sys.argv[2]]
n_samples = int(sys.argv[3])

model = [[], []]
for j in range(2):
    with open(model_path[j], 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            model[j].extend(line.split('\t'))
    model[j] = np.array(model[j])
    print(len(model[j]))

assert(len(model[0]) == len(model[1]))
size = len(model[1])

def compute_f1(decisions):
    n_correct_pred = 0
    n_total_pred = 0
    n_total = 0

    n_correct_pred_or = 0
    n_total_pred_or = 0
    n_total_or = 0

    for d in decisions:
        n_total += 1

        if d[-1] == '*':
            n_total_or += 1

        if d[0] == '-':
            continue

        n_total_pred += 1

        if d[-1] == '*':
            n_total_pred_or += 1

        if d[0] == '1':
            n_correct_pred += 1
            n_correct_pred_or += 1

    prec = n_correct_pred / n_total_pred
    rec = n_correct_pred / n_total
    try:
        all_f1 = 2 * (prec * rec) / (prec + rec)
    except:
        all_f1 = 0

    prec = n_correct_pred_or / n_total_pred_or
    rec = n_correct_pred_or / n_total_or
    try:
        inE_f1 = 2 * (prec * rec) / (prec + rec)
    except:
        inE_f1 = 0
    return all_f1, inE_f1



model_f1_all = [[], []]
model_f1_inE = [[], []]

for i in range(n_samples):
    chosen_i = np.random.randint(0, size, (size))
    model_chosen = [[], []]

    for j in range(2):
        model_chosen = model[j][chosen_i]
        all_f1, inE_f1 = compute_f1(model_chosen)
        model_f1_all[j].append(all_f1)
        model_f1_inE[j].append(inE_f1)

print(compute_f1(model[0]))
print(compute_f1(model[1]))

print('all:', stats.ttest_rel(model_f1_all[0], model_f1_all[1]))
print('in E:', stats.ttest_rel(model_f1_inE[0], model_f1_inE[1]))

