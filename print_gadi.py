import os
import glob
import operator
import collections

files = glob.glob('run_gadi_gpu.sh.o*')
acc = {}
for f in files:
    lines = collections.deque(14*[''], 14)
    with open(f, "r") as file:
        for line in file:
            lines.append(line)
        try:
            acc[file.name] = float(lines[0].split(' ')[5].split(',')[0])
        except:
            pass

max_acc = max(acc.values())
print(max_acc)

for fname in acc:
    if acc[fname] >= max_acc:
        print(fname)
        lines = collections.deque(14*[''], 14)
        with open(fname, "r") as file:
            j = 0
            for line in file:
                lines.append(line)
                if j <= 1:
                    print(line)
                j+=1
            print(lines[0])



