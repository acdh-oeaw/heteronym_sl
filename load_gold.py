import re

def load_gold(file = 'gold.txt'):
    f = open(file, 'r')
    txt = f.read()
    gold = {}
    i = 0
    for line in txt.split('\n\n'):
        if line=='\n':
            continue
        l = line.strip()
        sense = l.split(':')[0]
        gold[sense] = []
        #print(l)
        x = re.findall(r'\/(.*?)\/',l)
        #print(x)
        for el in x:
            gold[sense].append(el)
        i+=1

    return gold