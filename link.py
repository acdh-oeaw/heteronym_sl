import gensim.models as models
from gensim.corpora import Dictionary
from gensim.similarities import docsim as similarities
import string
from stop_words import get_stop_words
import matplotlib.patches as ptc

from load_gold import load_gold
from sentence_transformers import SentenceTransformer, util

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np
from nltk.corpus import wordnet
from laserembeddings import Laser
laser = Laser()

import wn
wn.download('ewn:2020')


model1 = SentenceTransformer('stsb-roberta-base')
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

stop_words = get_stop_words('en')

gold_words = ['bass', 'bow', 'desert', 'house', 'heteronym_sl', 'live','raven', 'row', 'wind', 'subject']
        #'can', 'close', 'second','colon', 'won'

def extract_wn_corpus(synsets):

    corpus = []
    for sense in synsets:
        corp = []

        wn = synsets[sense]['def'] + ', ' + synsets[sense]['hyperi'] + ', ' + synsets[sense]['hypo']
        wn = wn.replace(' , ', '')
        for ex in synsets[sense]['ex']:
            wn = wn + ', ' + ex

        wn = wn.translate(str.maketrans('', '', string.punctuation))
        #wn = wn.replace(' ,', ' ')
        #wn = wn.replace(',,', '; ')

        for word in wn.split():
            if word.lower() not in stop_words:
                corp.append(word)


        corpus.append(corp)
    return corpus


def make_vec(wik, dictionary, tfidf):

    doc = []
    el = wik.translate(str.maketrans('', '', string.punctuation))
    #print(el)
    for word in el.split():
        if word.lower() not in stop_words:
            doc.append(word)

    vec_bow = dictionary.doc2bow(doc)
    return tfidf[vec_bow]


def compute_similarity(corpus, file1, file2):

    dictionary = Dictionary(corpus)
    corpus_final = [dictionary.doc2bow(line) for line in corpus]

    tfidf = models.TfidfModel(corpus_final)

    vec_1 = make_vec(file1, dictionary, tfidf)
    vec_2 = make_vec(file2, dictionary, tfidf)

    # perform a similarity query against the corpus
    index = similarities.Similarity(corpus = corpus_final, num_features = len(dictionary), output_prefix = "pqr")
    sims = index[vec_1]
    sims2 = index[vec_2]

    return sims,sims2


def get_max(sims):
    highest = -1
    index = -1
    most_similar = sorted(list(enumerate(sims)), key=lambda x: x[1])
    for el in reversed(most_similar):
        # print(el)
        if el[1] > highest:
            highest = el[1]
            index = el[0]

    #print(index, highest)

    return highest


def get_emb(s):
    embedding = laser.embed_sentences(s, lang='en')
    return embedding


def get_laser_similarity(s1, s2):

    vA = get_emb(s1)
    vB = get_emb(s2)

    vA = np.squeeze(np.asarray(vA))
    vB = np.squeeze(np.asarray(vB))

    n1 = np.linalg.norm(vA)
    n2 = np.linalg.norm(vB)

    #cos = numpy.dot(vA, vB) / (numpy.sqrt(numpy.dot(vA, vA)) * numpy.sqrt(numpy.dot(vB, vB)))
    return np.dot(vA, vB) / n1 / n2


def rebuild_tree(orig_tree):
    node = orig_tree[0]
    children = orig_tree[1:]
    return (node, [rebuild_tree(t) for t in children])


def get_pronunciation(w):
    for row in w.split('\n'):
        if 'pronunciation' in row:
            return row.split(':')[1]


def get_sim_roberta(sentences1, sentences2):

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores[0]


def get_sim_roberta_stsm(sentences1, sentences2):

    embeddings1 = model1.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model1.encode(sentences2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores[0]

def get_wordnet_info(word):
    synsets = {}
    for s in wordnet.synsets(word):
        #print('\n\n',word, s)
        synsets[s.name()] = {}
        synsets[s.name()]['def'] = s.definition()
        synsets[s.name()]['ex'] = s.examples()
        synsets[s.name()]['hypo'] = ''
        synsets[s.name()]['hyper'] = ''
        synsets[s.name()]['hyperi'] = ''

        if s.hyponyms():
            #print("Hyponyms", s.hyponyms())
            for hyp in s.hyponyms():
                synsets[s.name()]['hypo'] = synsets[s.name()]['hypo']+','+ hyp.definition()
            #print("Indirect hyponyms", rebuild_tree(s.tree(lambda x: x.hyponyms()))[1])

        if s.hypernyms():
            #print("Hypernyms", s.hypernyms())
            for hyp in s.hypernyms():
                synsets[s.name()]['hyper'] = synsets[s.name()]['hyper']+','+ hyp.definition()

            for hyp in rebuild_tree(s.tree(lambda x: x.hypernyms()))[1]:
                #print(hyp, hyp[0])
                synsets[s.name()]['hyperi'] = synsets[s.name()]['hyperi'] +','+hyp[0].definition()
            #print("Indirect hypernyms", rebuild_tree(s.tree(lambda x: x.hypernyms()))[1])

    return synsets


def process_wik(wik0):
    w = {}
    for row in wik0.split('\n')[:-1]:
        w[row.split(':')[0].strip()] = row.split(':')[1].strip()

    w0 = w['senses'] + w['examples']
    w0 = w0.replace('[','')
    w0 = w0.replace(']', '')
    w0 = w0.replace('\\n', '')
    w0 = w0.replace('\'', '')

    return w0


def get_float(pos):
    if pos in ['verb','v']:
        return 0.1
    elif pos in ['noun','n']:
        return 0.3
    elif pos in ['adjective','a','s']:
        return 0.5
    else:
        print('weird pos ',pos)
        return 0.7


def get_features(words):
    features = {}
    for word in words:
        print('extracting features for ', word)
        wik0 = open('./wikis/ '+word+'0.txt', 'r') #blank space?
        wik1 = open('./wikis/ '+word+'1.txt', 'r')

        wik0 = wik0.read()
        wik1 = wik1.read()

        w0 = process_wik(wik0)
        w1 = process_wik(wik1)

        synsets = get_wordnet_info(word)

        corpus = extract_wn_corpus(synsets)

        for sense in synsets:

            features[sense] = {'wik0':{}, 'wik1':{},'word':word}

            wn = synsets[sense]['def']+', '+ synsets[sense]['hyperi'] +', '+synsets[sense]['hypo']
            wn = wn.replace(' , ','')
            for ex in synsets[sense]['ex']:
                wn = wn + ', '+ex

            wn = wn.replace(' ,', ' ')
            wn = wn.replace(',,','; ')

            r0 = get_sim_roberta(wn, w0)
            r1 = get_sim_roberta(wn, w1)

            features[sense]['wik0']['roberta'] = r0[0]
            features[sense]['wik1']['roberta'] = r1[0]

            r0_stsm = get_sim_roberta_stsm(wn, w0)
            r1_stsm = get_sim_roberta_stsm(wn, w1)

            features[sense]['wik0']['roberta_stsm'] = r0_stsm[0]
            features[sense]['wik1']['roberta_stsm'] = r1_stsm[0]


            sims, sims2 = compute_similarity(corpus, w0, w1)
            m0 = get_max(sims)
            m1 = get_max(sims2)

            features[sense]['wik0']['tfidf'] = m0
            features[sense]['wik1']['tfidf'] = m1


            sim0 = get_laser_similarity(wn, w0)
            sim1 = get_laser_similarity(wn, w1)
            features[sense]['wik0']['laser'] = sim0
            features[sense]['wik1']['laser'] = sim1

            features[sense]['wik0']['pos_wn']= get_float(sense.split('.')[1])
            features[sense]['wik1']['pos_wn'] = get_float(sense.split('.')[1])

            features[sense]['wik0']['pos'] = get_float(wik0.split('\n')[0].split(':')[1].strip())
            features[sense]['wik1']['pos'] = get_float(wik1.split('\n')[0].split(':')[1].strip())

            features[sense]['wik0']['pos_'] = features[sense]['wik0']['pos_wn'] == features[sense]['wik0']['pos']
            features[sense]['wik1']['pos_'] = features[sense]['wik1']['pos_wn'] == features[sense]['wik1']['pos']

    return features


def make_dataset(features, words=gold_words):
    dataset = {}

    i = 0
    for sense in features:
        #print(sense, f[sense])
        w = features[sense]['word']

        if w not in words:
            print('skipped sense ', sense)
            continue # FIXME

        wik0 = open('./wikis/ ' + w + '0.txt', 'r')  # blank space?
        wik1 = open('./wikis/ ' + w + '1.txt', 'r')

        wik0 = wik0.read()
        wik1 = wik1.read()

        gold = gold_standard[sense]

        dataset[i] = {}
        dataset[i+1] = {}

        for g in gold:
            if g in wik0:
                dataset[i]['label'] = True
                dataset[i + 1]['label'] = False
            elif g in wik1:
                dataset[i]['label'] = False
                dataset[i + 1]['label'] = True
            else:
                print('incorrect value')

          # add wordnet POS tag as feature
        dataset[i]['roberta'] = f[sense]['wik0']['roberta']
        dataset[i]['roberta_stsm'] = f[sense]['wik0']['roberta_stsm']
        dataset[i]['pos'] = f[sense]['wik0']['pos']
        dataset[i]['laser'] = f[sense]['wik0']['laser']
        dataset[i]['tfidf'] = f[sense]['wik0']['tfidf']
        dataset[i]['pos_wn'] = f[sense]['wik0']['pos_wn']
        dataset[i]['pos_'] = f[sense]['wik0']['pos_']

        dataset[i + 1]['roberta'] = f[sense]['wik1']['roberta']
        dataset[i + 1]['roberta_stsm'] = f[sense]['wik1']['roberta_stsm']
        dataset[i + 1]['pos'] = f[sense]['wik1']['pos']
        dataset[i + 1]['laser'] = f[sense]['wik1']['laser']
        dataset[i + 1]['tfidf'] = f[sense]['wik1']['tfidf']
        dataset[i + 1]['pos_wn'] = f[sense]['wik1']['pos_wn']
        dataset[i + 1]['pos_'] = f[sense]['wik0']['pos_']

        i += 2

    return dataset


def plot(data):

    names = list(data.keys())
    values = list(data.values())

    if 0 in data.keys():
        plt.xlabel('Maximum depth')
        plt.plot(names, values, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    else:
        plt.xlabel('Number of estimators')
        plt.plot(names, values,color = 'green', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 12)

    plt.ylabel('F1-score')

    plt.ylim(0.7, 0.9)

    plt.show()


def evaluate(model, test_features, test_labels):
    pr = model.predict(test_features)

    print(classification_report(test_labels, pr))
    accuracy = accuracy_score(test_labels, pr)

    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    return accuracy

def do_experiment(model, features, labels, logR=False, gaus=False):
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
    model.fit(train, train_labels)
    pr = model.predict(test)
    report = classification_report(test_labels, pr, output_dict = True)

    '''
    if gaus:
        imps = permutation_importance(gaus, test, test_labels)
        for i in imps.importances_mean:
            print(i, 1 / i)
    else:
        if logR:
            importance = model.coef_[0]
        else:
            importance = model.feature_importances_

        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
    '''

    exp = {'report': report, 'params': model.get_params(), 'acc': accuracy_score(test_labels, pr)}
    return exp, pr, test_labels


def get_best_model(train, train_labels):
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(train, train_labels)
    best_random = rf_random.best_estimator_
    return best_random, rf_random.best_params_

def train_best_model(features, labels):
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
    best_model, best_params = get_best_model(train, train_labels)
    random_accuracy = evaluate(best_model, test, test_labels)
    print('Best model acc ',random_accuracy)
    pr = best_model.predict(test)
    report = classification_report(test_labels, pr, output_dict = True)
    print(report)
    print(best_params)
    return report, best_params, test_labels


def plot_cutoffs(features, features_stsm, labels):
    bert_preds = {0.1: [], 0.15: [], 0.2: [], 0.25: [], 0.3: [], 0.35: [], 0.4: [], 0.45: [], 0.5: [], 0.55: [],
                  0.6: [], 0.65: [], 0.7: [], 0.75: [], 0.8: []}
    bert_stsm_preds = {0.1: [], 0.15: [], 0.2: [], 0.25: [], 0.3: [], 0.35: [], 0.4: [], 0.45: [], 0.5: [], 0.55: [],
                       0.6: [], 0.65: [], 0.7: [], 0.75: [], 0.8: []}
    laser_preds = {0.1: [], 0.15: [], 0.2: [], 0.25: [], 0.3: [], 0.35: [], 0.4: [], 0.45: [], 0.5: [], 0.55: [],
                   0.6: [], 0.65: [], 0.7: [], 0.75: [], 0.8: []}
    tfidf_preds = {0.1: [], 0.15: [], 0.2: [], 0.25: [], 0.3: [], 0.35: [], 0.4: [], 0.45: [], 0.5: [], 0.55: [],
                   0.6: [], 0.65: [], 0.7: [], 0.75: [], 0.8: []}

    cutoffs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    for el in features:
        for c in cutoffs:
            if el[0] > c:
                bert_preds[c].append(True)
            else:
                bert_preds[c].append(False)
            if el[1] > c:
                laser_preds[c].append(True)
            else:
                laser_preds[c].append(False)
            if el[2] > c:
                tfidf_preds[c].append(True)
            else:
                tfidf_preds[c].append(False)

    for el in features_stsm:
        for c in cutoffs:
            if el[0] > c:
                bert_stsm_preds[c].append(True)
            else:
                bert_stsm_preds[c].append(False)

    data = {'berts': {}, 'lasers': {}, 'tfidfs': {}, 'bert_stsm': {}}

    for c in cutoffs:
        report = classification_report(labels, bert_preds[c], output_dict=True)
        acc = report['macro avg']['f1-score']
        data['berts'][c] = acc
        # print('bert ', c, report)

        report = classification_report(labels, laser_preds[c], output_dict=True)
        acc = report['macro avg']['f1-score']
        data['lasers'][c] = acc
        # print('laser ',c, report)

        report = classification_report(labels, tfidf_preds[c], output_dict=True)
        acc = report['macro avg']['f1-score']
        data['tfidfs'][c] = acc
        # print('tfidf ',c, report)

        report = classification_report(labels, bert_stsm_preds[c], output_dict=True)
        acc = report['macro avg']['f1-score']
        data['bert_stsm'][c] = acc
        # print('bert_stsm ',c, report)

    colors = {'berts': 'c', 'tfidfs': 'm', 'lasers': 'y', 'bert_stsm': 'orange'}
    labels = {'berts': 'S-Bert Paraphrase', 'bert_stsm': 'S-Bert STS', 'tfidfs': 'TFIDF', 'lasers': 'LASER'}
    handles = []

    for d in ['berts', 'bert_stsm', 'lasers', 'tfidfs']:
        names = list(data[d].keys())
        values = list(data[d].values())
        plt.plot(names, values, color=colors[d], label=labels[d])
        patch = ptc.Patch(color=colors[d], label=labels[d])
        handles.append(patch)

    plt.ylabel('F1-score')
    plt.xlabel('Threshold')
    plt.ylim(0.2, 0.9)
    plt.legend(handles=handles)
    plt.show()


random_grid = {'bootstrap': [True, False],
 'max_depth': [None,2,4,6,8,10,15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 400, 600, 800, 1000]}


experiments = []

f = get_features(gold_words)
gold_standard = load_gold()

dataset = make_dataset(f)


features = []
features_stsm = []
labels = []

for row in dataset:
    if 'label' in dataset[row].keys():
        features.append([dataset[row]['roberta'] ,dataset[row]['laser'],
                          dataset[row]['tfidf'] ,dataset[row]['pos'], dataset[row]['pos_wn'] ])

        features_stsm.append([dataset[row]['roberta_stsm'] ,dataset[row]['laser'],
                          dataset[row]['tfidf'] ,dataset[row]['pos'], dataset[row]['pos_wn'] ])

        labels.append(dataset[row]['label'])
    else:
        print('no label ',dataset[row])

plot_cutoffs(features, features_stsm, labels)

r1, best_params, test_labels = train_best_model(features, labels)

#r2 = train_best_model(features_stsm, labels)

'''
data = {}
for est in random_grid['n_estimators']:
    RF = RandomForestClassifier(n_estimators=est, max_depth=best_params['max_depth'],
                            min_samples_split=best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'])

    exp, pr, test_labels = do_experiment(RF, features, labels)
    experiments.append(exp)

    report = classification_report(test_labels, pr, output_dict=True)
    acc = report['macro avg']['f1-score']

    x = est
    y = acc

    data[x] = y
plot(data)

data = {}
for depth in random_grid['max_depth']:

    RF = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=depth,
                            min_samples_split=best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'])

    exp, pr, test_labels = do_experiment(RF, features, labels)
    experiments.append(exp)
    report = classification_report(test_labels, pr, output_dict=True)
    acc = report['macro avg']['f1-score']

    x = depth
    y = acc

    if x==None:
        x=0

    data[x] = y

plot(data)
'''

#LogR = LogisticRegression()
#exp_lr, pr_lr = do_experiment(LogR, logR=True)

#dtree = DecisionTreeClassifier(max_depth=5)
#exp_dt, pr_dt = do_experiment(dtree)

#gaus = GaussianNB()
#exp_g, pr_g = do_experiment(gaus, gaus=True)

