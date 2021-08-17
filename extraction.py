import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.text import TextCollection
import pandas as pd
from sklearn.cluster import DBSCAN
import sklearn.svm as svm
import numpy as np


train = 'train_text.txt'
codebooks = ['Platzeck', 'Anike Peters', 'Matthias Platzeck', 'SPD', 'Greenpeace', 'Peters']  # important ORG and PER

# 小写codebooks
codebook = []
for i in range(len(codebooks)):
    codebook.append(codebooks[i].lower())
# print(codebook)


def getText(filename):
    sentences_seq = []
    headline_words = []
    f = open(filename, 'r', encoding='utf-8')

    for line in f.readlines():
        line = line.strip()

        if line.startswith('<headline>') and line.endswith('</headline>'):
            word = line.split()
            for i in range(len(word)):
                if word[i].startswith('<headline>'):
                    word[i] = word[i][10:]
                elif word[i].endswith('</headline>'):
                    word[i] = word[i][:-11]
                headline_words.append(word[i].lower())
            continue

        if len(line) > 0:
            if line.startswith('<text>'):
                line = line[6:]
            elif line.endswith('</text>'):
                line = line[:-8]
            sentences_seq.append(line)

    f.close()
    # print('headline_words - {}'.format(headline_words))
    # print('sentences_seq - {}'.format(sentences_seq))

    return sentences_seq, headline_words


def getClassFeatures(S, H):
    # assign POS tags
    tokens = []
    tags = []
    for i in range(len(S)):
        token = word_tokenize(S[i])
        tokens.append(token)
        tag = pos_tag(token)
        tags.append(tag)
    # print('tokens - {}'.format(tokens))
    # print('tags - {}'.format(tags))

    corpus = TextCollection(tokens)
    # print(corpus)

    # entity recognition
    ne_chunked_sents = [nltk.ne_chunk(tag) for tag in tags]
    # print(ne_chunked_sents)

    #  extract all named entities
    named_entities = []
    for ne_tagged_sentence in ne_chunked_sents:
        for tagged_tree in ne_tagged_sentence:
            #  extract only chunks having NE labels
            if hasattr(tagged_tree, 'label'):
                entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  # get NE name
                entity_type = tagged_tree.label()  # get NE category
                named_entities.append((entity_name, entity_type))
                #  get unique named entities
                named_entities = list(set(named_entities))

    # store named entities in a data frame
    entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type'])
    # display results
    # print(entity_frame)
    # print('named_entities - {}'.format(named_entities))

    classification_features = []
    for sentence in S:
        imp_organs = []
        imp_pers = []
        organs = []
        pers = []
        head_words = []
        tfidf_sum = 0

        splitted_sentence = sentence.split()
        # print(splitted_sentence)
        for i in range(len(splitted_sentence)):
            if splitted_sentence[i].lower() in H:
                head_words.append(splitted_sentence[i].lower())

            for named_entity in named_entities:
                if splitted_sentence[i].lower() == named_entity[0].lower() and splitted_sentence[i].lower() in codebook:
                    if named_entity[1] == 'ORGANIZATION':
                        imp_organs.append(splitted_sentence[i].lower())
                    elif named_entity[1] == 'PERSON':
                        imp_pers.append(splitted_sentence[i].lower())

                if splitted_sentence[i].lower() == named_entity[0].lower():
                    if named_entity[1] == 'ORGANIZATION':
                        organs.append(splitted_sentence[i].lower())
                    elif named_entity[1] == 'PERSON':
                        pers.append(splitted_sentence[i].lower())

            tfidf_sum += corpus.tf_idf(splitted_sentence[i], sentence)
        tfidf_aver = tfidf_sum / len(splitted_sentence)

        imp_organs = list(set(imp_organs))
        imp_pers = list(set(imp_pers))
        organs = list(set(organs))
        pers = list(set(pers))
        head_words = list(set(head_words))

        # print('imp_organs - {}'.format(imp_organs))
        # print('imp_persons - {}'.format(imp_pers))
        # print('organs - {}'.format(organs))
        # print('persons - {}'.format(pers))
        # print('head_words - {}'.format(head_words))

        # k值
        k1 = len(imp_organs)
        k2 = len(imp_pers)
        k3 = len(organs)
        k4 = len(pers)
        k5 = len(head_words)
        k6 = tfidf_aver

        classification_features.append([k1, k2, k3, k4, k5, k6])
    # print(classification_features)
    return classification_features


# SVM Classifier
def svmClassifier(train_features, pred_features):
    # train
    svm_model = svm.SVC()
    X0 = train_features  # 训练数据
    y = ['irrelevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'irrelevant']  # 标签
    svm_model.fit(X0, y)

    # predict
    X1 = pred_features
    Y_pred = svm_model.predict(X1)
    print(Y_pred)

    return Y_pred


# 1. Learning Relevant Sentences
def classificationModel(S, H):
    classification_features = getClassFeatures(S, H)
    pred_rst = svmClassifier(classification_features, classification_features)
    return pred_rst


# 2. Filtering by Density-Based Clustering
def array_pad(l):
    max_len = 0
    for i in range(len(l)):
        if len(l[i]) > max_len:
            max_len = len(l[i])
    # print(max_len)
    for i in range(len(l)):
        pad_len = max_len - len(l[i])
        # print(pad_len)
        for j in range(pad_len):
            l[i].append(0)
    return l


def clusteringModel(S, H):
    pred_rst = classificationModel(S, H)
    # select all sentences which are classified as relevant
    rel_sents = []
    rel_sents_idx = []
    splitted_rel_sents = []
    splitted_rel_sents_words = []
    for i in range(len(pred_rst)):
        if pred_rst[i] == 'relevant':
            rel_sents.append(S[i])
            rel_sents_idx.append(i)
    # print(rel_sents)
    # print(rel_sents_idx)

    clustering_features = []
    characters = '`~!@#$%^&*()_-+={}|[]\\:";\'<>?,./'
    for i in range(len(rel_sents)):
        sing = rel_sents[i].split()
        splitted_rel_sents.append(sing)
        for word in sing:
            if word in characters:
                continue
            splitted_rel_sents_words.append(word.lower())

    # print(splitted_rel_sents)
    # print(splitted_rel_sents_words)

    for i in range(len(splitted_rel_sents)):
        features = []
        for j in range(len(splitted_rel_sents[i])):
            count = splitted_rel_sents_words.count(splitted_rel_sents[i][j].lower())
            features.append(count)
        clustering_features.append(features)
        # print(features)

    # print(clustering_features)

    clustering_features = array_pad(clustering_features)

    noise_cluster = []

    X2 = np.array(clustering_features).reshape(len(clustering_features), -1)
    # print(X2)
    db = DBSCAN(eps=1, min_samples=2).fit(X2)
    # print(db.labels_)  # -1是噪点
    for i in range(len(db.labels_)):
        if db.labels_[i] == -1:
            noise_cluster.append(i)
    # print(noise_cluster)

    return rel_sents_idx, noise_cluster


# 3. Statement Extraction Step
# Our technique combines sentences which are classified as relevant by our SVM
# and do not belong to any cluster in DBSCAN clustering.


def kmp(mom_string, son_string):
    # 传入一个母串和一个子串
    # 返回子串匹配上的第一个位置，若没有匹配上返回-1
    test = ''
    if type(mom_string) != type(test) or type(son_string) != type(test):
        return -1
    if len(son_string) == 0:
        return 0
    if len(mom_string) == 0:
        return -1
    # 求next数组
    next = [-1] * len(son_string)
    if len(son_string) > 1:  # 这里加if是怕列表越界
        next[1] = 0
        i, j = 1, 0
        while i < len(son_string) - 1:  # 这里一定要-1，不然会像例子中出现next[8]会越界的
            if j == -1 or son_string[i] == son_string[j]:
                i += 1
                j += 1
                next[i] = j
            else:
                j = next[j]

    # kmp框架
    m = s = 0  # 母指针和子指针初始化为0
    while (s < len(son_string) and m < len(mom_string)):
        # 匹配成功,或者遍历完母串匹配失败退出
        if s == -1 or mom_string[m] == son_string[s]:
            m += 1
            s += 1
        else:
            s = next[s]

    if s == len(son_string):  # 匹配成功
        return m-s, m-1
    # 匹配失败
    return -1


def combine(str1, str2):
    return str1 + str2


# Combine Sentences to Statements
def generageStatement(train_data):
    statements = []
    S, H = getText(train_data)
    rel_idx, db_rst = clusteringModel(S, H)

    for i in range(len(db_rst)):
        for j in range(len(rel_idx)):
            statements.append(S[rel_idx[db_rst[i]]])

    statements = list(set(statements))

    text = ''
    for i in range(len(S)):
        text += S[i]
    # print(text)

    i = 0
    j = 0

    while i < len(statements):
        while j < len(statements):
            start_i, end_i = kmp(text, statements[i])
            start_j, end_j = kmp(text, statements[j])
            # print(start_i, end_i, start_j, end_j)
            if end_i + 1 == start_j:
                tempi = statements[i]
                tempj = statements[j]
                statements.remove(tempi)
                statements.remove(tempj)
                statements.append(combine(tempi, tempj))
                i = 0
                j = 0
            elif end_j + 1 == start_i:
                tempi = statements[i]
                tempj = statements[j]
                statements.remove(tempi)
                statements.remove(tempj)
                statements.append(combine(tempj, tempi))
                i = 0
                j = 0
            else:
                j += 1
        i += 1

    return statements


if __name__ == '__main__':
    print(generageStatement(train))
