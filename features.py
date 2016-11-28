import os
import re
import itertools
import copy
from nltk.stem import SnowballStemmer

node_depth = 1
arc_depth = 3

# Reads conll file and returns a parse vector
# p_v[doc] = (cat, [sents])
# where sent = [(token, pos_tag, dep_head, label)]
def read_parsed_sentences(file):
    print("==> Reading parser output")
    parse_vector = []
    line_pattern = re.compile(r'(\d+)\t(\S+)\t_\t(\w+)\t\w+\t\S+\t(\d+)\t(\S+)')
    cat_pattern = re.compile(r'1\t_CAT_(\w+)')
    sentence_root = ('ROOT', '_', 0, '_')
    sentence = None
    document = None
    for line in file:
        if line == "\n":
            continue
        if "_CAT_" in line:
            if document:
                parse_vector.append(document)
            cat = cat_pattern.match(line).group(1)
            document = (cat, [])
            continue
        match = line_pattern.match(line)
        if not match:
            print("NO MATCH:", line)
            continue
        word_number = match.group(1)
        token = match.group(2)
        pos_tag = match.group(3)
        dep_head = match.group(4)
        label = match.group(5)
        if word_number == '1':
            if sentence:
                document[1].append(sentence)
            sentence = []
            sentence.append(sentence_root)
        sentence.append((token, pos_tag, dep_head, label))
    parse_vector.append(document)
    return parse_vector

def get_categories(parse_vector):
    categories = set()
    for (cat, _) in parse_vector:
        categories.add(cat)
    return categories

def read_categories(cat_file):
    categories = []
    for cat in cat_file:
        categories.append(cat)
    return categories

# Returns all unique tokens, pos_tags and arc labels in parse_vector
def get_tokens(parse_vector):
    stemmer = SnowballStemmer('swedish')
    tokens = set(["__UNKNOWN_TOKEN__"])
    for (_, sents) in parse_vector:
        for sent in sents:
            for (token, _, _, _) in sent:
                tokens.add(stemmer.stem(token))
    return tokens

def make_feature_set(node_depth, arc_depth, parse_vector):
    print("==> Gathering features")
    feature_set = set(["__UNKNOWN_PATH__"])
    no_of_docs = len(parse_vector)
    docs_treated = 0
    for (cat, sents) in parse_vector:
        for sent in sents:
            for (index, item) in enumerate(sent):
                if index == 0:
                    continue
                try:
                    arcs = tuple(map(lambda item: item[3], get_arcs_to_degree(sent, index, arc_depth)))
                    nodes = tuple(map(lambda item: item[1], get_arcs_to_degree(sent, index, node_depth)))
                # IndexErrors indicate upstream parsing failure
                except:
                    continue
                combined = arcs + nodes
                feature_set.add(combined)
        docs_treated += 1
        print("Treated {} documents of {}".format(docs_treated, no_of_docs))
    return feature_set

# Returns items following n arcs from start
def get_arcs_to_degree(items, start, n):
    returned_items = []
    index = start
    while n > 0:
        #print(index, items)
        next_item = items[index]
        returned_items.append(next_item)
        index = int(next_item[2])
        n -= 1
    return returned_items
def make_empty_map(feature_list_file):
    feature_map = dict()
    for feature in feature_list_file:
        feature = feature.strip()
        feature_map[feature] = 0
    return feature_map

def make_feature_map(node_depth, arc_depth, doc, empty_feature_map, tokens):
    feature_map = copy.copy(empty_feature_map)
    for token in tokens:
        if token not in feature_map:
            feature_map['__UNKNOWN_TOKEN__'] += 1
        else:
            feature_map[token] += 1
    (cat, sents) = doc
    for sent in sents:
        for (index, item) in enumerate(sent):
            if index == 0:
                continue
            try:
                arcs = tuple(map(lambda item: item[3], get_arcs_to_degree(sent, index, arc_depth)))
                nodes = tuple(map(lambda item: item[1], get_arcs_to_degree(sent, index, node_depth)))
            # IndexErrors indicate upstream parsing failure
            except:
                continue
            combined = arcs + nodes
            if combined not in feature_map:
                feature_map['__UNKNOWN_PATH__'] += 1
            else:
                feature_map[combined] += 1
    return feature_map
