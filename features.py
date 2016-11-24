import os
import re

node_depth = 2
arc_depth = 2
ARC_SET = ['++','+A','+F','AA','AG','AN','AT','CA','DB','DT','EF','EO','ES','ET','FO','FP','FS','FV','I?','IC','IG','IK','IM','IO','IP','IQ','IR','IS','IT','IU','IV','JC','JG','JR','JT','KA','MA','MS','NA','OA','OO','OP','PL','PR','PT','RA','SP','SS','TA','TT','UK','VA','VO','VS','XA','XF','XT','XX','YY','CJ','HD','IF','PA','UA','VG']

# Reads conll file and returns a parse vector
# p_v[doc] = (cat, [sents])
# where sent = [(token, pos_tag, dep_head, label)]
def read_parsed_sentences(file):
    parse_vector = []
    line_pattern = re.compile(r'(\d+)\t(\S+)\t_\t(\w+)\t\w+\t(\w+)\t(\d+)\t(\w+)')
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
            continue
        word_number = match.group(1)
        token = match.group(2)
        pos_tag = match.group(3)
        morph = match.group(4)
        dep_head = match.group(5)
        label = match.group(6)
        if word_number == '1':
            if sentence:
                document[1].append(sentence)
            sentence = []
            sentence.append(sentence_root)
        sentence.append((token, pos_tag, dep_head, label))
    parse_vector.append(document)
    return parse_vector

def make_feature_matrix(nodes, arcs, parse_vector):
    pass

with open('parseoutput.conll') as conll:
    parse_v = read_parsed_sentences(conll)
    feat_matrix = make_feature_matrix(node_depth, arc_depth, parse_v)
    print (parse_v[2][1][4])
