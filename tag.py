from os import listdir
import re
import nltk
from nltk.tag import HunposTagger
from nltk import word_tokenize

def get_doc_paths(dir_path):
    doc_filenames = listdir(dir_path)
    doc_paths = []
    for doc_name in doc_filenames:
        if doc_name.startswith('.'):
            continue
        doc_path = dir_path + '/' + doc_name
        doc_paths.append(doc_path)
    return doc_paths

def get_cat(docname):
    pattern = r'.*?\w+02([A-Za-z]+)\d+'
    m = re.match(pattern, docname)
    if m:
        return m.group(1)
    else:
        return None

def stripText(text):
    patterns = [
        re.compile(r'\r'), # CR
        re.compile(r'.*?Motion till riksdagen\n.*\n.*\n.*\n'), # Intro
        re.compile(r'[A-Z]\w+ den \d{1,2} [a-z]+ \d{4,4}\n'), # Time and place
        re.compile(r'\n[A-Z]\w+.*?\([a-z]+\)'), # MPs
        re.compile(r'^\n*'), # Leading NLs
        re.compile(r'\n*$') # Trailing NLs
    ]
    sub_text = text
    for pattern in patterns:
        sub_text = pattern.sub("", sub_text)
    return sub_text

def make_doc_list(doc_dir_path):
    doclist = []
    doc_paths = get_doc_paths(doc_dir_path)
    for doc_path in doc_paths:
        doc = make_doc(doc_path)
        doclist.append(doc)
    return doclist

def make_doc(doc_path):
    with open(doc_path, 'r', encoding='utf-8') as doc_file:
        cat = get_cat(doc_path)
        if cat is None:
            cat = '__UNKNOWN_CAT__'
        text = stripText(doc_file.read())
        return {'cat': cat, 'text': text}

def tokenize_list(doc_list):
    tokenized_list = []
    for doc in doc_list:
        tokenized_doc = tokenize(doc)
        tokenized_list.append(tokenized_doc)
    return tokenized_list

def tokenize(doc):
    tokens = word_tokenize(doc['text'])
    return {'cat': doc['cat'], 'tokens': tokens}

def tag_list(tokenized_list, tagger):
    print("==> Tagging doc list")
    pos_split_char = '_'
    tagged_list = []
    number_of_docs = len(tokenized_list)
    i = 0
    for doc in tokenized_list:
        i += 1
        print("{} of {} docs tagged".format(i, number_of_docs))
        cat, token_syntax_morph = tag(doc, tagger)
        tagged_list.append((cat, token_syntax_morph))
    return tagged_list

def tag(tokenized_doc, tagger):
    pos_split_char = '_'
    tagged = tagger.tag(tokenized_doc['tokens'])
    token_syntax_morph = []
    for (token, stx_morph_bytes) in tagged:
        if stx_morph_bytes is None:
            continue
        stx_morph = stx_morph_bytes.decode('utf-8')
        syntax = stx_morph.split(pos_split_char)[0]
        morph = pos_split_char.join(stx_morph.split(pos_split_char)[1:])
        if token == '':
            token = '_'
        if morph == '':
            morph = '_'
        if syntax == '':
            syntax = '_'
        token_syntax_morph.append((token, syntax, morph))
    return (tokenized_doc['cat'], token_syntax_morph)

def write_conll_file(path, tagged):
    print('==> Writing conll')
    if type(tagged) != list:
        tagged = [tagged]
    with open(path, 'w') as conll_file:
        for (cat, token_syntax_morph) in tagged:
            print('\n1\t_CAT_{}\t_\t_\t_\t_\n'.format(cat), file=conll_file)
            row = 0
            for (token, syntax, morph) in token_syntax_morph:
                row += 1
                line = "{0}\t{1}\t_\t{2}\t{2}\t{3}".format(row, token, syntax, morph)
                print(line, file=conll_file)
                if morph == 'MAD':
                    print('', file=conll_file)
                    row = 0
