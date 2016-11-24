from os import listdir
import re
import nltk
from nltk.tag import HunposTagger
from nltk import word_tokenize

print('IMPORT DONE')

def get_cat(docname):
    pattern = r'\w+02([A-Za-z]+)\d+'
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

def make_doclist(doc_dir_path):
    doclist = []
    doc_filenames = listdir(doc_dir_path)
    for doc_name in doc_filenames:
        doc_path = doc_dir_path + '/' + doc_name
        with open(doc_path, 'r', encoding='utf-8') as doc_file:
            cat = get_cat(doc_name)
            if cat is None:
                continue
            text = stripText(doc_file.read())
            doclist.append({
                'cat': cat,
                'text' : text
            })
    return doclist

def tokenize(doc_list):
    tokenized_list = []
    for doc in doc_list:
        tokens = word_tokenize(doc['text'])
        tokenized_list.append({
            'tokens' : tokens,
            'cat': doc['cat']
        })
    return tokenized_list

def tag(tokenized_list, tagger):
    pos_split_char = '_'
    tagged_list = []
    number_of_docs = len(tokenized_list)
    i = 0
    for doc in tokenized_list:
        i += 1
        print("{} of {} docs tagged".format(i, number_of_docs))
        tagged = tagger.tag(doc['tokens'])
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
        tagged_list.append((doc['cat'], token_syntax_morph))
    return tagged_list

def write_conll_file(path, tagged_list):
    print('writing conll')

    with open(path, 'w') as conll_file:
        for (cat, token_syntax_morph) in tagged_list:
            print('\n1\t_CAT_{}\t_\t_\t_\t_\n'.format(cat), file=conll_file)
            row = 0
            for (token, syntax, morph) in token_syntax_morph:
                row += 1
                line = "{0}\t{1}\t_\t{2}\t{2}\t{3}".format(row, token, syntax, morph)
                print(line, file=conll_file)
                if morph == 'MAD':
                    print('', file=conll_file)
                    row = 0


# Corpus file path
DOC_DIR_PATH = 'testdata'
# CoNLL file path
CONLL_PATH = 'tagged.conll'
# Tagger model path
TAG_MODEL_PATH = 'model/suc-suctags.model'

doc_list = make_doclist(DOC_DIR_PATH)
tokenized_list = tokenize(doc_list)
ht = HunposTagger(TAG_MODEL_PATH)
tagged_list = tag(tokenized_list, ht)
write_conll_file(CONLL_PATH, tagged_list)

#java -jar maltparser-1.8.1/maltparser-1.8.1.jar -c swemalt-1.7.2.mco -m parse -i tagged.conll -o parseoutput.conll 
