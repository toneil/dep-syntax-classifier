import argparse
from subprocess import call
from nltk.tag import HunposTagger
import tag
import features
import vectors
import classifier

print('==> Imports done')

TAG_MODEL_PATH = 'model/suc-suctags.model'
MIDDLE_TAG_DUMP = 'tagged.conll'
MIDDLE_MALT_PARSE_DUMP = 'parseoutput.conll'

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--features_map', action="store_true")
parser.add_argument('-s', '--feature_set', action="store_true")
parser.add_argument('-f', '--feature_list', type=str)
parser.add_argument('-c', '--categories', type=str)
parser.add_argument('-i', '--infile', type=str)
parser.add_argument('-o', '--out', type=str)
parser.add_argument('-n', '--nodes', type=int)
parser.add_argument('-a', '--arcs', type=int)

args = parser.parse_args()

if args.feature_set:
    doc_list = tag.make_doc_list(args.infile)
    tokenized_list = tag.tokenize_list(doc_list)
    ht = HunposTagger(TAG_MODEL_PATH)
    tagged_list = tag.tag_list(tokenized_list, ht)
    tag.write_conll_file(MIDDLE_TAG_DUMP, tagged_list)
    call(['java' ,'-jar' ,'maltparser-1.8.1/maltparser-1.8.1.jar', '-c' ,'swemalt-1.7.2.mco', '-m' ,'parse' , '-i', MIDDLE_TAG_DUMP, '-o', MIDDLE_MALT_PARSE_DUMP])
    # Extract and write categories and features to file
    with open(MIDDLE_MALT_PARSE_DUMP) as conll_file, \
         open(args.feature_list, 'w') as feature_file, \
         open(args.categories, 'w') as category_file:
        # Extract
        parse_v = features.read_parsed_sentences(conll_file)
        categories = features.get_categories(parse_v)
        tokens = features.get_tokens(parse_v)
        feature_set = features.make_feature_set(args.nodes, args.arcs, parse_v)
        # Write
        for cat in categories:
            print(cat, file=category_file)
        for token in tokens:
            print(token, file=feature_file)
        for feature in feature_set:
            print(feature, file=feature_file)

elif args.features_map and args.out:
    # SETUP
    # Get features and target classes
    with open(args.feature_list) as feature_list_file, open(args.categories) as category_file:
        empty_feature_map = features.make_empty_map(feature_list_file)
        categories = features.read_categories(category_file)
    clf = classifier.make_classifier()
    feature_list = list(empty_feature_map)
    feature_list.sort()
    # TRAIN ONE SAMPLE
    # Assign category and tag text
    doc = tag.make_doc(args.infile)
    cat = doc['cat']
    tokenized = tag.tokenize(doc)
    ht = HunposTagger(TAG_MODEL_PATH)
    tagged = tag.tag(tokenized, ht)
    # Write to intermediary conll file
    tag.write_conll_file(MIDDLE_TAG_DUMP, tagged)
    # Run maltparser
    call(['java' ,'-jar' ,'maltparser-1.8.1/maltparser-1.8.1.jar', '-c' ,'swemalt-1.7.2.mco', '-m' ,'parse' , '-i', MIDDLE_TAG_DUMP, '-o', MIDDLE_MALT_PARSE_DUMP])
    with open(MIDDLE_MALT_PARSE_DUMP) as conll, open(args.out, 'w') as feature_file:
        # Get parse vector from parsed file
        parse_v = features.read_parsed_sentences(conll)
        # Get all tokens in doc
        tokens = features.get_tokens(parse_v)
        feature_map = features.make_feature_map(args.nodes, args.arcs, parse_v[0], empty_feature_map, tokens)
        feature_vector = vectors.vectorize_map(feature_map, feature_list)
        clf = classifier.train_one_sample(feature_vector, cat, clf, categories)

        print(cat, classifier.predict_one(feature_vector, clf))
