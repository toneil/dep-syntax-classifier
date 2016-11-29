import argparse
from os import listdir, devnull, environ
from random import randint, shuffle
from subprocess import call, STDOUT
from nltk.tag import HunposTagger
import tag
import features
import vectors
import classifier

print('==> Imports done')

FNULL = open(devnull, 'w')
TAG_MODEL_PATH = 'tools/suc-suctags.model'
MIDDLE_TAG_DUMP = 'products/tagged.conll'
MIDDLE_MALT_PARSE_DUMP = 'products/parseoutput.conll'

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--features_map', action="store_true")
parser.add_argument('-s', '--feature_set', action="store_true")
parser.add_argument('-f', '--feature_list', type=str)
parser.add_argument('-c', '--categories', type=str)
parser.add_argument('-i', '--infile', type=str)
parser.add_argument('-o', '--outfile', type=str)
parser.add_argument('-n', '--nodes', type=int)
parser.add_argument('-a', '--arcs', type=int)

args = parser.parse_args()

if args.feature_set:
    print("==> Making doc list")
    doc_list = tag.make_doc_list(args.infile)
    tokenized_list = tag.tokenize_list(doc_list)
    ht = HunposTagger(TAG_MODEL_PATH, encoding='utf-8')
    tagged_list = tag.tag_list(tokenized_list, ht)
    tag.write_conll_file(MIDDLE_TAG_DUMP, tagged_list)
    call(['java' ,'-jar' ,'tools/maltparser/maltparser-1.8.1.jar', '-c' ,'swemalt-1.7.2.mco', '-m' ,'parse' , '-i', MIDDLE_TAG_DUMP, '-o', MIDDLE_MALT_PARSE_DUMP], stdout=FNULL)
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

elif args.features_map:
    # SETUP
    # Get features and target classes
    feature_list = features.get_feature_list(args.feature_list)
    categories = features.read_categories(args.categories)
    ht = HunposTagger(TAG_MODEL_PATH)
    clf = classifier.make_classifier()
    doc_paths = tag.get_doc_paths(args.infile)
    shuffle(doc_paths)
    no_of_docs = len(doc_paths)
    doc_counter = 0
    test_matrix = []
    test_targets = []
    # TRAIN ON ONE SAMPLE
    for doc_path in doc_paths:
        doc_counter += 1
        print()
        print("==> Doc {} of {}".format(doc_counter, no_of_docs))
        doc = tag.make_doc(doc_path)
        cat = doc['cat']
        tokenized = tag.tokenize(doc)
        tagged = tag.tag(tokenized, ht)
        # Write to intermediary conll file
        tag.write_conll_file(MIDDLE_TAG_DUMP, tagged)
        print("==> Running MaltParser")
        call(['java' ,'-jar' ,'tools/maltparser/maltparser-1.8.1.jar', '-c' ,'swemalt-1.7.2.mco', '-m' ,'parse' , '-i', MIDDLE_TAG_DUMP, '-o', MIDDLE_MALT_PARSE_DUMP], stdout=FNULL)
        with open(MIDDLE_MALT_PARSE_DUMP) as conll:
            # Get parse vector from parsed file
            parse_v = features.read_parsed_sentences(conll)
            print("==> Extracting features")
            tokens = features.get_tokens(parse_v)
            feature_map = features.make_feature_map(args.nodes, args.arcs, parse_v[0], tokens)
            feature_vector = vectors.vectorize_map(feature_map, feature_list)
            # Set aside random samples for test set
            if randint(0, 9) < 2:
                print("==> Adding {} to test set".format(doc_path))
                test_matrix.append(feature_vector)
                test_targets.append(cat)
            else:
                clf = classifier.train_one_sample(feature_vector, cat, clf, categories)
    # TEST
    print('==> Testing on {} docs out of {}'.format(len(test_targets), no_of_docs))
    for (target, feature_vector) in zip(test_targets, test_matrix):
        predicted = classifier.predict_one(feature_vector, clf)
        print("{} => {}".format(target, predicted))
    print(classifier.score(test_matrix, test_targets, clf))

    # SAVE MODEL TO FILE
    print('==> Saving model to {}'.format(args.outfile))
    classifier.write_to_file(clf, args.outfile)
    print('==> Model saved')
