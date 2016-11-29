from sklearn import linear_model
import pickle

def train_one_sample(feature_vector, cat, clf_wrapper, categories=None):
    (clf, is_init) = clf_wrapper
    if is_init:
        clf.partial_fit([feature_vector], [cat])
        return clf_wrapper
    else:
        clf.partial_fit([feature_vector], [cat], categories)
        return (clf, True)

def make_classifier():
    clf = linear_model.SGDClassifier()
    return (clf, False)

def write_to_file(clf_wrapper, path):
    (clf, _) = clf_wrapper
    with open(path, 'w') as outfile:
        clf.sparisfy()
        pickle.dump(clf_wrapper, outfile)

def load_from_file(path):
    with open(path) as infile:
        clf_wrapper = pickle.load(path)
        return clf_wrapper

def predict_one(feature_vector, clf_wrapper):
    (clf, _) = clf_wrapper
    return clf.predict([feature_vector])

def score(feature_matrix, target_vector, clf_wrapper):
    (clf, _) = clf_wrapper
    return clf.score(feature_matrix, target_vector)
