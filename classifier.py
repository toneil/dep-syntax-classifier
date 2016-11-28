from sklearn import linear_model

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

def write_to_file(classifier, path):
    pass

def predict_one(feature_vector, clf_wrapper):
    (clf, _) = clf_wrapper
    return clf.predict([feature_vector])

def score(feature_matrix, target_vector, clf_wrapper):
    (clf, _) = clf_wrapper
    return clf.score(feature_matrix, target_vector)
