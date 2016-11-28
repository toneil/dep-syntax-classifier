import numpy as np

def vectorize_map(feature_map, feature_list):
    weight_list = []
    for feature in feature_list:
        weight_list.append(feature_map.get(feature, 0))
    return np.array(weight_list)
