import numpy as np

def calculate_repeatability(feature1, feature2, threshold=0.2):
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    pairwise_distance = np.sqrt(((feature1 - feature2) ** 2))
    # Find matches below the threshold
    matches = np.argwhere(pairwise_distance < threshold)
    # Calculate repeatability as the ratio of matched features to total features
    repeatability = len(matches) / max(len(feature1), len(feature2))

    return repeatability