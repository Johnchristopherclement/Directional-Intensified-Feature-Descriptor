import numpy as np

def calculate_repeatability(feature1, feature2, threshold=0.2):
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)

    # Calculate pairwise distances between features
    #feature1 = feature1.reshape(-1, 1, feature1.shape[-1])
    #feature2 = feature2.reshape(160, -1)

    # Compute the Euclidean distance between the feature descriptors
    #pairwise_distance = np.sqrt(np.sum((feature1 - feature2) ** 2, axis=-1))
    pairwise_distance = np.sqrt(((feature1 - feature2) ** 2))
    #min_pairwise_distance = np.min(pairwise_distances, axis=1)

    # Find matches below the threshold
    matches = np.argwhere(pairwise_distance < threshold)
    # Calculate repeatability as the ratio of matched features to total features
    repeatability = len(matches) / max(len(feature1), len(feature2))

    return repeatability