import numpy as np
from scipy import stats

def featuresarray_load(data_array):
    """
    Extract features from EEG data array
    """
    features = []
    for i in range(data_array.shape[0]):
        # Basic statistical features
        mean = np.mean(data_array[i], axis=-1)
        std = np.std(data_array[i], axis=-1)
        var = np.var(data_array[i], axis=-1)
        ptp = np.ptp(data_array[i], axis=-1)
        skew = stats.skew(data_array[i], axis=-1)
        kurt = stats.kurtosis(data_array[i], axis=-1)
        
        # Combine features
        feature_vector = np.concatenate([
            mean.reshape(-1, 1),
            std.reshape(-1, 1),
            var.reshape(-1, 1),
            ptp.reshape(-1, 1),
            skew.reshape(-1, 1),
            kurt.reshape(-1, 1)
        ], axis=1)
        
        features.append(feature_vector)
    
    return np.array(features) 