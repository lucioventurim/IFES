# Wavelet Spectograms

from sklearn.base import TransformerMixin
import numpy as np
import pywt

from skimage.transform import resize


def create_cwt_images(X, n_scales, rescale_size, wavelet_name="morl"):
    n_samples = X.shape[0]
    n_signals = 1

    # range of scales from 1 to n_scales
    scales = np.arange(1, n_scales + 1)

    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_signals), dtype='float32')


    for sample in range(n_samples):
        serie = X[sample]

        # continuous wavelet transform
        coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
        # resize the 2D cwt coeffs
        rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode='constant')
        X_cwt[sample, :, :, 0] = rescale_coeffs

    return X_cwt



class Wavelet(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # amount of pixels in X and Y
        rescale_size = 227
        # determine the max scale size
        n_scales = 100

        X_cwt = create_cwt_images(X, n_scales, rescale_size)

        print(f"shapes (n_samples, x_img, y_img) of X_train_cwt: {X_cwt.shape}")

        return X_cwt
