from utils import persist_results, metrics
import os
import numpy as np

def sig_image(data,size):
    X=np.zeros((data.shape[0],size,size))
    for i in range(data.shape[0]):
        X[i]=(data[i,:].reshape(size,size))
    return X.astype(np.float16)

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    metrics.scores("2021.06.10_12.29.24.csv")

    """
    data = np.load('raw_signal_data.npy')

    #print(data)
    print(data.shape)

    x_n = sig_image(data, 50)

    print(x_n)
    print(x_n.shape)
    """

if __name__ == "__main__":
    main()
