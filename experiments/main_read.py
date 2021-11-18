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

    metrics.scores("2021.11.16_20.26.27.csv")


if __name__ == "__main__":
    main()
