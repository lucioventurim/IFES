from utils import persist_results, metrics
import os


def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    metrics.scores("2021.05.07_15.50.00.csv")


if __name__ == "__main__":
    main()
