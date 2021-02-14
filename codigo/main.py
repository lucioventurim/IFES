from mfpt import MFPT
from paderborn import Paderborn
from experimenter import Experimenter
from classifiers import Classifiers, Scoring


def main():
    debug = 0
    sample_size = 8192

    database = MFPT(debug=debug)
    # database = Paderborn(debug=debug)

    database_acq = database.load()
    # print(database_acq)

    database_exp = Experimenter(database_acq, sample_size)
    database_exp.perform(Classifiers(), Scoring())


if __name__ == "__main__":
    main()
