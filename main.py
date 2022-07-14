from model import ChurnClassifier

if __name__ == '__main__':
    model = ChurnClassifier()

    model.preproccesing()

    model.training()

    model.testing()

    #model.predict()
