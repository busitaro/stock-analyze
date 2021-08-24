from injector import Module
from predict.predict import Predict
from predict.logic.randomPredict import RandomPredict


class PredictDIModule(Module):
    def configure(self, binder):
        binder.bind(Predict, to=RandomPredict)
