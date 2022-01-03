from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC, Recall
from tensorflow.keras.utils import to_categorical

from .model_architecture import ModelArchitecture


class UrlModel(ModelArchitecture):
    def get_model(self):
        model = Sequential()
        model.add(Dense(64, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", Recall(), AUC()],
        )
        return model

    def get_trained_model(self, X, y, validation_data=None):

        model = self.get_model()

        model.fit(
            x=X,
            y=to_categorical(y),
            epochs=5,
            batch_size=32,
            verbose="auto",
            validation_data=validation_data,
        )

        return model


class UrlRf(ModelArchitecture):
    def get_model(self):
        model = RandomForestClassifier(n_estimators=100)
        return model

    def get_trained_model(self, X, y):
        model = self.get_model()
        model.set_params(**{"verbose": 1})
        model.fit(X, y)
        model.set_params(**{"verbose": 0})
        return model
