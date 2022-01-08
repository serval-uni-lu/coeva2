from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC, Recall
from tensorflow.keras.utils import to_categorical

from .model_architecture import ModelArchitecture


class LcldModel(ModelArchitecture):
    def get_model(self):
        model = Sequential()
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
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

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        early_stop = EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=25
        )
        model.fit(
            x=X_train,
            y=to_categorical(y_train),
            epochs=100,
            batch_size=512,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
        )
        return model


class LcldRf(ModelArchitecture):
    def get_model(self):
        params = {
            "n_estimators": 100,
            # "min_samples_split": 6,
            # "min_samples_leaf": 2,
            # "max_depth": 10,
            # "bootstrap": True,
        }
        model = RandomForestClassifier(**params)
        return model

    def get_trained_model(self, X, y):

        model = self.get_model()
        model.set_params(**{"verbose": 1, "n_jobs": -1})
        model.fit(X, y)
        model.set_params(**{"verbose": 0, "n_jobs": 1})
        return model
