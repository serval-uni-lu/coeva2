from typing import Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from attacks.coeva2.constraints import Constraints
import autograd.numpy as anp
import pandas as pd
import pickle

class LcldConstraints(Constraints):
    def __init__(
        self,
        #amount_feature_index: int,
        feature_path: str,
        constraints_path: str,
    ):
        self._provision_constraints_min_max(constraints_path)
        self._provision_feature_constraints(feature_path)
        self._fit_scaler()
        #self._amount_feature_index = amount_feature_index

    def _fit_scaler(self) -> None:
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        min_c, max_c = self.get_constraints_min_max()

        self._scaler = self._scaler.fit([min_c, max_c])

    @staticmethod
    def _date_feature_to_month(feature):
        return np.floor(feature / 100) * 12 + (feature % 100)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # ----- PARAMETERS

        tol = 1e-3
        # should write a function in utils for this part
        with open('../data/lcld/section_names_idx.pkl', 'rb') as f:
            section_names_idx = pickle.load(f)

        with open('../data/lcld/imports_idx.pkl', 'rb') as f:
            imports_idx = pickle.load(f)

        with open('../data/lcld/dll_imports_idx.pkl', 'rb') as f:
            dll_imports_idx = pickle.load(f)

        with open('../data/lcld/freq_idx.pkl', 'rb') as f:
            freq_idx = pickle.load(f)

        # NumberOfSections equals the sum of sections names not set to 'none'(label encoded to 832)
        g1 = np.absolute(x[:,12893] - np.count_nonzero(x[:,section_names_idx]!=832, axis=1))

        # header_FileAlignment < header_SectionAlignment
        g2 = x[:,13956] - x[:,10840]


        #The value for FileAlignment should be a power of 2
        m = x[:,13956]
        m = np.array(m, dtype=np.float)
        g3 = np.absolute( np.log2(m, out=np.zeros_like(m), where=(m!=0)) % 1  - 0)
        
       
        #api_import_nb is higher than the sum of total imports that we have considered as features
        g4 = np.sum(x[:,imports_idx], axis=1) - x[:,271]

        # api_dll_nb is higher than the sum of total dll that we have considered as features
        g5 = np.sum(x[:,dll_imports_idx], axis=1) - x[:,8607]

        #Sum of individual byte frequencies is equal to 1. There is a small  difference due to rounding effect
        g6 = np.absolute(1-np.sum(x[:,freq_idx],axis=1))

        # FileEntropy is related to freqbytes through Shanon entropy
        m = x[:,freq_idx]
        m = np.array(m, dtype=np.float)
        logarithm = np.log2(m, out=np.zeros_like(m), where=(m!=0))
        g7 = np.absolute(x[:,23549] + np.sum(x[:,freq_idx]*logarithm,axis=1))

        constraints = anp.column_stack(
            [g1, g2, g3, g4, g5, g6, g7]
        )
        constraints[constraints <= tol] = 0.0

        return constraints

    def get_nb_constraints(self) -> int:
        return 10

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return self._scaler.transform(x)

    def get_constraints_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._constraints_min, self._constraints_max

    def get_mutable_mask(self) -> np.ndarray:
        return self._mutable_mask

    def get_feature_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._feature_min, self._feature_max

    def get_feature_type(self) -> np.ndarray:
        return self._feature_type

    def get_amount_feature_index(self) -> int:
        return self._amount_feature_index

    def _provision_feature_constraints(self, path: str) -> None:
        df = pd.read_csv(path, low_memory=False)
        self._feature_min = df["min"].to_numpy()
        self._feature_max = df["max"].to_numpy()
        self._mutable_mask = df["mutable"].to_numpy()
        self._feature_type = df["type"].to_numpy()

    def _provision_constraints_min_max(self, path: str) -> None:
        df = pd.read_csv(path, low_memory=False)
        self._constraints_min = df["min"].to_numpy()
        self._constraints_max = df["max"].to_numpy() 
        self._fit_scaler()
