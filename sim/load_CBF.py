import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod
from numpy.typing import NDArray
#from models.sos_cbf import evaluate_h, evaluate_grad_h
from models.train_nn_cbf import load_model

"""
    This class is used to import the CBF (Control Barrier Funciton) for the simulation and making it available to the simulation.
"""
from abc import ABC, abstractmethod

class CBF(ABC):

    @abstractmethod
    def get_cbf_value(self, state: NDArray[np.float64]) -> float:
        ...

    @abstractmethod
    def get_cbf_gradient(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        ...
"""
class RFFCBF(CBF):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.theta = data["theta"]
        self.W = data["W"]
        self.b = data["b"]

    def get_cbf_value(self, state):
        x = np.asarray(state).reshape(1, -1)
        return float(evaluate_h(x, self.theta, self.W, self.b)[0])

    def get_cbf_gradient(self, state):
        x = np.asarray(state).reshape(1, -1)
        return evaluate_grad_h(x, self.theta, self.W, self.b)[0]

"""
class NNCBF(CBF):
    def __init__(self, eqx_path):
        self.model = load_model(eqx_path)

        @eqx.filter_jit
        def value(model, x):
            return model(x)

        @eqx.filter_jit
        def gradient(model, x):
            return jax.grad(lambda s: model(s))(x)

        self._value = value
        self._gradient = gradient

    def get_cbf_value(self, state):
        x = jnp.asarray(state, dtype=jnp.float32)
        return float(self._value(self.model, x))

    def get_cbf_gradient(self, state):
        x = jnp.asarray(state, dtype=jnp.float32)
        return np.asarray(self._gradient(self.model, x))
