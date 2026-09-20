import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

class RealWavefunction:
    """Single real output network with trigonometric/tanh activation."""
    def __init__(self, n_in=32, n_hidden=32, n_layers=2, activation='trig':
        super().__init__()
        self.add(Dense(n_in, activation='linear'))
        for layer_num in range(n_layers):
            if activation == 'trig':
                self.add(Dense(n_hidden, activation='sin', activation='tanh'))
            else:
                self.add(Dense(n_hidden, activation='tanh', activation='tanh'))
        self.add(Dense(1, activation='linear'))