import numpy as np

class kalman_filter:
    """
    A class that represents kalman filter  
    ...  
    Attributes
    ----------
    R : np.ndarray
        The measurement noise covariance matrix (nz * nz)
    H : np.ndarray
        Observation matrix (nz * nx)
        Transforms the state from the state-space to the measurement-space.
    """

    def __init__(self, R=None, H=None):
        """
        Parameters
        ----------
        R : np.ndarray
            Measurement noise covariance matrix (nz * nz)
        H : np.ndarray
            Observation matrix (nz * nx)
            Transforms the state from the state-space to the 
            measurement-space.
        """
        self.R = R
        self.H = H
    
    def update(self, x_hat, p_hat, z):
        """
        Parameters
        ----------
        x_hat : np.ndarray
            State vector (nx * 1)
            Represents the mathematical prediction of the current state.
        p_hat : np.ndarray
            State covariance matrix (nx * nx)
            Represents the uncertainty of the current state.
        z : np.ndarray
            Measurement vector (nz * 1)
            Sensor measurements at current time step.  
        Returns
        -------
        x : np.ndarray
            State vector (nx * 1)
            Represents the estimated state after filteration.
        p : np.ndarray
            State covariance matrix (nx * nx)
            Represents the estimation uncertainty about the estimated 
            state.
        """
        x, p = self.filter(x_hat, p_hat, z, self.R, self.H)

        return x, p
    
    @staticmethod
    def filter(x_hat, p_hat, z, R, H):
        """
        Parameters
        ----------
        x_hat : np.ndarray
            State vector (nx * 1)
            Represents the mathematical prediction of the current state.
        p_hat : np.ndarray
            State covariance matrix (nx * nx)
            Represents the uncertainty of the current state.
        z : np.ndarray
            Measurement vector (nz * 1)
            Sensor measurements at current time step.  
        R : np.ndarray
            Measurement noise covariance matrix (nz * nz)
        H : np.ndarray
            Observation matrix (nz * nx)
            Transforms the state from the state-space to the 
            measurement-space.
        Returns
        -------
        x : np.ndarray
            State vector (nx * 1)
            Represents the estimated state after filteration.
        p : np.ndarray
            State covariance matrix (nx * nx)
            Represents the estimation uncertainty about the estimated 
            state.
        """
        y = z - H @ x_hat
        S = H @ p_hat @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except:
            S_inv = np.linalg.pinv(S)
        K = p_hat @ H.T @ S_inv
        
        x = x_hat + K @ y
        p = (np.eye(x.shape[0]) - K @ H) @ p_hat

        return x, p

