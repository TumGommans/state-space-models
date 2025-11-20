"""AR(p) model estimation using Ordinary Least Squares (OLS)."""

import numpy as np

from typing import Dict

class ARModel:
    """Autoregressive model of order p estimated using Ordinary Least Squares."""
    
    def __init__(self, p: int):
        """Initialize the AR(p) model.
        
        Args:
            p: Order of the autoregressive model (must be positive)
            
        Raises:
            ValueError: If p is not a positive integer
        """
        if not isinstance(p, (int, np.integer)):
            raise ValueError(f"p must be an integer, got {type(p)}")
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
            
        self.p = p
        self.data = None
        self.coefficients = None
        
    def fit(self, data: np.ndarray) -> Dict[str, float]:
        """Fit the AR(p) model to the data using OLS estimation.
        
        Args:
            data: Time series data as a 1D numpy array
            
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'ar1', 'ar2', ..., 'arp': Autoregressive coefficients
                - 'sigma2': Variance of the error terms
                
        Raises:
            ValueError: If data is not a numpy array, not 1D, or too short
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(f"data must be a numpy array, got {type(data)}")
        if data.ndim != 1:
            raise ValueError(f"data must be 1-dimensional, got {data.ndim} dimensions")
        if len(data) <= self.p:
            raise ValueError(f"data length ({len(data)}) must be greater than p ({self.p})")
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or infinite values")
            
        self.data = data.copy()
        
        X, y = self._create_lagged_matrix()
        
        coefficients = self._estimate_ols(X, y)
        
        residuals = y - X @ coefficients
        sigma2 = np.var(residuals, ddof=self.p)
        
        self.coefficients = {
            **{f'ar{i+1}': coefficients[i] for i in range(self.p)},
            'sigma2': sigma2
        }
        
        return self.coefficients
    
    def _create_lagged_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Create the lagged design matrix and response vector for OLS estimation.
        
        Returns:
            tuple: (X, y) where,
                - X is the lagged matrix of shape (T-p, p)
                - y is the response vector of shape (T-p,)
        """
        T = len(self.data)
        n_obs = T - self.p
        
        X = np.zeros((n_obs, self.p))
        
        for i in range(self.p):
            X[:, i] = self.data[self.p - i - 1 : T - i - 1]
        
        y = self.data[self.p:]
        
        return X, y
    
    def _estimate_ols(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate OLS coefficients using the normal equations.
        
        Args:
            X: Feature matrix of shape (n, p)
            y: Response vector of shape (n,)
            
        Returns:
            np.ndarray: Estimated coefficients of shape (p,)
            
        Raises:
            np.linalg.LinAlgError: If X'X is singular
        """
        try:
            coefficients = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "X'X is singular. The design matrix may be rank deficient."
            )
        return coefficients
    
    def __repr__(self) -> str:
        """String representation of the AR model."""
        if self.coefficients is None:
            return f"ARModel(p={self.p}, fitted=False)"
        else:
            return f"ARModel(p={self.p}, fitted=True)"
