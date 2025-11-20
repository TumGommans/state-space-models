"""AR(1) model with noise estimation using Ordinary Least Squares (OLS)."""

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class KalmanFilterResults:
    """Mutable container for Kalman filter results."""
    xi_filtered: np.ndarray
    P_filtered: np.ndarray
    xi_predicted: np.ndarray
    P_predicted: np.ndarray
    log_likelihood: float
    forecast_errors: np.ndarray
    
    @property
    def xi_T_T(self) -> float:
        """Final filtered state estimate."""
        return self.xi_filtered[-1]
    
    @property
    def P_T_T(self) -> float:
        """Final filtered state variance."""
        return self.P_filtered[-1]


class AR1PlusNoiseModel:
    """AR(1) plus noise model."""
    
    def __init__(self):
        """Initialize the AR(1) plus noise model."""
        self.params: Optional[Dict[str, float]] = None
        self.filter_results: Optional[KalmanFilterResults] = None
        self.y: Optional[pd.Series] = None
        self.optimization_result = None
        
    def fit(
        self, 
        y: pd.Series, 
        init_params: Optional[Dict[str, float]] = None,
        method: str = 'L-BFGS-B',
        options: Optional[Dict] = None
    ) -> 'AR1PlusNoiseModel':
        """Estimate model parameters by maximum likelihood using the Kalman filter.
        
        Args:
            y: Observed time series (already demeaned).
            init_params : Initial parameter values. Default: {'F': 0.1, 'Q': 4.0, 'R': 8.0}
            method : Optimization method for scipy.optimize.minimize.
            options : Additional options for the optimizer.
            
        Returns:
            AR1PlusNoiseModel: Fitted model instance.
        """
        self.y = y
        
        if init_params is None:
            init_params = {'F': 0.1, 'Q': 4.0, 'R': 8.0}
        
        init_vector = self._params_to_vector(init_params)
        
        bounds = [(None, None), (1e-8, None), (1e-8, None)]
        
        if options is None:
            options = {'disp': False, 'maxiter': 1000}
        
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=init_vector,
            args=(y,),
            method=method,
            bounds=bounds,
            options=options
        )
        
        self.optimization_result = result
        
        optimal_params = self._vector_to_params(result.x)
        self.params = optimal_params
        
        self.filter_results = self._kalman_filter(
            y=y.values,
            F=optimal_params['F'],
            Q=optimal_params['Q'],
            R=optimal_params['R']
        )
        return self
    
    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to optimization vector.
        
        Args:
            params: Dictionary with keys 'F', 'Q', 'R'.
            
        Returns:
            np.ndarray: [F, Q, R]
        """
        return np.array([
            params['F'],
            params['Q'],
            params['R']
        ])
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert optimization vector to parameter dictionary.
        
        Args:
            vector: [F, Q, R].
            
        Returns:
            Dict[str, float]: Dictionary with keys 'F', 'Q', 'R'
        """
        return {
            'F': vector[0],
            'Q': vector[1],
            'R': vector[2]
        }
    
    def _negative_log_likelihood(self, vector: np.ndarray, y: pd.Series) -> float:
        """Compute negative log-likelihood for optimization.
        
        Args:
            vector: Parameter vector [F, log(Q), log(R)].
            y: Observed time series.
            
        Returns:
            float: Negative log-likelihood value.
        """
        params = self._vector_to_params(vector)
        
        try:
            results = self._kalman_filter(
                y=y.values,
                F=params['F'],
                Q=params['Q'],
                R=params['R']
            )
            return -results.log_likelihood
        except (np.linalg.LinAlgError, ValueError):
            # Return large value if numerical issues occur
            return 1e10
    
    def _kalman_filter(
        self, 
        y: np.ndarray, 
        F: float, 
        Q: float, 
        R: float
    ) -> KalmanFilterResults:
        """Run Kalman filter for AR(1) plus noise model.
    
        Args:
            y: Observed time series (T x 1).
            F: Transition coefficient.
            Q: State innovation variance.
            R: Measurement error variance.
            
        Returns:
            KalmanFilterResults: Container with filtered states, variances, and log-likelihood.
        """
        T = len(y)
        H = 1.0
        
        xi_predicted = np.zeros(T)
        P_predicted = np.zeros(T) 
        xi_filtered = np.zeros(T) 
        P_filtered = np.zeros(T)  
        forecast_errors = np.zeros(T)
        
        log_likelihood = 0.0
        
        P_1_0 = Q / (1 - F**2)
        
        xi_predicted[0] = 0.0
        P_predicted[0] = P_1_0
        
        forecast_error = y[0] - H * xi_predicted[0]
        forecast_variance = H**2 * P_predicted[0] + R
        
        if forecast_variance <= 0:
            raise ValueError("Non-positive forecast variance encountered.")
        
        K = P_predicted[0] * H / forecast_variance
        
        xi_filtered[0] = xi_predicted[0] + K * forecast_error
        P_filtered[0] = P_predicted[0] - K * H * P_predicted[0]
        
        forecast_errors[0] = forecast_error
        
        log_likelihood += -0.5 * (
            np.log(2 * np.pi) + 
            np.log(forecast_variance) + 
            forecast_error**2 / forecast_variance
        )

        for t in range(1, T):
            xi_predicted[t] = F * xi_filtered[t-1]
            P_predicted[t] = F**2 * P_filtered[t-1] + Q
            
            forecast_error = y[t] - H * xi_predicted[t]
            forecast_variance = H**2 * P_predicted[t] + R
            
            if forecast_variance <= 0:
                raise ValueError(f"Non-positive forecast variance at t={t}.")
            
            K = P_predicted[t] * H / forecast_variance
            
            xi_filtered[t] = xi_predicted[t] + K * forecast_error
            P_filtered[t] = P_predicted[t] - K * H * P_predicted[t]
            
            forecast_errors[t] = forecast_error
            
            log_likelihood += -0.5 * (
                np.log(2 * np.pi) + 
                np.log(forecast_variance) + 
                forecast_error**2 / forecast_variance
            )
        
        return KalmanFilterResults(
            xi_filtered=xi_filtered,
            P_filtered=P_filtered,
            xi_predicted=xi_predicted,
            P_predicted=P_predicted,
            log_likelihood=log_likelihood,
            forecast_errors=forecast_errors
        )
    
    def summary(self) -> str:
        """Generate a summary of the fitted model.
        
        Returns:
            str: Summary string with parameter estimates and diagnostics.
        """
        if self.params is None:
            return "Model not yet fitted. Call fit() first."
        
        summary = "Parameter Estimates:\n"
        summary += f"  F (φ):      {self.params['F']:.6f}\n"
        summary += f"  Q (σ²_η):   {self.params['Q']:.6f}\n"
        summary += f"  R (σ²_ε):   {self.params['R']:.6f}\n\n"
        
        if self.filter_results is not None:
            summary += "Final Filtered State:\n"
            summary += f"  ξ̂_{{T|T}}:    {self.filter_results.xi_T_T:.6f}\n"
            summary += f"  P̂_{{T|T}}:    {self.filter_results.P_T_T:.6f}\n\n"
            summary += f"Log-Likelihood: {self.filter_results.log_likelihood:.4f}\n"
        
        if self.optimization_result is not None:
            summary += f"Optimization Success: {self.optimization_result.success}\n"
            summary += f"Number of Iterations: {self.optimization_result.nit}\n"
        
        return summary
    
    def get_filtered_states(self) -> pd.DataFrame:
        """Get filtered state estimates as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with columns: xi_filtered, P_filtered, xi_predicted, 
            P_predicted, forecast_errors.
        """
        if self.filter_results is None:
            raise ValueError("Model not yet fitted. Call fit() first.")
        
        return pd.DataFrame({
            'xi_filtered': self.filter_results.xi_filtered,
            'P_filtered': self.filter_results.P_filtered,
            'xi_predicted': self.filter_results.xi_predicted,
            'P_predicted': self.filter_results.P_predicted,
            'forecast_errors': self.filter_results.forecast_errors
        }, index=self.y.index if self.y is not None else None)
