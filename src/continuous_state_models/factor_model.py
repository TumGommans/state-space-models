"""MLE for Factor State Space Model."""

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FactorModelResults:
    """Mutable container for factor model Kalman filter results."""
    xi_filtered: np.ndarray      
    P_filtered: np.ndarray       
    xi_predicted: np.ndarray     
    P_predicted: np.ndarray      
    log_likelihood: float
    forecast_errors: np.ndarray
    
    @property
    def xi_T_T(self) -> np.ndarray:
        """Final filtered state estimate."""
        return self.xi_filtered[-1, :]
    
    @property
    def P_T_T(self) -> np.ndarray:
        """Final filtered state covariance."""
        return self.P_filtered[-1, :, :]


@dataclass
class ForecastResults:
    """Mutable container for out-of-sample forecast results."""
    forecasts: np.ndarray       
    actuals: np.ndarray         
    forecast_errors: np.ndarray 
    msfe_y1: float              
    msfe_y2: float              
    
    def __str__(self):
        return (
            f"MSFE (y1): {self.msfe_y1:.6f}\n"
            f"MSFE (y2): {self.msfe_y2:.6f}"
        )


class FactorModel:
    """Factor model in state space form with one common factor and two idiosyncratic components."""
    
    def __init__(self):
        """Initialize the factor model."""
        self.params: Optional[Dict[str, float]] = None
        self.filter_results: Optional[FactorModelResults] = None
        self.y: Optional[np.ndarray] = None
        self.optimization_result = None
        self.forecast_results: Optional[ForecastResults] = None
        
    def fit(
        self,
        y1: pd.Series,
        y2: pd.Series,
        init_params: Optional[Dict[str, float]] = None,
        method: str = 'L-BFGS-B',
        options: Optional[Dict] = None
    ) -> 'FactorModel':
        """Estimate model parameters by maximum likelihood using the Kalman filter.
        
        Args:
            y1: First observed time series (de-meaned GDP growth).
            y2: Second observed time series (de-meaned consumption growth).
            init_params: Initial parameter values.
            method: Optimization method for scipy.optimize.minimize.
            options: Additional options for the optimizer.
            
        Returns:
            FactorModel: Fitted model instance.
        """
        self.y = np.column_stack([y1.values, y2.values])
        
        if init_params is None:
            init_params = {
                'h1': 0.5, 'h2': 0.5,
                'f0': 1.0, 'f1': 1.0, 'f2': 1.0,
                'r1': 1.0, 'r2': 1.0,
                'q1': 1.0, 'q2': 1.0
            }
        
        init_vector = self._params_to_vector(init_params)
        
        bounds = [
            (None, None),
            (None, None),
            (None, None),            
            (None, None),            
            (None, None),
            (None, None),
            (None, None),            
            (None, None),
            (None, None),
        ]
        
        if options is None:
            options = {'disp': False, 'maxiter': 2000}
        
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=init_vector,
            args=(self.y,),
            method=method,
            bounds=bounds,
            options=options
        )
        
        self.optimization_result = result
        
        optimal_params = self._vector_to_params(result.x)
        self.params = optimal_params
        
        self.filter_results = self._kalman_filter(
            y=self.y,
            **optimal_params
        )
        
        return self
    
    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to optimization vector.
        
        Args:
            params: Dictionary with keys: h1, h2, f0, f1, f2, r1, r2, q1, q2
            
        Returns:
            np.ndarray: [h1, h2, f0, f1, f2, r1, r2, q1, q2]
        """
        return np.array([
            params['h1'],
            params['h2'],
            params['f0'],
            params['f1'],
            params['f2'],
            params['r1'],
            params['r2'],
            params['q1'],
            params['q2']
        ])
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert parameter dictionary to optimization vector.
        
        Args:
            vector: [h1, h2, f0, f1, f2, r1, r2, q1, q2]
            
        Returns:
            dict: Dictionary with keys: h1, h2, f0, f1, f2, r1, r2, q1, q2
        """
        return {
            'h1': vector[0],
            'h2': vector[1],
            'f0': vector[2],
            'f1': vector[3],
            'f2': vector[4],
            'r1': vector[5],
            'r2': vector[6],
            'q1': vector[7],
            'q2': vector[8]
        }
    
    def _negative_log_likelihood(self, vector: np.ndarray, y: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization.
        
        Args:
            vector: Parameter vector.
            y: Observed time series (T x 2).
            
        Returns:
            float: Negative log-likelihood value.
        """
        params = self._vector_to_params(vector)
        
        try:
            results = self._kalman_filter(y=y, **params)
            return -results.log_likelihood
        except (np.linalg.LinAlgError, ValueError):
            # Return large value if numerical issues occur
            return 1e10
    
    def _kalman_filter(
        self,
        y: np.ndarray,
        h1: float,
        h2: float,
        f0: float,
        f1: float,
        f2: float,
        r1: float,
        r2: float,
        q1: float,
        q2: float
    ) -> FactorModelResults:
        """Run Kalman filter for the factor model.
        
        Args:
            y: Observed time series (T x 2).
            h1, h2, f0, f1, f2, r1, r2, q1, q2: Model parameters.
            
        Returns:
            FactorModelResults: Container with filtered states, variances, and log-likelihood.
        """
        T = len(y)
        
        # Define system matrices
        H = np.array(
            [[h1, 1.0, 0.0],
            [h2, 0.0, 1.0]]
        )
        F = np.array(
            [[f0, 0.0, 0.0],
            [0.0, f1, 0.0],
            [0.0, 0.0, f2]]
        )
        R = np.array(
            [[r1, 0.0],
            [0.0, r2]]
        )
        Q = np.array(
            [[1.0, 0.0, 0.0],
            [0.0, q1, 0.0],
            [0.0, 0.0, q2]]
        )
        
        xi_predicted = np.zeros((T, 3))       
        P_predicted = np.zeros((T, 3, 3))     
        xi_filtered = np.zeros((T, 3))        
        P_filtered = np.zeros((T, 3, 3))      
        forecast_errors = np.zeros((T, 2)) 
        
        log_likelihood = 0.0
        
        xi_predicted[0] = np.zeros(3)
        P_predicted[0] = 1e6 * np.eye(3)
        
        for t in range(T):
            # Updating step
            forecast_error = y[t, :] - H @ xi_predicted[t]
            
            forecast_variance = H @ P_predicted[t] @ H.T + R
            
            try:
                forecast_variance_inv = np.linalg.inv(forecast_variance)
            except np.linalg.LinAlgError:
                raise ValueError(f"Singular forecast variance matrix at t={t}.")
            
            K = P_predicted[t] @ H.T @ forecast_variance_inv

            xi_filtered[t] = xi_predicted[t] + K @ forecast_error
            P_filtered[t] = P_predicted[t] - K @ H @ P_predicted[t]
        
            forecast_errors[t] = forecast_error
            
            sign, logdet = np.linalg.slogdet(forecast_variance)
            if sign <= 0:
                raise ValueError(f"Non-positive definite forecast variance at t={t}.")
            
            log_likelihood += -0.5 * (
                2 * np.log(2 * np.pi) + 
                logdet + 
                forecast_error @ forecast_variance_inv @ forecast_error
            )
            
            # Prediction step
            if t < T - 1:
                xi_predicted[t + 1] = F @ xi_filtered[t]
                P_predicted[t + 1] = F @ P_filtered[t] @ F.T + Q
        
        return FactorModelResults(
            xi_filtered=xi_filtered,
            P_filtered=P_filtered,
            xi_predicted=xi_predicted,
            P_predicted=P_predicted,
            log_likelihood=log_likelihood,
            forecast_errors=forecast_errors
        )
    
    def forecast(self, y_holdout: np.ndarray) -> ForecastResults:
        """Generate one-step-ahead out-of-sample forecasts.
        
        This method iteratively applies the Kalman filter prediction and updating steps
        to generate forecasts for the holdout sample. For each period:
        1. Predict ξ_{T+h|T+h-1} using the transition equation
        2. Predict y_{T+h|T+h-1} using the measurement equation
        3. Observe actual y_{T+h}
        4. Update to get ξ_{T+h|T+h}
        5. Roll forward
        
        Args:
            y_holdout: Holdout sample observations (K x 2), where K is the forecast horizon.
            
        Returns:
            ForecastResults: Container with forecasts, actuals, errors, and MSFE.
        """
        if self.params is None or self.filter_results is None:
            raise ValueError("Model must be fitted before forecasting. Call fit() first.")
        
        K = len(y_holdout)
        
        params = self.params
        h1, h2 = params['h1'], params['h2']
        f0, f1, f2 = params['f0'], params['f1'], params['f2']
        r1, r2 = params['r1'], params['r2']
        q1, q2 = params['q1'], params['q2']
        
        H = np.array(
            [[h1, 1.0, 0.0],
            [h2, 0.0, 1.0]]
        )
        F = np.array(
            [[f0, 0.0, 0.0],
            [0.0, f1, 0.0],
            [0.0, 0.0, f2]]
        )
        R = np.array(
            [[r1, 0.0],
            [0.0, r2]]
        )
        Q = np.array(
            [[1.0, 0.0, 0.0],
            [0.0, q1, 0.0],
            [0.0, 0.0, q2]]
        )
        
        # Initialize with final filtered state from estimation
        xi_current = self.filter_results.xi_T_T.copy()
        P_current = self.filter_results.P_T_T.copy()
        
        forecasts = np.zeros((K, 2))
        forecast_errors_oos = np.zeros((K, 2))
        
        for h in range(K):
            # Prediction step:
            xi_predicted = F @ xi_current
            P_predicted = F @ P_current @ F.T + Q
            
            # Forecast:
            y_forecast = H @ xi_predicted
            forecasts[h] = y_forecast
            
            # Forecast error
            forecast_error = y_holdout[h] - y_forecast
            forecast_errors_oos[h] = forecast_error
            
            # Updating step with actual observation
            forecast_variance = H @ P_predicted @ H.T + R
            forecast_variance_inv = np.linalg.inv(forecast_variance)
            
            K_gain = P_predicted @ H.T @ forecast_variance_inv
            xi_current = xi_predicted + K_gain @ forecast_error
            P_current = P_predicted - K_gain @ H @ P_predicted
        
        msfe_y1 = np.mean(forecast_errors_oos[:, 0]**2)
        msfe_y2 = np.mean(forecast_errors_oos[:, 1]**2)
        
        forecast_results = ForecastResults(
            forecasts=forecasts,
            actuals=y_holdout,
            forecast_errors=forecast_errors_oos,
            msfe_y1=msfe_y1,
            msfe_y2=msfe_y2
        )
        self.forecast_results = forecast_results

        return forecast_results
    
    def summary(self) -> str:
        """Generate a summary of the fitted model.
        
        Returns:
            str: Summary string with parameter estimates and diagnostics.
        """
        if self.params is None:
            return "Model not yet fitted. Call fit() first."
        
        summary = "Parameter Estimates:\n"
        summary += f"  h1 (Factor loading 1):     {self.params['h1']:.6f}\n"
        summary += f"  h2 (Factor loading 2):     {self.params['h2']:.6f}\n"
        summary += f"  f0 (Common factor AR):     {self.params['f0']:.6f}\n"
        summary += f"  f1 (Idiosyncratic 1 AR):   {self.params['f1']:.6f}\n"
        summary += f"  f2 (Idiosyncratic 2 AR):   {self.params['f2']:.6f}\n"
        summary += f"  r1 (Measurement var 1):    {self.params['r1']:.6f}\n"
        summary += f"  r2 (Measurement var 2):    {self.params['r2']:.6f}\n"
        summary += f"  q1 (State innovation 1):   {self.params['q1']:.6f}\n"
        summary += f"  q2 (State innovation 2):   {self.params['q2']:.6f}\n\n"
        
        if self.filter_results is not None:
            summary += "Final Filtered State:\n"
            summary += f"  ξ̂₀_{{T|T}} (Common):        {self.filter_results.xi_T_T[0]:.6f}\n"
            summary += f"  ξ̂₁_{{T|T}} (Idiosync 1):    {self.filter_results.xi_T_T[1]:.6f}\n"
            summary += f"  ξ̂₂_{{T|T}} (Idiosync 2):    {self.filter_results.xi_T_T[2]:.6f}\n\n"
            summary += f"Log-Likelihood: {self.filter_results.log_likelihood:.4f}\n"
        
        if self.optimization_result is not None:
            summary += f"Optimization Success: {self.optimization_result.success}\n"
            summary += f"Number of Iterations: {self.optimization_result.nit}\n"
        
        if self.forecast_results is not None:
            summary += "Out-of-Sample Forecast Results:\n"
            summary += f"MSFE (y1 - GDP):          {self.forecast_results.msfe_y1:.6f}\n"
            summary += f"MSFE (y2 - Consumption):  {self.forecast_results.msfe_y2:.6f}\n"
        
        return summary