"""Estimating a markov-switching model using MLE."""

import pandas as pd
import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize

class MarkovSwitchingMLE:
    """Markov Switching Model with Maximum Likelihood Estimation."""

    def __init__(self):
        """Initialize the model."""
        self.p11_init = 0.5
        self.p22_init = 0.5
        self.estimates = None
        self.ll_value = None
        self.filtered_probs = None
    
    def estimate_mle(
        self, 
        data: pd.Series, 
        start_state_1: int = 1, 
        use_long_run: bool = False
    ):
        """Estimate model parameters via maximum likelihood.
        
        Args:
            data: Time series data to estimate
            start_state_1: Initial probability mass in state 1 (0 or 1), by default 1
            use_long_run: Whether to use long-run probabilities for initialization
        """
        mean_y = data.mean()
        std_y = data.std()

        mean_y_init = mean_y
        sigma1_init = 0.5 * std_y
        sigma2_init = std_y

        params_init = [self.p11_init, self.p22_init, mean_y_init, sigma1_init, sigma2_init]

        if not use_long_run:
            args = (data, start_state_1, 1 - start_state_1)
        else:
            args = (
                data, 
                (1 - self.p22_init) / (2 - self.p11_init - self.p22_init),
                (1 - self.p11_init) / (2 - self.p11_init - self.p22_init),
            )

        results = minimize(
            self.hamilton_loglikelihood, 
            params_init,
            args=args,
            bounds=[
                (0.0001, 0.9999), 
                (0.0001, 0.9999), 
                (None, None), 
                (1e-6, None), 
                (1e-6, None)
            ]
        )

        self.estimates = results.x
        self.ll_value = -results.fun
        
        self._store_filtered_probs(data, args[1], args[2])

    def _store_filtered_probs(
        self, 
        data: pd.Series, 
        state1_init: float, 
        state2_init: float
    ):
        """Store filtered probabilities from the Hamilton filter."""
        p11, p22, mu, sigma1, sigma2 = self.estimates
        T = len(data)
        
        state = np.array([state1_init, state2_init])
        P = np.array([[p11, 1 - p11], [1 - p22, p22]])
        
        filtered_probs = np.zeros((T, 2))
        prior = P @ state
        
        for t in range(T):
            
            f1 = norm.pdf(data.iloc[t], mu, sigma1)
            f2 = norm.pdf(data.iloc[t], mu, sigma2)
            pdfs = np.array([f1, f2], dtype=float)
            
            pred_density = prior.dot(pdfs)
            
            state = (pdfs * prior) / (pred_density + 1e-6)
            state /= np.sum(state + 1e-6)
            
            filtered_probs[t] = state
            prior = P @ state
        
        self.filtered_probs = filtered_probs

    @staticmethod
    def hamilton_loglikelihood(
        params: tuple, 
        data: pd.Series, 
        state1_init: float, 
        state2_init: float
    ) -> float:
        """Compute the NLL based on the hamilton filter outputs.
        
        Args:
            params: The model parameters
            data: The data to evaluate the likelihood at
            state1_init: Initial probability of being in state 1
            state2_init: Initial probability of being in state 2
        
        Returns:
            float: the NLL value
        """
        p11, p22, mu, sigma1, sigma2 = params
        T = len(data)

        P = np.array([[p11, 1 - p11], [1 - p22, p22]])
        
        xi_filtered_prev = np.array([state1_init, state2_init])
        xi_predicted = P @ xi_filtered_prev
        
        loglikelihood = 0
        
        for t in range(T):

            f1 = norm.pdf(data.iloc[t], mu, sigma1)
            f2 = norm.pdf(data.iloc[t], mu, sigma2)
            pdfs = np.array([f1, f2], dtype=float)
            
            pred_density = xi_predicted.dot(pdfs)
            pred_density = max(pred_density, 1e-10) 
            
            loglikelihood += np.log(pred_density)
            
            xi_filtered_curr = (pdfs * xi_predicted) / pred_density
            xi_predicted = P @ xi_filtered_curr

        return -loglikelihood

    def summary(self) -> str:
        """Generate a summary of the fitted model.
        
        Returns:
            str: Summary with parameter estimates and diagnostics
        """
        if self.estimates is None:
            return "Model not yet fitted. Call estimate_mle() first."
        
        summary = "Parameter Estimates:\n"
        summary += f"  p11   :   {self.estimates[0]:.6f}\n"
        summary += f"  p22   :   {self.estimates[1]:.6f}\n"
        summary += f"  mu    :   {self.estimates[2]:.6f}\n"
        summary += f"  sigma1:   {self.estimates[3]:.6f}\n"
        summary += f"  sigma2:   {self.estimates[4]:.6f}\n\n"
        
        summary += f"Log-Likelihood: {self.ll_value:.4f}\n"
        
        return summary
    
    def compute_steady_state_prob(self, state: int = 2) -> float:
        """Compute steady-state probability of being in a given state.

        Args:
            state: State number (1 or 2), by default 2
            
        Returns:
            float: Steady-state probability
        """
        if self.estimates is None:
            raise ValueError("Model not yet fitted. Call estimate_mle() first.")
        
        p11, p22 = self.estimates[0], self.estimates[1]
        
        if state == 1:
            return (1 - p22) / (2 - p11 - p22)
        elif state == 2:
            return (1 - p11) / (2 - p11 - p22)
        else:
            raise ValueError("State must be 1 or 2")
    
    def compute_expected_duration(self, state: int = 2) -> float:
        """Compute expected duration in a given state (in quarters).
        
        Args:
            state: State number (1 or 2), by default 2
            
        Returns:
            float: Expected duration in quarters
        """
        if self.estimates is None:
            raise ValueError("Model not yet fitted. Call estimate_mle() first.")
        
        p11, p22 = self.estimates[0], self.estimates[1]
        
        if state == 1:
            return 1 / (1 - p11)
        elif state == 2:
            return 1 / (1 - p22)
        else:
            raise ValueError("State must be 1 or 2")
    
    def forecast_one_step_ahead(self, y_new: float) -> dict:
        """Generate one-step ahead forecast and forecast error.
        
        Args:
            y_new: Actual value of the first out-of-sample observation
            
        Returns:
            dict: Dictionary containing:
            - 'forecast': one-step ahead forecast
            - 'forecast_error': the forecast error
            - 'predicted_state_probs': predicted state probabilities
            - 'filtered_state_T': last filtered state from estimation
        """
        if self.estimates is None:
            raise ValueError("Model not yet fitted. Call estimate_mle() first.")
        
        if self.filtered_probs is None:
            raise ValueError("Filtered probabilities not available.")
        
        p11, p22, mu, _, _ = self.estimates
        
        state_T_given_T = self.filtered_probs[-1].copy()
        
        P = np.array([[p11, 1 - p11], [1 - p22, p22]])
        state_T1_given_T = P @ state_T_given_T
    
        means = np.array([mu, mu])
        forecast = np.dot(means, state_T1_given_T)

        forecast_error = y_new - forecast
        
        return {
            'forecast': forecast,
            'forecast_error': forecast_error,
            'predicted_state_probs': state_T1_given_T,
            'filtered_state_T': state_T_given_T
        }
