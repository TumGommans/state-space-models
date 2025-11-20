"""Estimating a finite mixture model using MLE."""

import pandas as pd
import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize

class FiniteMixture:
    """Finite mixture model with 2 latent states/regimes."""

    def __init__(self):
        """Initialize the model."""
        self.p1_init = 0.5
        self.estimates = None
        self.ll_value = None
    
    def estimate_mle(self, data: pd.Series):
        """Estimate model parameters using MLE.
        
        Args:
            data: the data to estimate parameters for
        """
        mean_y = data.mean()
        std_y = data.std()

        mean_y_init = mean_y
        sigma1_init = 0.5 * std_y
        sigma2_init = std_y

        params_init = [self.p1_init, mean_y_init, sigma1_init, sigma2_init]

        estimates = minimize(
            self.log_likelihood, 
            params_init, 
            args=(data,),
            bounds=[(0.001,0.999),(None, None), (1e-6, None), (1e-6,None)]
        )

        self.estimates = estimates.x
        self.ll_value = -estimates.fun

    @staticmethod
    def log_likelihood(params: tuple, data: pd.Series) -> float:
        """Compute the negative log-likelihood value for optimization.
        
        Args:
            params: tuple containing all model parameters.
            data: data to evaluate likelihood at.

        Returns:
            float: the NLL value
        """
        p1, mu, sigma1, sigma2 = params
        ll = np.log(p1 * norm.pdf(data, mu, sigma1) + (1 - p1) * norm.pdf(data, mu, sigma2))

        return -np.sum(ll)

    def summary(self) -> str:
        """Generate a summary of the fitted model.
        
        Returns:
            str: Summary string with parameter estimates and diagnostics.
        """
        if self.estimates is None:
            return "Model not yet fitted. Call estimate_mle() first."
        
        p1_est, mu_est, sigma1_est, sigma2_est = self.estimates
        summary = "Parameter Estimates:\n"
        summary += f"  p1    :   {p1_est:.6f}\n"
        summary += f"  mu    :   {mu_est:.6f}\n"
        summary += f"  sigma1:   {sigma1_est**2:.6f}\n"
        summary += f"  sigma2:   {sigma2_est**2:.6f}\n\n"
        
        summary += f"Log-Likelihood: {self.ll_value:.4f}\n"
        summary += f"Posterior prob: {self.posterior_probability:.4f}\n"
        
        return summary

    @property
    def posterior_probability(self):
        """Posterior probability of being in state 2 given y=0."""
        numerator = (1 - self.estimates[0]) * norm.pdf(0, self.estimates[1], self.estimates[3])
        denominator = self.estimates[0] * norm.pdf(0, self.estimates[1], self.estimates[2]) \
            + (1 - self.estimates[0]) * norm.pdf(0, self.estimates[1], self.estimates[3])
        
        return numerator / denominator
