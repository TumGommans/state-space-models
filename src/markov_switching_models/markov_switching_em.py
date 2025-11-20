"""Estimating a markov-switching model using ME."""

import warnings
import numpy as np

from scipy.stats import multivariate_normal
from typing import Dict, Optional

class MarkovSwitchingEM:
    """Bivariate Markov Switching model estimated using the EM algorithm."""
    
    def __init__(self):
        """Initializing the model."""
        self.n_states = 2
        self.n_dim = 2
        
        self.p11 = None
        self.p22 = None
        self.mu1 = None
        self.mu2 = None
        self.Sigma1 = None
        self.Sigma2 = None
        
        self.P = None
        
        self.xi_filtered = None
        self.xi_smoothed = None
        self.xi_pred = None    
        
        self.P_star = None
        
        self.loglik_history = []

        self.data = None
        self.T = None
        
    def _initialize_parameters(
        self,
        data: np.ndarray,
        p11: float,
        p22: float,
        mu1: np.ndarray,
        mu2: np.ndarray,
        Sigma1: np.ndarray,
        Sigma2: np.ndarray
    ):
        """Initialize parameters for EM algorithm."""
        self.data = data
        self.T = len(data)
        
        self.p11 = p11
        self.p22 = p22
        self.mu1 = mu1.copy()
        self.mu2 = mu2.copy()
        self.Sigma1 = Sigma1.copy()
        self.Sigma2 = Sigma2.copy()
        
        self._update_transition_matrix()
        
    def _update_transition_matrix(self):
        """Update transition matrix from p11 and p22."""
        self.P = np.array([
            [self.p11, 1 - self.p22],
            [1 - self.p11, self.p22]
        ])
        
    def _observation_density(self, y: np.ndarray, state: int) -> float:
        """Compute f(y_t | S_t = state).
        
        Args:
            y: observation vector (2-dimensional)
            state: state index (1 or 2)
            
        Returns:
            float: Density value
        """
        if state == 1:
            mu, Sigma = self.mu1, self.Sigma1
        else:
            mu, Sigma = self.mu2, self.Sigma2
            
        try:
            density = multivariate_normal.pdf(y, mean=mu, cov=Sigma)
        except:
            density = 1e-300
            
        return max(density, 1e-300)
    
    def _hamilton_filter(self) -> float:
        """Run Hamilton filter (forward pass).
        
        Returns:
            Log-likelihood value
        """
        T = self.T
        
        self.xi_filtered = np.zeros((T, self.n_states))
        self.xi_pred = np.zeros((T, self.n_states))
        
        steady_state = np.array([
            (1 - self.p22) / (2 - self.p11 - self.p22),
            (1 - self.p11) / (2 - self.p11 - self.p22)
        ])
        self.xi_pred[0] = steady_state
        
        loglik = 0.0
        
        for t in range(T):
            f1 = self._observation_density(self.data[t], state=1)
            f2 = self._observation_density(self.data[t], state=2)
            f_vec = np.array([f1, f2])
            
            # Update step:
            numerator = f_vec * self.xi_pred[t]
            denominator = np.sum(numerator)
            
            if denominator < 1e-300:
                warnings.warn(f"Numerical instability at t={t}")
                denominator = 1e-300
                
            self.xi_filtered[t] = numerator / denominator
            
            # Add to log-likelihood
            loglik += np.log(denominator)
            
            # Prediction step:
            if t < T - 1:
                self.xi_pred[t + 1] = self.P @ self.xi_filtered[t]
                
        return loglik
    
    def _kim_smoother(self):
        """Run Kim smoother to compute smoothed probabilities.
        
        Computes:
        - smoothed state probabilities
        - joint probabilities P(St=i, St-1=j | IT)
        """
        T = self.T
        
        self.xi_smoothed = np.zeros((T, self.n_states))
        self.P_star = np.zeros((T, self.n_states, self.n_states))
        
        self.xi_smoothed[T - 1] = self.xi_filtered[T - 1]
        
        # Backward recursion
        for t in range(T - 2, -1, -1):
            xi_pred_safe = np.maximum(self.xi_pred[t + 1], 1e-300)
            
            self.xi_smoothed[t] = self.xi_filtered[t] * (
                self.P.T @ (self.xi_smoothed[t + 1] / xi_pred_safe)
            )
            
        for t in range(1, T):
            outer_prod = np.outer(self.xi_smoothed[t], self.xi_filtered[t - 1])
            xi_pred_expanded = np.outer(self.xi_pred[t], np.ones(self.n_states))
            
            self.P_star[t] = (self.P * outer_prod) / np.maximum(xi_pred_expanded, 1e-300)
            
        self.P_star[0] = np.outer(self.xi_smoothed[0], np.ones(self.n_states))
        
    def _e_step(self) -> float:
        """E-step: Run filter and smoother to compute expected sufficient statistics.
        
        Returns:
            Log-likelihood value
        """
        loglik = self._hamilton_filter()
        self._kim_smoother()
        return loglik
    
    def _m_step(self):
        """M-step: Update parameters given smoothed probabilities."""
        T = self.T
        
        sum_P11 = np.sum([self.P_star[t][0, 0] for t in range(1, T)])
        sum_P22 = np.sum([self.P_star[t][1, 1] for t in range(1, T)])
        
        sum_p1_lagged = np.sum([self.xi_smoothed[t - 1][0] for t in range(1, T)])
        sum_p2_lagged = np.sum([self.xi_smoothed[t - 1][1] for t in range(1, T)])
        
        self.p11 = sum_P11 / max(sum_p1_lagged, 1e-300)
        self.p22 = sum_P22 / max(sum_p2_lagged, 1e-300)
        
        self.p11 = np.clip(self.p11, 0.01, 0.99)
        self.p22 = np.clip(self.p22, 0.01, 0.99)
        
        self._update_transition_matrix()
                
        weights1 = self.xi_smoothed[:, 0]
        weights2 = self.xi_smoothed[:, 1]
        
        sum_weights1 = np.sum(weights1)
        sum_weights2 = np.sum(weights2)
        
        self.mu1 = np.sum(weights1[:, np.newaxis] * self.data, axis=0) / max(sum_weights1, 1e-300)
        self.mu2 = np.sum(weights2[:, np.newaxis] * self.data, axis=0) / max(sum_weights2, 1e-300)
        
        self.Sigma1 = np.zeros((self.n_dim, self.n_dim))
        self.Sigma2 = np.zeros((self.n_dim, self.n_dim))
        
        for t in range(T):
            resid1 = self.data[t] - self.mu1
            resid2 = self.data[t] - self.mu2
            
            self.Sigma1 += weights1[t] * np.outer(resid1, resid1)
            self.Sigma2 += weights2[t] * np.outer(resid2, resid2)
            
        self.Sigma1 /= max(sum_weights1, 1e-300)
        self.Sigma2 /= max(sum_weights2, 1e-300)
        
        self.Sigma1 = self._ensure_positive_definite(self.Sigma1)
        self.Sigma2 = self._ensure_positive_definite(self.Sigma2)
        
    def _ensure_positive_definite(
        self, 
        Sigma: np.ndarray, 
        min_eig: float = 1e-6
    ) -> np.ndarray:
        """Ensure covariance matrix is positive definite."""
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, min_eig)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def estimate(
        self,
        data: np.ndarray,
        p11_init: float = 0.8,
        p22_init: float = 0.8,
        mu1_init: Optional[np.ndarray] = None,
        mu2_init: Optional[np.ndarray] = None,
        Sigma1_init: Optional[np.ndarray] = None,
        Sigma2_init: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tol: float = 1e-10,
        verbose: bool = True
    ) -> Dict:
        """Estimate model parameters using EM algorithm.
        
        Args:
            data: (T x 2) array of observations
            p11_init: initial value for P(S_t=1 | S_{t-1}=1)
            p22_init: initial value for P(S_t=2 | S_{t-1}=2)
            mu1_init: initial mean for state 1 (if None, use data mean)
            mu2_init: initial mean for state 2 (if None, use data mean)
            Sigma1_init: initial covariance for state 1 (if None, use data covariance)
            Sigma2_init: initial covariance for state 2 (if None, use scaled data covariance)
            max_iter: maximum number of EM iterations
            tol: convergence tolerance for log-likelihood
            verbose: whether to print progress
            
        Returns:
            Dictionary with estimation results
        """
        data = np.asarray(data)
        if data.ndim == 1:
            raise ValueError("Data must be 2-dimensional (T x 2)")
        if data.shape[1] != 2:
            raise ValueError(f"Data must have 2 columns, got {data.shape[1]}")
            
        if mu1_init is None:
            mu1_init = np.mean(data, axis=0)
        if mu2_init is None:
            mu2_init = np.mean(data, axis=0)
        if Sigma1_init is None:
            Sigma1_init = np.cov(data.T)
        if Sigma2_init is None:
            Sigma2_init = 0.5 * np.cov(data.T)
            
        self._initialize_parameters(
            data=data,
            p11=p11_init,
            p22=p22_init,
            mu1=mu1_init,
            mu2=mu2_init,
            Sigma1=Sigma1_init,
            Sigma2=Sigma2_init
        )
        
        self.loglik_history = []
        
        for iteration in range(max_iter):
            # E-step
            loglik = self._e_step()
            self.loglik_history.append(loglik)
            
            if verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"Iteration {iteration:3d}: Log-likelihood = {loglik:.6f}")
                
            # Check convergence
            if iteration > 0:
                loglik_change = loglik - self.loglik_history[-2]
                if abs(loglik_change) < tol:
                    if verbose:
                        print(f"\nConverged after {iteration + 1} iterations")
                    break
                    
            # M-step
            self._m_step()
        else:
            if verbose:
                print(f"\nReached maximum iterations ({max_iter})")
                
        final_loglik = self._e_step()
        
        return {
            'loglik': final_loglik,
            'n_iterations': len(self.loglik_history),
            'p11': self.p11,
            'p22': self.p22,
            'mu1': self.mu1,
            'mu2': self.mu2,
            'Sigma1': self.Sigma1,
            'Sigma2': self.Sigma2,
            'xi_smoothed': self.xi_smoothed
        }
    
    def summary(self) -> str:
        """Generate summary of estimation results."""
        if self.p11 is None:
            return "Model not yet estimated."
        
        summary = "Transition Probabilities:\n"
        summary += f"  p11 = {self.p11:.6f}\n"
        summary += f"  p22 = {self.p22:.6f}\n\n"
        
        summary += "State 1 Parameters:\n"
        summary += f"  μ1 = [{self.mu1[0]:.6f}, {self.mu1[1]:.6f}]\n"
        summary += f"  Σ1 = \n{self.Sigma1}\n\n"
        
        summary += "State 2 Parameters:\n"
        summary += f"  μ2 = [{self.mu2[0]:.6f}, {self.mu2[1]:.6f}]\n"
        summary += f"  Σ2 = \n{self.Sigma2}\n\n"
        
        if self.loglik_history:
            summary += f"Final Log-Likelihood: {self.loglik_history[-1]:.6f}\n"
            summary += f"Number of Iterations: {len(self.loglik_history)}\n\n"
            
        return summary
