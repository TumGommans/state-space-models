"""EM Algorithm for Unrestricted State Space Model."""

import warnings
import numpy as np

from typing import Dict


class UnrestrictedStateSpaceEM:
    """Unrestricted State Space Model estimated via EM Algorithm."""
    
    def __init__(self, y_data: np.ndarray):
        """Initialize the state space model.
        
        Args:
            y_data: Observations, shape (T, 2) where T is number of time periods
        """
        self.y = y_data
        self.T, self.dim = y_data.shape
        
        if self.dim != 2:
            raise ValueError("This implementation is for 2-dimensional data only")
        
        self.F = None
        self.R = None
        self.Q = None
        
        # Filtered and smoothed estimates
        self.xi_filtered = None     
        self.P_filtered = None      
        self.xi_predicted = None    
        self.P_predicted = None     
        self.xi_smoothed = None     
        self.P_smoothed = None      
        self.P_cross = None
        
        self.log_likelihood = None
        
    def initialize_parameters(
        self, 
        F_init: np.ndarray, 
        R_init: np.ndarray, 
        Q_init: np.ndarray, 
        xi_0: np.ndarray, 
        P_0: np.ndarray
    ):
        """
        Set initial parameter values for EM algorithm.
        
        Args:
            F_init: Initial transition matrix (2x2)
            R_init: Initial observation noise covariance (2x2)
            Q_init: Initial state noise covariance (2x2)
            xi_0: Initial state mean (2,)
            P_0: Initial state covariance (2x2)
        """
        self.F = F_init.copy()
        self.R = R_init.copy()
        self.Q = Q_init.copy()
        self.xi_0 = xi_0.copy()
        self.P_0 = P_0.copy()
        
    def kalman_filter(self) -> float:
        """Run Kalman filter with current parameter estimates.
        
        Returns:
            float: Log-likelihood value
        """
        self.xi_filtered = np.zeros((self.T, self.dim))
        self.P_filtered = np.zeros((self.T, self.dim, self.dim))
        self.xi_predicted = np.zeros((self.T, self.dim))
        self.P_predicted = np.zeros((self.T, self.dim, self.dim))
        
        # Initial prediction (t=0 to t=1)
        self.xi_predicted[0] = self.F @ self.xi_0
        self.P_predicted[0] = self.F @ self.P_0 @ self.F.T + self.Q
        
        log_likelihood = 0.0
        
        for t in range(self.T):
            e_t = self.y[t] - self.xi_predicted[t]
            Sigma_t = self.P_predicted[t] + self.R
            
            # Kalman gain
            try:
                Sigma_t_inv = np.linalg.inv(Sigma_t)
            except np.linalg.LinAlgError:
                warnings.warn(f"Sigma_t is singular at t={t}, using pseudo-inverse")
                Sigma_t_inv = np.linalg.pinv(Sigma_t)
            
            K_t = self.P_predicted[t] @ Sigma_t_inv
            
            # Update step
            self.xi_filtered[t] = self.xi_predicted[t] + K_t @ e_t
            self.P_filtered[t] = self.P_predicted[t] - K_t @ Sigma_t @ K_t.T
            
            # Ensure symmetry of covariance matrix
            self.P_filtered[t] = 0.5 * (self.P_filtered[t] + self.P_filtered[t].T)
            
            # Log-likelihood contribution
            sign, logdet = np.linalg.slogdet(2 * np.pi * Sigma_t)
            if sign <= 0:
                warnings.warn(f"Non-positive determinant at t={t}")
                logdet = 0
            log_likelihood += -0.5 * (logdet + e_t.T @ Sigma_t_inv @ e_t)
            
            # Prediction step for next time period
            if t < self.T - 1:
                self.xi_predicted[t+1] = self.F @ self.xi_filtered[t]
                self.P_predicted[t+1] = self.F @ self.P_filtered[t] @ self.F.T + self.Q
        
        self.log_likelihood = log_likelihood

        return log_likelihood
    
    def kalman_smoother(self):
        """Run Kalman smoother with current filtered estimates."""
        self.xi_smoothed = np.zeros((self.T, self.dim))
        self.P_smoothed = np.zeros((self.T, self.dim, self.dim))
        self.P_cross = np.zeros((self.T-1, self.dim, self.dim))
        
        # Initialize with filtered estimates at T
        self.xi_smoothed[-1] = self.xi_filtered[-1]
        self.P_smoothed[-1] = self.P_filtered[-1]
        
        for t in range(self.T-2, -1, -1):
            # Smoother gain
            try:
                P_pred_inv = np.linalg.inv(self.P_predicted[t+1])
            except np.linalg.LinAlgError:
                warnings.warn(f"P_predicted is singular at t={t+1}, using pseudo-inverse")
                P_pred_inv = np.linalg.pinv(self.P_predicted[t+1])
            
            J_t = self.P_filtered[t] @ self.F.T @ P_pred_inv
            
            # Smoothed state
            self.xi_smoothed[t] = (
                self.xi_filtered[t] + 
                J_t @ (self.xi_smoothed[t+1] - 
                self.xi_predicted[t+1])
            )
            
            # Smoothed covariance
            self.P_smoothed[t] = (
                self.P_filtered[t] + 
                J_t @ (self.P_smoothed[t+1] - 
                self.P_predicted[t+1]) @ J_t.T
            )
            
            # Ensure symmetry
            self.P_smoothed[t] = 0.5 * (self.P_smoothed[t] + self.P_smoothed[t].T)
            
            # Cross-covariance term: P_{t+1,t|T}
            self.P_cross[t] = self.P_smoothed[t+1] @ P_pred_inv @ self.F @ self.P_filtered[t]
    
    def m_step(self):
        """M-step: Update parameters based on smoothed estimates."""
        sum_xi_xi = np.zeros((self.dim, self.dim))
        sum_xi_xi_lag = np.zeros((self.dim, self.dim))
        sum_xi_lag_xi_lag = np.zeros((self.dim, self.dim))
        sum_y_xi = np.zeros((self.dim, self.dim))
        
        for t in range(self.T):
            xi_xi_t = np.outer(self.xi_smoothed[t], self.xi_smoothed[t]) + self.P_smoothed[t]
            sum_xi_xi += xi_xi_t
    
            sum_y_xi += np.outer(self.y[t], self.xi_smoothed[t])
            
            if t < self.T - 1:
                sum_xi_xi_lag += (
                    np.outer(self.xi_smoothed[t+1], self.xi_smoothed[t]) + 
                    self.P_cross[t]
                )
            
            if t > 0:
                xi_lag_xi_lag = (
                    np.outer(self.xi_smoothed[t-1], self.xi_smoothed[t-1]) + 
                    self.P_smoothed[t-1]
                )
                sum_xi_lag_xi_lag += xi_lag_xi_lag
        
        # Update F
        try:
            self.F = sum_xi_xi_lag @ np.linalg.inv(sum_xi_lag_xi_lag)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in F update, using pseudo-inverse")
            self.F = sum_xi_xi_lag @ np.linalg.pinv(sum_xi_lag_xi_lag)
        
        # Update R
        self.R = np.zeros((self.dim, self.dim))
        for t in range(self.T):
            y_minus_xi = self.y[t] - self.xi_smoothed[t]
            self.R += (np.outer(y_minus_xi, y_minus_xi) + self.P_smoothed[t])
        self.R /= self.T
        
        # Update Q
        self.Q = np.zeros((self.dim, self.dim))
        for t in range(1, self.T):
            xi_xi_t = np.outer(self.xi_smoothed[t], self.xi_smoothed[t]) + self.P_smoothed[t]
            xi_xi_lag = np.outer(self.xi_smoothed[t], self.xi_smoothed[t-1]) + self.P_cross[t-1]
            xi_lag_xi_lag = (
                np.outer(self.xi_smoothed[t-1], self.xi_smoothed[t-1]) + 
                self.P_smoothed[t-1]
            )
            self.Q += (xi_xi_t - self.F @ xi_xi_lag - xi_xi_lag.T @ self.F.T + 
                      self.F @ xi_lag_xi_lag @ self.F.T)
        self.Q /= self.T
    
    def em_algorithm(
        self, 
        max_iter: int = 200, 
        tol: float = 1e-10, 
        verbose: bool = True
    ) -> Dict:
        """Run the EM algorithm for parameter estimation.
        
        Args:
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance for log-likelihood change
            verbose: If True, print progress information
        
        Returns:
            dict: Dictionary containing estimation results
        """
        if self.F is None:
            raise ValueError("Parameters must be initialized before running EM")
        
        log_likelihoods = []
        
        params_at_20 = None
        params_at_200 = None
        
        for iteration in range(max_iter):
            # E-step: Run Kalman filter and smoother
            log_lik = self.kalman_filter()
            self.kalman_smoother()
            log_likelihoods.append(log_lik)
            
            # Check convergence
            if iteration > 0:
                ll_change = abs(log_likelihoods[-1] - log_likelihoods[-2])
                if ll_change < tol:
                    if verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break
            
            # M-step
            self.m_step()
            
            # Store parameters AFTER M-step
            if iteration + 1 == 20:
                params_at_20 = {
                    'F': self.F.copy(),
                    'R': self.R.copy(),
                    'Q': self.Q.copy(),
                    'log_likelihood': log_lik
                }
                if verbose:
                    print(f"After Iteration 20: Log-likelihood = {log_lik:.6f}")
                    print(f"F = \n{self.F}")
                    print(f"R = \n{self.R}")
                    print(f"Q = \n{self.Q}")
            
            if iteration + 1 == 200:
                params_at_200 = {
                    'F': self.F.copy(),
                    'R': self.R.copy(),
                    'Q': self.Q.copy(),
                    'log_likelihood': log_lik
                }
                if verbose:
                    print(f"After Iteration 200: Log-likelihood = {log_lik:.6f}")
                    print(f"F = \n{self.F}")
                    print(f"R = \n{self.R}")
                    print(f"Q = \n{self.Q}")
        
        results = {
            'F': self.F.copy(),
            'R': self.R.copy(),
            'Q': self.Q.copy(),
            'log_likelihood': self.log_likelihood,
            'log_likelihoods': np.array(log_likelihoods),
            'iterations': len(log_likelihoods),
            'xi_smoothed': self.xi_smoothed.copy(),
            'P_smoothed': self.P_smoothed.copy(),
            'params_at_20': params_at_20,
            'params_at_200': params_at_200
        }
        
        return results


def estimate_continuous_em(
    y_gdp: np.ndarray, 
    y_consumption: np.ndarray
) -> Dict:
    """Estimate unrestricted state space model using EM.
    
    Args:
        y_gdp: De-meaned GDP growth (T=268,)
        y_consumption: De-meaned consumption growth (T=268,)
    
    Returns:
        dict: Dictionary containing results at iterations 20 and 200
    """
    y_data = np.column_stack([y_gdp, y_consumption])
    
    model = UnrestrictedStateSpaceEM(y_data)
    
    F_init = 0.8 * np.eye(2)
    R_init = 2.0 * np.eye(2)
    Q_init = 3.0 * np.eye(2)
    
    xi_0 = np.zeros(2)
    P_0 = 1e6 * np.eye(2)
    
    model.initialize_parameters(F_init, R_init, Q_init, xi_0, P_0)
    
    results = model.em_algorithm(max_iter=1000, verbose=True)
    
    results_summary = {
        'iteration_20': results['params_at_20'],
        'iteration_200': results['params_at_200'],
        'all_log_likelihoods': results['log_likelihoods'],
        'F': results['F'],
        'R': results['R'],
        'Q': results['Q'],
    }

    return results_summary