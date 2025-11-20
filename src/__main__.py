"""Main script to extract all results."""

import pandas as pd
import numpy as np

from src.markov_switching_models.finite_mixture import FiniteMixture
from src.markov_switching_models.markov_switching_mle import MarkovSwitchingMLE
from src.markov_switching_models.markov_switching_em import MarkovSwitchingEM

from src.continuous_state_models.ar_model import ARModel
from src.continuous_state_models.ar_noise_model import AR1PlusNoiseModel
from src.continuous_state_models.factor_model import FactorModel
from src.continuous_state_models.em_estimation import estimate_continuous_em

df = pd.read_csv("/workspace/data/growth_data.csv")

y_gdp = df['y'].iloc[:268]
y_gdp_holdout = df['y'].iloc[268:280]
y_gdp_demeaned = y_gdp - y_gdp.mean()

y_cons = df['c'].iloc[:268]
y_cons_holdout = df['c'].iloc[268:280]
y_cons_demeaned = y_cons - y_cons.mean()

def run_question_1A():
    """Extract results for exercise 1a."""
    print("Question 1A")
    print("-" * 70)    

    model = FiniteMixture()
    model.estimate_mle(data=y_gdp)

    print(model.summary())

def run_question_1B():
    """Extract results for exercise 1b."""
    print("\n\n")
    print("Question 1B")
    print("-" * 70)    

    model = MarkovSwitchingMLE()
    model.estimate_mle(data=y_gdp, start_state_1=1)
    print("Starting in state 1")
    print(model.summary())

    model.estimate_mle(data=y_gdp, start_state_1=0)
    print("Starting in state 2")
    print(model.summary())

    model.estimate_mle(data=y_gdp, use_long_run=True)
    print("Using long-run probabilities")
    print(model.summary())

def run_question_1C():
    """Extract results for exercise 1c."""
    print("\n\n")
    print("Question 1C")
    print("-" * 70)

    model = MarkovSwitchingMLE()
    model.estimate_mle(data=y_gdp, start_state_1=1)

    steady_state_prob_2 = model.compute_steady_state_prob(state=2)
    expected_duration_quarters = model.compute_expected_duration(state=2)
    expected_duration_years = expected_duration_quarters / 4
    forecast_results = model.forecast_one_step_ahead(y_gdp_holdout.iloc[0])

    summary = "Results:\n"
    summary += f"  steady-state probability    : {steady_state_prob_2:.6f}\n"
    summary += f"  expected quarterly duration : {expected_duration_quarters:.6f}\n"
    summary += f"  expected yearly duration    : {expected_duration_years:.6f}\n\n"

    summary += f"  forecast          : {forecast_results['forecast']:.6f}\n"
    summary += f"  forecast error    : {forecast_results['forecast_error']:.6f}\n"

    print(summary)

def run_question_1D():
    """Extract results for exercise 1d."""
    print("\n\n")
    print("Question 1D")
    print("-" * 70)
    
    y_bivariate = np.column_stack([y_gdp.values, y_cons.values])
    
    y_mean = np.mean(y_bivariate, axis=0)
    y_cov = np.cov(y_bivariate.T)
    
    model = MarkovSwitchingEM()
    
    results = model.estimate(
        data=y_bivariate,
        p11_init=0.8,
        p22_init=0.8,
        mu1_init=y_mean,
        mu2_init=y_mean,
        Sigma1_init=y_cov,
        Sigma2_init=0.5 * y_cov,
        max_iter=1000,
        tol=1e-6,
        verbose=False
    )
    print("\n" + model.summary())

def run_question_1():
    """Run all subquestions of exercise 1."""
    run_question_1A()
    run_question_1B()
    run_question_1C()
    run_question_1D()

def run_question_2A():
    """Extract results for exercise 2a."""
    print("\n\n")
    print("Question 2A")
    print("-" * 70)

    model = ARModel(p=1)
    coefficients = model.fit(data=y_gdp_demeaned.to_numpy())

    summary = "Parameter Estimates:\n"
    summary += f"  phi.   :     {coefficients['ar1']:.6f}\n"
    summary += f"  sigma2 :     {coefficients['sigma2']:.6f}\n"
    print(summary)

def run_question_2B():
    """Extract results for exercise 2b."""
    print("\n\n")
    print("Question 2B")
    print("-" * 70)

    model = AR1PlusNoiseModel()
    model.fit(y_gdp_demeaned, init_params={'F': 0.1, 'Q': 4.0, 'R': 8.0})
    print(model.summary())

def run_question_2C():
    """Extract results for exercise 2c."""
    print("\n\n")
    print("Question 2C")
    print("-" * 70)
    
    y1_full = df['y']
    y2_full = df['c']

    y1_mean = y1_full.iloc[:268].mean()
    y2_mean = y2_full.iloc[:268].mean()

    y1_demeaned = y1_full - y1_mean
    y2_demeaned = y2_full - y2_mean

    y1_estimation = y1_demeaned.iloc[:268]
    y2_estimation = y2_demeaned.iloc[:268]

    y1_holdout = y1_demeaned.iloc[268:280]
    y2_holdout = y2_demeaned.iloc[268:280]

    y_holdout = np.column_stack([y1_holdout.values, y2_holdout.values])

    model_2c = FactorModel()
    model_2c.fit(
        y1=y1_estimation,
        y2=y2_estimation,
        init_params={
            'h1': 0.5, 'h2': 0.5,
            'f0': 1.0, 'f1': 1.0, 'f2': 1.0,
            'r1': 1.0, 'r2': 1.0,
            'q1': 1.0, 'q2': 1.0
        }
    )
    forecast_results = model_2c.forecast(y_holdout)
    print(model_2c.summary())

def run_question_2D():
    """Extract results for exercise 2d."""
    print("\n\n")
    print("Question 2D")
    print("-" * 70)

    y1_full = df['y']
    y2_full = df['c']

    y1_mean = y1_full.iloc[:268].mean()
    y2_mean = y2_full.iloc[:268].mean()

    y1_demeaned = y1_full - y1_mean
    y2_demeaned = y2_full - y2_mean

    y1_estimation = y1_demeaned.iloc[:268]
    y2_estimation = y2_demeaned.iloc[:268]

    results = estimate_continuous_em(y1_estimation, y2_estimation)

    print(f"\n    F: {results['F']}")
    print(f"    R: {results['R']}")
    print(f"    Q: {results['Q']}")

def run_question_2():
    """Run all subquestions of exercise 2."""
    run_question_2A()
    run_question_2B()
    run_question_2C()
    run_question_2D()

if __name__ == "__main__":
    run_question_1()
    run_question_2()
