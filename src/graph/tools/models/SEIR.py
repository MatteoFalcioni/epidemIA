#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:30:19 2026

@author: Giulio Colombini
"""

import numpy  as np
import pandas as pd
from scipy.optimize  import minimize
from scipy.integrate import solve_ivp

# Susceptible - Infectious - Removed and incidence function.

# Model attributes

name = 'SEIR'


PARAMETER_FIELDS = {'beta'     : float,
                    'mu'       : float,
                    'gamma'    : float,
                    'I0'       : float,
                    'E0'       : float,
                    'baseline' : float,
                    'f'        : float,
                    'N'        : int}
                    
                    
                    
fit_parameters_defaults = {'beta'     : 0.8,
                           'mu'       : 0.2,
                           'gamma'    : 0.2,
                           'f' : 1e-2/2,
                            'N' : 4e5,
                           'E0'       : None, # Infer from incidence data
                           'I0'       : None, # Infer from incidence data.
                           'baseline' : None} # Infer from incidence data.

fixed_parameters_defaults = {'f' : 1e-2/2,
                             'N' : 4e5}


EXPLAIN_PARAMETERS = {'beta'     : 'Infectivity rate (average number of people infected by an infectious person per day).',
                      'mu'       : 'Recovery rate (inverse of the average infectious period in days).',
                      'gamma'    : 'Symptoms development rate (inverse of the average incubation period in days).',
                      'I0'       : 'Initial number of infectious people at the start of the simulation.',
                      'E0'       : 'Initial number of exposed people at the start of the simulation.',
                      'baseline' : 'Baseline number of daily infections (accounts for external factors and noise).',
                      'f' : 'Detection fraction (proportion of actual infections that are detected and reported).',
                      'N' : 'Total population size (number of individuals in the population being modeled).'}
                      
                      
# Fitting bounds for the parameters
bounds = ((0.,     np.inf), # beta,
          (0.,     np.inf), # mu,
          (0.,     np.inf), # gamma,
          (0.,     np.inf), # I0
          (0.,     np.inf), # E0
          (0.,     10)) # baseline infections

# Fitting function

def fit_model(incidence_data, metric, 
              fit_parameters_initial_values :dict = fit_parameters_defaults,
              fixed_parameters :dict | None = fixed_parameters_defaults):

    # Use defaults if no parameters are provided

    if fixed_parameters == None:
        fixed_parameters = {}

    # Right-precedence or between dicts keeps the union of keys,
    # and gives right precedence to values, so that we overwrite
    # parameters only if they are provided.

    fixed_parameters = fixed_parameters_defaults | fixed_parameters

    # Extract fixed parameters from dictionary.

    N = fixed_parameters['N'] # Population size
    f = fixed_parameters['f'] # Detection fraction

    # We need to take care of E00, I00 and baseline0 since by default
    # they are inferred.

    fit_parameters_defaults['E0']       = incidence_data[0]/f
    fit_parameters_defaults['I0']       = incidence_data[0]/f
    fit_parameters_defaults['baseline'] = 3.

    if fit_parameters_initial_values == None:
        fit_parameters_initial_values = {}

    # Right-precedence or between dicts keeps the union of keys,
    # and gives right precedence to values, so that we overwrite
    # parameters only if they are provided.
    
    fit_parameters_initial_values=fit_parameters_defaults|fit_parameters_initial_values

    # Extract fit parameters from dictionary.

    beta0 = fit_parameters_initial_values['beta']
    mu0   = fit_parameters_initial_values['mu']
    gamma0= fit_parameters_initial_values['gamma']
    E00   = fit_parameters_initial_values['E0']
    I00   = fit_parameters_initial_values['I0']
    bl0   = fit_parameters_initial_values['baseline']

    # Arrange args in arrays for handling by the minimizer.
    x0         = np.array([beta0, gamma0, mu0, E00, I00, bl0])
    fixed_args = np.array([f, N])
    
    # Perform minimization of metric.
    optimum = minimize(metric, x0 = x0, 
                       args = ((0., len(incidence_data)), 
                               _incidence, fixed_args, incidence_data),
                       bounds = bounds)

    # Unpack result.
    fit_beta, fit_gamma, fit_mu, fit_E0, fit_I0, fit_bl= optimum.x

    # Build dictionary.
    fitted_parameters = {'beta' : fit_beta,
                         'gamma': fit_gamma,
                         'mu'   : fit_mu,
                         'E0'   : fit_E0,
                         'I0'   : fit_I0,
                         'baseline'   : fit_bl,
                         'N'    : N,
                         'f'    : f}

    # Begin construction of return.

    fit_result = {}
    fit_result['success']           = optimum.success
    fit_result['status']            = optimum.status
    fit_result['metric_minimum']    = optimum.fun
    fit_result['fitted_parameters'] = fitted_parameters
    fit_result['fixed_parameters']  = fixed_parameters
    fit_result['gradient']          = optimum.jac
    fit_result['R0']                = fit_beta/fit_mu

    return fit_result

# ODE field function
def seir(t, y, beta, gamma, mu, N):
    return np.array([- beta * y[0] * y[2] / N, 
                       beta * y[0] * y[2] / N - gamma * y[1],
                       gamma* y[1] - mu * y[2],
                       mu * y[2]])

# Incidence function (for private use)
def _incidence(x, t_span, f, N):
    beta, gamma, mu, E0, I0, bl = x # Respectively beta  = infectivity, 
                                    #              gamma = symptoms development rate, 
                                    #              mu    = recovery rate, 
                                    #              E0    = initial exposed people,
                                    #              I0    = initial infective people,
                                    #              bl    = baseline incidence.

    y0 = np.array([N-E0-I0, E0, I0, 0.])

    sol = solve_ivp(seir, t_span = (t_span), y0 = y0,
                    args = (beta, gamma, mu, N), 
                    t_eval = np.arange(*t_span, 1.),
                    method = 'Radau')
    s, e, i, r = sol.y

    incidence = bl + f * gamma * e

    return incidence

# Public interface for the incidence function
def incidence(beg, end, parameters):
    beta = parameters['beta']
    gamma= parameters['gamma']
    mu   = parameters['mu']
    I0   = parameters['I0']
    E0   = parameters['E0']
    bl   = parameters['baseline']
    N    = parameters['N']
    f    = parameters['f']

    pars_vector = np.array([beta, gamma, mu, E0, I0, bl])

    duration = (end - beg).days 

    t_span = (0, duration+1)
    
    sim = _incidence(pars_vector, t_span, f, N)

    dt_index = pd.date_range(beg, end, freq = '1D')

    return sim, dt_index
