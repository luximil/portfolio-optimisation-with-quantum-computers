"""
The code in this files was taken from the original authors of the paper in
which the constant correlation model was presented: Ledoit and Wolf 2003.
Minor changes have been implemented to adapt it to our application.
Source: https://github.com/pald22/covShrinkage/blob/main/covCor.py
"""

import numpy as np
import numpy.matlib as mt
import pandas as pd
import math


def LedoitWolfCovShrink(Y, k=None):

    # Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    # Post-Condition: Sigmahat dataframe is returned

    # de-mean returns if required
    N, p = Y.shape                      # sample size and matrix dimension

    # default setting
    if k is None or math.isnan(k):

        mean = Y.mean(axis=0)
        # Demean
        Y = Y.sub(mean, axis=1)
        k = 1

    # Vars
    # Adjust effective sample size
    n = N-k

    # Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n

    # compute shrinkage target
    samplevar = np.diag(sample.to_numpy())
    sqrtvar = pd.DataFrame(np.sqrt(samplevar))
    # Mean correlation.
    rBar = (np.sum(np.sum(sample.to_numpy()
                          / np.matmul(sqrtvar.to_numpy(),
                                      sqrtvar.T.to_numpy()))) - p) / (p*(p-1))
    target = pd.DataFrame(rBar*np.matmul(sqrtvar.to_numpy(), sqrtvar.T.to_numpy()))
    target[np.logical_and(np.eye(p), np.eye(p))] = sample[np.logical_and(np.eye(p), np.eye(p))]

    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(), Y.to_numpy()))
    sample2 = pd.DataFrame(np.matmul(Y2.T.to_numpy(), Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat = pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(), sample.to_numpy()))
    pihat = sum(piMat.sum())

    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target, ord='fro')**2

    # diagonal part of the parameter that we call rho
    rho_diag = np.sum(np.diag(piMat))

    # off-diagonal part of the parameter that we call rho
    term1 = pd.DataFrame(np.matmul((Y**3).T.to_numpy(), Y.to_numpy())/n)
    term2 = pd.DataFrame(np.transpose(mt.repmat(samplevar, p, 1))*sample)
    thetaMat = term1-term2
    thetaMat[np.logical_and(np.eye(p), np.eye(p))] = pd.DataFrame(np.zeros((p, p)))[np.logical_and(np.eye(p), np.eye(p))]
    rho_off = rBar*(np.matmul((1/sqrtvar).to_numpy(), sqrtvar.T.to_numpy())*thetaMat).sum().sum()

    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0, min(1, kappahat/n))

    # Compute shrinkage estimator
    sigmahat = shrinkage*target + (1-shrinkage) * sample
    sigmahat.index = Y.columns
    sigmahat.columns = Y.columns

    return sigmahat
