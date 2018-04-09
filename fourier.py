#!/usr/bin/python
# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import pylab as pl
import pdb
from numpy import fft

FRAQHA =  10# 5-> 1/5 frequencies

# extrapoFourier extends a certain sequence using the first 20 % lowest frequency harmonics

def extrapoFourier(x, n_extend,cutoff=0.2):
    #x = x[-12:]
    n = x.size
    # Fraction of harmonics (1/5 = 20 % of low freqs)
    fraqha = int(1/cutoff)
    # Number of discrete frequencies to consider
    n_ha = n / fraqha
    # original time range
    t = np.arange(0, n)
    # linear regression
    p = np.polyfit(t, x, 1)
    # removing linear trend
    x_nolin = x - p[0] * t
    # fft    ....
    x_freq = fft.fft(x_nolin)
    # equivalent freq values
    f = fft.fftfreq(n)
    # orders equivalent frequencies
    indexes = range(n)
    indexes.sort(key=lambda i: np.absolute(f[i]))
    # extends time range
    t = np.arange(0, n+n_extend)
    # initializes final sequence
    extrap = np.zeros(t.size)
    # adds the relevant harmonics, we take one harmonic every two because the cosine incorporates the two complex frequencies
    for i in indexes[:1 + n_ha * 2]:
        ampli = np.absolute(x_freq[i]) / n
        phase = np.angle(x_freq[i])
        extrap += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    # adds linear trend
    fullextrap = extrap + p[0] * t
    # filter for fixing value at t=0
    #filt = np.exp(-abs(t * 1.0 / fraqha))
    #constant = np.ones(t.size) * (x[0] - fullextrap[n_extend])
    #dif = np.multiply(constant, filt)
    # returns modified interpolated sequence
    return fullextrap #+ dif


# fill_nan interpolates nan values using one dimensional linear interpolation

def fill_nan(A):
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    #pdb.set_trace()
    f = np.interp(inds, inds[good], A[good])
    return f


def main():
    # test sequence
    x = np.array([3,4,5,1,2,4,2,5,2,4,np.nan,13,5,6,12,10,6,12,2,3,4,np.nan,1,2,3,4,10,2,1,4,2])
    # removes nans
    x = fill_nan(x)
    # extends 20 %
    n_extend = x.size / 2
    # extends positive times
    t = np.arange(0,x.size+n_extend)
    # extrapolates
    extrapolation = extrapoFourier(x, n_extend)
    # shows the plot
    pl.plot(t, extrapolation, 'r')
    pl.plot(np.arange(0, x.size), x, 'b')
    pl.show()


if __name__ == '__main__':
    main()

