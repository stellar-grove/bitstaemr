import numpy as np
import scipy.stats as stats

def createPoissonData(mu, n):
    return stats.poisson.rvs(mu=mu,size=n)

def createUniformData(begin,width,n):
    return stats.uniform.rvs(loc=begin,width=width,size=n)

def createNormalData(mean,std,size):
    return stats.norm.rvs(mean,std,size)

def createGammaData(alpha, size):
    return stats.gamma.rvs(alpha,size)

# This requires that you pass the scale as equal to 1/lambda
def createExponentialData(scale,location,size):
    return stats.expon.rvs(scale=(scale),loc=location,size=size)

def createBinomialData(n, p, size):
    return stats.binom.rvs(n,p,size)

def createBernoulliData(p,size):
    return stats.bernoulli.rvs(p = p,size = size)
    