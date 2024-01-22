import numpy as np
import matplotlib.pyplot as plt

class constants(object):
    def __init__(self, config={}) -> None:
        self.config = config
    
    pulp = {"costs": np.array([5,2,7]),
            "price": np.array([9, 6, 8]),
            
            }
    
    gibbs = {"num_samples":10000,
                "mu_x":0,
                "mu_y":0,
                "sigma_x":1,
                "sigma_y":1,
                "rho":0.5
                }

def gibbs_sampler(num_samples, mu_x, mu_y, sigma_x, sigma_y, rho):
    # Initialize x and y to random values
    x = np.random.normal(mu_x, sigma_x)
    y = np.random.normal(mu_y, sigma_y)
    # Initialize arrays to store samples
    samples_x = np.zeros(num_samples)
    samples_y = np.zeros(num_samples)
    # Run Gibbs sampler
    for i in range(num_samples):
        # Sample from P(x|y)
        x = np.random.normal(mu_x + rho * (sigma_x / sigma_y) * (y - mu_y), np.sqrt((1 - rho ** 2) * sigma_x ** 2))
        # Sample from P(y|x)
        y = np.random.normal(mu_y + rho * (sigma_y / sigma_x) * (x - mu_x), np.sqrt((1 - rho ** 2) * sigma_y ** 2))
        # Store samples
        samples_x[i] = x
        samples_y[i] = y
    return samples_x, samples_y

# def run_gibbs(num_samples, mu_x, mu_y, sigma_x, sigma_y, rho):
#     samples_x, samples_y = gibbs_sampler(num_samples, mu_x, mu_y, sigma_x, sigma_y, rho)

# # Plot the samples
# plt.scatter(samples_x, samples_y, s=5)
# plt.title('Gibbs Sampling of Bivariate Normal Distribution')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()