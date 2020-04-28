from scipy.special import comb
import math
import numpy as np
import matplotlib.pyplot as plt


def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probas = [comb(n_classifier, k) *
              error**k * (1 - error)**(n_classifier - k)
              for k in range(k_start, n_classifier + 1)]
    return sum(probas)


# print(ensemble_error(n_classifier=11, error=0.25))

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error,
                             ) for error in error_range]
plt.plot(error_range, ens_errors, label='Ensemble Errors',
         linewidth=2)
plt.plot(error_range, error_range, label='Base Error',
         linewidth=2)
plt.xlabel('Base Error')
plt.ylabel('Base Error/Ensemble Error')
plt.legend(loc='upper left')
plt.grid()
plt.show()




