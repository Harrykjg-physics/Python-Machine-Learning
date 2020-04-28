import matplotlib.pyplot as plt
import numpy as np


def gini(p):
    return (1 - p) * p + (1 - (1 - p)) * (1 - p)


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def error(p):
    return 1 - np.max([p, 1 - p])


fig = plt.figure()
ax = plt.subplot(111)

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [0.5 * e if e else None for e in ent]
err = [error(i) for i in x]
gin = [gini(j) for j in x]
for i, lab, ls, c in zip([ent, sc_ent, gin, err],
                         ['Entropy', 'Entropy(scailed)', 'Gini',
                          'Misclassification error'],
                         ['--', '--', '-', '-.'],
                         ['black', 'lightgray', 'red',
                          'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
          fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i = 1)')
plt.ylabel('Impurity Index')
plt.show()

