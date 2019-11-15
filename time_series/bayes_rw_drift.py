# Simulate a random walk with drift
# Use bayesian methods to recover posterior distribution 
# of model parameters
#
# Model:
# y_t | mu, sig2 ~ N(y_{t-1} | mu, sig2)
#             mu ~ N(m, s2)
#           sig2 ~ IG(a, b)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def rinvgamma(shape, scale):
    return 1 / np.random.gamma(shape, 1 / scale)


def bayes_rw_drift(y, nmcmc, nburn, m=0, s=1, a=2, b=1):
    y_diff = np.diff(y)
    y_diff_sum = y_diff.sum()
    y_diff_sq = np.diff(y) ** 2
    y_diff_sq_sum = (y_diff_sq).sum()
    N = y_diff.shape[0]

    state = dict(mu=0, sig2=1)

    def update_mu():
        s2_inv = s ** -2
        sig2_inv = 1 / state['sig2']
        s2_mu = 1 / (s2_inv + N * sig2_inv)
        m_mu = (m * s2_inv + y_diff_sum * sig2_inv) * s2_mu
        state['mu'] = np.random.normal(m_mu, np.sqrt(s2_mu))

    def update_sig2():
        a_sig2 = a + N / 2
        x = y_diff_sq_sum - 2 * state['mu'] * y_diff_sum + N * state['mu'] ** 2
        b_sig2 = b + x / 2
        state['sig2'] = rinvgamma(a_sig2, b_sig2)

    def update():
        update_mu()
        update_sig2()

    out = dict(mu=[], sig2=[])
    for i in trange(nburn + nmcmc):
        update()
        if i >= nburn:
            out['mu'].append(state['mu'])
            out['sig2'].append(state['sig2'])
    
    out['mu'] = np.array(out['mu'])
    out['sig2'] = np.array(out['sig2'])

    return out


def sim_rw_with_drift(n, mu=0, sig2=1):
    err = np.random.randn(n) * np.sqrt(sig2) + mu
    y = err.cumsum()

    return dict(y=y, mu=mu, sig2=sig2)


def summarize(out, key):
    print('{} est: {:.4f} ({:.4f}, {:.4f}) | true: {}'
          .format(key,
                  out[key].mean(),
                  np.quantile(out[key], .025),
                  np.quantile(out[key], .975),
                  simdat[key]))


def forecast(steps, out, y):
    nsamps = out['mu'].shape[0]
    x = np.random.normal(out['mu'][:, None],
                         np.sqrt(out['sig2'])[:, None],
                         (nsamps, steps))
    return y[-1] + x.cumsum(1)


# Simulate data
np.random.seed(0)
ntrain = 400
ntest = 600
nobs = ntrain + ntest
simdat = sim_rw_with_drift(nobs, .1, .3)

# Plot data
plt.plot(simdat['y']); plt.show()

# MCMC
out = bayes_rw_drift(y=simdat['y'][:ntrain], nmcmc=1000, nburn=500)

# Print summary of posterior
summarize(out, 'mu')
summarize(out, 'sig2')

# Make forecast
steps = ntest
preds = forecast(steps, out, simdat['y'][:ntrain])
pred_mean = preds.mean(0)
pred_lower, pred_upper = np.quantile(preds, [.025, .975], axis=0)

# Plot predictions
plt.plot(simdat['y'], label='Data')
plt.plot(ntrain + np.arange(steps), pred_mean, color='red',
         label='Posterior predictive', ls='--')
plt.fill_between(ntrain + np.arange(steps), pred_lower, pred_upper,
                 color="red", alpha=.3, label='95% CI')
plt.axvline(ntrain, label='Train / Test split', color='grey', ls=':')
plt.legend()
plt.show()

