# OMSCS_notes
Notes gathered from other students

```python
s = mean(daily_returns - daily_rf_returns)/std(daily_returns - daily_rf_returns)

SR= sqrt(252)* s
```

market_cap = n_shares * price

CAPM

$r_i(t) = \beta_i r_m(t) + \alpha_i(t)$

CAPM for portfolios leverages the same equation but for a variety of stocks:

$r_p(t) = \sum_{i} w_i (\beta_i r_m(t) + \alpha_i(t))$

Recall that passive investing CAPM would be different than an active investing CAPM:

- Passive CAPM: $r_p(t) = \beta_p r_m(t) + \alpha_p(t)$
- Active CAPM: $r_p(t) = \beta_p r_m(t) +  \sum_{i} w_i \alpha_i(t)$

 The implications of CAPM are:

- Expected value of $\alpha$ is almost always zero
- The only way to beat the market is by choosing $\beta$
- Choose high $\beta$ in up markets
- Choose low $\beta$ in down markets
