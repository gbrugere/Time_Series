# Linear Time Series
ARIMA modelling of Industrial Production Index (IPI) for wine production

# Part I : the data

## What does the chosen series represent ?

For this assignement, we chose the IPI series for wine production on the
INSEE website. Following the assigment, our series is corrected from
seasonal variations and working days. The series is monthly, with 399
values from January 1990 up to March 2023.

IPI series for wine production           |  Decomposition of additive time series
:-------------------------:|:-------------------------:
![](https://github.com/gbrugere/Time_Series/blob/main/image/plot(x).png)  |  ![](https://github.com/gbrugere/Time_Series/blob/main/image/plot(decompose(x))2.png)

We are working under the assumption that the series is already corrected
for seasonal variation and will not take this phenomenon into account.

However, the series present a mainly growing trend, with drops
corresponding to 2001, 2008 (subprime crisis) and 2020 (covid crisis).

Theses two remarks can be verified by checking the ACF and PACF plots
for the data.
ACF          |  PACF
:-------------------------:|:-------------------------:
![image](https://github.com/gbrugere/Time_Series/blob/main/image/acf(x).png) | ![image](https://github.com/gbrugere/Time_Series/blob/main/image/pacf(x).png)

The PACF does not show periodicity, indicating that the series is not
seasonal.

The ACF is showing a slowly decaying autocorrelation, hinting at the
growing trend indicated earlier.

## Transform the series to make it stationary if necessary

In order to correct the trend of the series, we differentiate it at the
first order. We now have $xstat_t = x_{t+1} - x_t$ where $x$ is the
initial series and $xstat$ the corrected, stationary series.

In order to test this series for stationnary, we compute the Augmented
Dickey-Fuller Test (ADF) and the Kwiatkowski-Phillips-Schmidt-Shin Test
(KPSS).\

ADF          |  KPSS
:-------------------------:|:-------------------------:
![image](https://github.com/gbrugere/Time_Series/blob/main/image/adf(x).png) | ![image](https://github.com/gbrugere/Time_Series/blob/main/image/kpss(x).png)
![image](https://github.com/gbrugere/Time_Series/blob/main/image/adf(xstat).png) | ![image](https://github.com/gbrugere/Time_Series/blob/main/image/kpss(xstat).png)

As we can see,the ADF test, which tests for the presence of unit root
against stationarity, is 0.05 for $x$, has a p-value of which makes it
hard to reject the null hypothesis, and the KPSS test, testing for
stationarity has a p-value of 0.01 for $x$, which makes us want to
reject the null hypothesis.

For our differenciated series, the ADF test has a low p-value of 0.01,
so we want to reject the null hypothesis in favour of stationarity and
the KPSS test has a high p-value of 0.1 for which we are not likely to
reject the null hypothesis of stationarity.

## Graphically represent the chosen series before and after transforming it

The left plot is the initial series $x$ and the right plot is the
differenciated series $xstat$. We can notice that the differenciated
series seems intuitively more stationary than the initial one. We may
also notice that, due to the differenciation, the mean of $xstat$ is
equal to $0$.

Plot of $x$        | Plot of $xstat$
:-------------------------:|:-------------------------:
![image](https://github.com/gbrugere/Time_Series/blob/main/image/plot(x).png) | ![image](https://github.com/gbrugere/Time_Series/blob/main/image/plot(xstat).png)

# Part II: : ARMA models

## Pick (and justify your choice) an ARMA(p,q) model for your corrected time series Xt. Estimate the model parameters and check its validity.

In order to find the ARMA model parameters, we first need to plot the
ACF and PACF for $xstat$.

ACF $xtstat$         | PACF $xstat$
:-------------------------:|:-------------------------:
![image](https://github.com/gbrugere/Time_Series/blob/main/image/acf(xstat).png) | ![image](https://github.com/gbrugere/Time_Series/blob/main/image/pacf(xstat).png)

As the can see, the last significant lag for ACF is the $14^{th}$, so
$q = 14-1 = 13$, and the last significant lag for PACF is the $7^{th}$,
so $p = 7$.

We then minimise the BIC and AIC criterions by finding out the minimum
of the coefficient for the BIC and AIC matrix.
AIC matrix        | BIC matrix
:-------------------------:|:-------------------------:
![image](https://github.com/gbrugere/Time_Series/blob/main/image/AICmatrix.png) | ![image](https://github.com/gbrugere/Time_Series/blob/main/image/BICmatrix.png)

The minimisation of the coefficient provides us with two potential
couples : $(p,q) = (1,2)$ and $(p,q) = (5,3)$

We then plot the p-values for the Ljung-Box test for different lags to
discriminate between the two models.

![image](https://github.com/gbrugere/Time_Series/blob/main/image/LBtest.png)

The Ljung-Box tests for independance of the residuals against their
autocorrelation. To ensure that the model fits, we do not want to reject
this test. Thus, we chose the model associated with the first column
$p\textunderscore value\textunderscore 1$, that is to say the parameters
$(p,q) = (1,2)$.

##  Write the ARIMA(p,d,q) model for the chosen series

Since the series differenciated at the first order follows an ARMA(1,2),
the model we chose is an ARIMA(1,1,2).

The equation for the model is
$X_t = \phi_1 X_{t-1}  +  \epsilon_{t}  - \psi_1 \epsilon_{t-1} -  \psi_2 \epsilon_{t-2}$

# Part III : Prediction

## Write the equation satisfied by the confidence region of level α on the future values 

Our model follows an ARMA(1,2), thus we consider the following equation
:

$$X_T = \phi_1X_{T-1} + \epsilon_T - \psi_1\epsilon_{T-1} -\psi_2\epsilon_{T-2}$$
In addition we know that : $\forall h>0$
$\mathbb{E}[\epsilon_{T + H} | X_T, X_{T-1}, ...] = 0$, we get the
following calculations:\

$\begin{aligned}
  \hat{X}_{T+1} &= \mathbb{E}[X_{T+1} | X_T, X_{T-1}, ...] \\
  &= \mathbb{E}[\phi_1X_{T} + \epsilon_{T+1} - \psi_1\epsilon_{T} -\psi_2\epsilon_{T-1} | X_T, X_{T-1}, ...] \\
  &= \mathbb{E}[\phi_1X_{T} - \psi_1\epsilon_{T} -\psi_2\epsilon_{T-1} | X_T, X_{T-1}, ...] \\
  &= \phi_1X_{T} - \psi_1\epsilon_{T} -\psi_2\epsilon_{T-1}\\
\end{aligned}$ And :\
$\begin{aligned}
  \hat{X}_{T+2} &= \mathbb{E}[X_{T+2} | X_T, X_{T-1}, ...] \\
  &= \mathbb{E}[\phi_1X_{T+1} + \epsilon_{T+2} - \psi_1\epsilon_{T+1} -\psi_2\epsilon_{T} | X_T, X_{T-1}, ...] \\
  &= \mathbb{E}[\phi_1X_{T+1} -\psi_2\epsilon_{T} | X_T, X_{T-1}, ...] \\
  &= \mathbb{E}[\phi_1X_{T+1} | X_T, X_{T-1}, ...] - \psi_2\epsilon_{T} \\
  &= \phi_1\hat{X}_{T+1} - \psi_2\epsilon_{T} \\
\end{aligned}
\\
$\
Let : $\mathbb{X}_T = \begin{pmatrix} X_{T+1} \\ X_{T+2} \end{pmatrix}$
and
$\mathbb{\hat{X}}_T = \begin{pmatrix} \hat{X}_{T+1} \\ \hat{X}_{T+2} \end{pmatrix}$\
Then :\
$\begin{aligned}
  \mathbb{X}_T - \mathbb{\hat{X}}_T &= \begin{pmatrix} X_{T+1} - \hat{X}_{T+1} \\ X_{T+2} - \hat{X}_{T+2} \end{pmatrix} \\
  &= \begin{pmatrix} (\phi_1X_{T} + \epsilon_{T+1} - \psi_1\epsilon_{T} -\psi_2\epsilon_{T-1}) - (\phi_1X_{T} - \psi_1\epsilon_{T} -\psi_2\epsilon_{T-1}) \\
  (\phi_1X_{T+1} + \epsilon_{T+2} - \psi_1\epsilon_{T+1} -\psi_2\epsilon_{T}) - (\phi_1\hat{X}_{T+1} - \psi_2\epsilon_{T}) \end{pmatrix} \\
  &= \begin{pmatrix} \epsilon_{T+1} \\ \epsilon_{T+2} + \phi_1(X_{T+1} - \hat{X}_{T+1}) - \psi_1\epsilon_{T+1}  \end{pmatrix} \\
  &= \begin{pmatrix} \epsilon_{T+1} \\ \epsilon_{T+2} + (\phi_1 - \psi_1)\epsilon_{T+1} \end{pmatrix} \\
\end{aligned}$\
\
In fine, we have this equation :
$$\mathbb{X}_T - \mathbb{\hat{X}}_T =  \begin{bmatrix} 1 & 0 \\ \phi_1 - \psi_1 & 1 \end{bmatrix}
    \begin{pmatrix}\epsilon_{T+1} \\ \epsilon_{T+2} \end{pmatrix}$$
According to the guidelines: $(\epsilon_{t})_{t \in \mathbf{N}}$ are
Gaussian and i.i.d. Thus:\

::: center
$\begin{pmatrix}\epsilon_{T+1} \\ \epsilon_{T+2} \end{pmatrix} \sim \mathcal{N}_2(0,\,\begin{pmatrix}
\sigma^{2} & 0 \\ 0 & \sigma^{2}
\end{pmatrix})$ ,where $\sigma \in \mathbb{R}^*$
:::

and so we have :
$$\mathbb{X}_T - \mathbb{\hat{X}}_T \sim \mathcal{N}_2(\mu,\,\Sigma)$$
we immediatly have $\mu = 0$ , and : $$\begin{aligned}
\mathbb{V}[\mathbb{X}_T - \mathbb{\hat{X}}_T] &= \begin{bmatrix} 1 & 0 \\ \phi_1 - \psi_1 & 1 \end{bmatrix}^{T}
\begin{pmatrix}
\sigma^{2} & 0 \\ 0 & \sigma^{2}
\end{pmatrix}
\begin{bmatrix} 1 & 0 \\ \phi_1 - \psi_1 & 1 \end{bmatrix}^{T}

= \sigma^{2}\begin{pmatrix}
1 + (\phi_1 - \psi_1)^{2} & \phi_1 - \psi_1 \\
\phi_1 - \psi_1 & 1
\end{pmatrix}
\end{aligned}$$

$$\Sigma = \sigma^{2}\begin{pmatrix}
1 + (\phi_1 - \psi_1)^{2} & \phi_1 - \psi_1 \\
\phi_1 - \psi_1 & 1
\end{pmatrix}$$\
Moreover $\Sigma$ is a positive-definite matrix and therefore is
invertible and admits a square root\
which we will note : $\Sigma^{\frac{1}{2}}$

Thus :
$$\Vert(\Sigma^{-\frac{1}{2}}(\mathbb{X}_T - \mathbb{\hat{X}}_T))\Vert_{2}^{2} \sim \chi^{2}(2)$$
i.e :
$$(\mathbb{X}_T - \mathbb{\hat{X}}_T)^T\Sigma^{-1}(\mathbb{X}_T - \mathbb{\hat{X}}_T) \sim \chi^{2}(2)$$

So the confidence region of level $\alpha$ of $(X_{T+1}, X_{T+2})$ is :
$$\{X \in \mathbb{R}^{2} | (X-\hat{X})^T\Sigma(X-\hat{X}) \leq q_{1 - \alpha} \}$$

where $(q_{\alpha})_{\alpha \in [0, 1]}$ represent the quantiles of
order $\alpha$ of the $\chi^{2}(2)$ law

## Give the hypotheses used to get this region

We used several asumptions in order to obtain this results :

1.  The residuals $(\epsilon_t)_t$ were white noises following a
    gaussian distribution $\mathcal{N}_2(0,\,\sigma)$ with $\sigma$
    known and strictly positive

2.  The model is correct and the coefficients $\psi_1, \psi_2, \phi_1$
    are known.

## Graphically represent this region for $\alpha$= 95%. Comment on it.


![image](https://github.com/gbrugere/Time_Series/blob/main/image/plot_arima.png)


This graph represents the tail of the time series $x$ understudy,
followed by two blue points which are the two next prediction under the
assumption that the series follows a ARIMA(1,1,2) model. The grey bars
around the points represents the 95% confidence interval for the
predictions.

## Open question

$Y_{T+1}$ is available faster than $X_{T+1}$. This property can improve
the prediction of $X_{T+1}$ if $Y_T$ instantly causes $X_T$ in the
Granger sense. *i.e.*
$$\mathbb{E}[X_{T+1}|((X_{T}, Y_{T}), (X_{T-1}, Y_{T-1}), ...)\cup Y_{T+1}] \ne \mathbb{E}[X_{T+1}|((X_{T}, Y_{T}), (X_{T-1}, Y_{T-1}), ...)]$$

To test this condition we can compute a Granger causality test.
