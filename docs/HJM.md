# HJM Framework

We start by defining the object of interest which are the interest rate curves. We model them through the zero-coupon prices and instantaneous forward rates associated :
$$P(t,T)=e^{-\int_t^T f(t,u) du} $$

The instantaneous short-rate is defined as $r_t = f(t,t)$, thus the current money account $\beta$ becomes:
$$\beta_t=e^{-\int_0^t r_u du}=e^{-\int_0^t f(u,u) du}$$

The focus of the HJM framework is to model the joint distribution of the curves rates. We pose the following dynamic:
$$df(t,T)=\mu(t,T)dt+\sigma(t,T)dW_t$$
Where $W_t$ is a d-dimensional Wiener process and $\mu, \sigma$ are F-adapted processes.

We can now write the dynamic of the zero-coupon bond curves
$$Y(t,T) :=log(P(t,T)) = -\int_t^T f(u,T)du$$

$$dY(t,T) = f(t,t) dt -\int_t^T df(t,u)du$$
$$          = r_t dt - \int_t^T \mu(t,u) dt du - \int_t^T \sigma(t,u) dW_t du$$

Defining $M(t,T) = \int_t^T \mu(t,u) du$ and $\Sigma(t,T) = \int_t^T \sigma(t,u) du$ we get
$$dY(t,T) = (r_t-M(t,T)) dt - \Sigma(t,T) dW_t$$

The dynamic of the zero-coupon bonds becomes:
$$\frac{dP(t,T)}{P(t,T)} = (r_t - M(t,T) + \frac{1}{2} \Sigma(t,T)\Sigma(t,T)^\top)dt - \Sigma(t,T) dW_t$$

Knowing from the arbitrage-free condition, we get that $\frac{P}{\beta}$ shoud be a Q-martingale and thus $M(t,T)=\frac{1}{2} \Sigma(t,T)\Sigma(t,T)^\top$. Differentiating this last equation, we get :
$$\mu(t,T) = \sigma(t,T) \int_t^T \sigma(t,u)^\top du$$

____
This leads to the main equation of the HJM framework, which describes the evolution of the forwards depending only on the volatility functions:
$$df(t,T) = (\sigma(t,T) \int_t^T \sigma(t,u)^\top du) dt + \sigma(t,T) dW_t$$
____



