# LGM1F

## Definition
Let's assume the instantaneous short rate follows a mean-reverting process:
$$dr_t = ... dt - \kappa_t r_t dt + \sigma_r dW_t $$

Now we can decompose this process into a state variable $x_t$ and a deterministic component enabling the fin to zero-coupon spot curve
$$r_t = x_t + f(0,t)$$

## Diffusion equation
Following the HJM equations, the dynamic is given by :
$$dx_t = (y_t - \kappa_t x_t) dt + \sigma_r dW_t$$
$$VAR(x_t)=y_t = \int_0^t exp(-2 \int_u^t \kappa_s ds) \sigma_u^2 du$$

## Zero-Coupon prices
We recall the definition of the zero-coupon prices
$$P(t,T) = e^{-\int_t^T f_u du}$$

We get the following expression of the zero-coupon price as a function of the state variable x_t
$$P(t,T) = \frac{P(0,T)}{P(0,t)} exp(-x_t G(t,T)-0.5 y_t G(t,T)^2 )$$
$$G(t,T) = \int_t^T exp(-\int_t^u \kappa_s ds) du$$

remark
$$\sigma_p(t,T) = \sigma_r(t) G(t,T)$$
$$G(t,T)=(G(0,T)-G(0,t))e^{\int_0^t \kappa_u du}$$

## options on zero-coupons bonds
Consider a european call option on a zero-coupon bond, paying at maturty T an amount
$$V_T = (P(T,T+\tau)-K)^+ $$

Then the price of the call is given by
$$V_t = P(t,T+\tau)\Phi(d_+)-P(t,T)K\Phi(d_-) $$

with 
$$d_{+-}=\frac{ln(\frac{P(t,T+\tau)}{KP(t,T)}) +- v/2}{\sqrt v}$$
$$v(t,T) = \int_t^T (\sigma_p(u,T+\tau)-\sigma_p(u,T))^2 du$$

Remark : The price of a put option is given by
$$P(t,T)K\Phi(-d_-) - P(t,T+\tau)\Phi(-d_+) $$

## cap/floors

## Swaption

Consider a swaption of expiry $T_0$ paying a coupon c at maturities $T_1,..,T_N$. By defining $\tau_i=T_i-T_{i-1}$ the value at expiry $T_0$ is the following :
$$V(T_0)=(1-P(T_0,T_N) - c \sum_{i=1}^{N}{\tau_i P(T_0,T_i)})^+$$

### Swaptions 1: semi analytical formula
Let's define Jamishidian strikes as $K_i=P(T_0, T_i, x*)$ where
$$1 = P(T_0,T_N,x*) + c \sum_1^N {\tau_i P(T_0,T_{i},x*)}$$

Then the Jamishidian trick gives us the following formula :
$$V(T_0) = (K_N - P(T_0,T_N,x_{T_0}))^+ + c \sum_1^N \tau_i (K_i-P(T_0,T_i,x_{T_0}))^+$$

### Swaptions 2: approximate formula for fast calibration

# Application to the constant case

In this section, we will consider the constant case where all parameters are fixed with respect to the time.

The general case formulas still hold,but with simplified expressions :

$$G(t,T)= \int_t^T \exp(-\kappa (u-t)) du = \frac{1-exp(-\kappa (T-t))}{\kappa}= G(0,T-t)= G(T-t)$$

$$y_t = \frac{\sigma^2}{2\kappa} (1-exp(-2\kappa t))$$

For pricing of swaptions, we need to express the quantity v:

$$v(t,T) = \int_t^T (\sigma_p(u,T+\tau)-\sigma_p(u,T))^2 du
         = \sigma^2 G(T)^2 \frac{G(2(T-t))}{2}$$


## diffusion of $x_t$ in the constant case

$$exp(\kappa t)x_t = x_0+\int_0^t exp(\kappa u) y_u du + \int_0^t exp(\kappa u) \sigma dW_u$$

$$x_t = e^{-\kappa t} x_0+ 
         \frac{\sigma^2}{2\kappa^2}(1-e^{-kt})^2 +
         \sigma \sqrt{\frac{1-e^{-2kt}}{2\kappa}}\epsilon$$


____________________________________________________________________


# Annex
## Annex 1: recall of the HJM
The HJM framework focuses on describing the evolution of the whole T-indexed bond prices structures P(.,T), starting from an initial state T->P(0,T) , and depending on a finite number of brownian motions.

By writing the dynamic as following :
$$\frac{dP(t,T)}{P(t,T)} = r_t dt -\sigma_B(t,T) dt$$

By writing that B(t,t)=1 we get
$$P(t,T) = \frac{P(0,T)}{P(0,t)} e^{\int_0^t -\frac{1}{2}(\sigma_B(s,T)^2-\sigma_B(s,t)^2) - (\sigma_B(s,T)-\sigma_B(s,t)) dW_s}$$

Using the instantenous forward definition, we get
$$f(t,T)= f(0,T)+\int_0^t \frac{\partial \sigma_B(s,T)}{\partial T} \sigma_B(s,T)  ds + \int_0^t \frac{\partial \sigma_B(s,T)}{\partial T} dW_s $$

By defining $\sigma_f=\frac{\partial \sigma_B(s,T)}{\partial T}$ we get the HJM equation :

$$\begin{equation*}
df(t,T) = \sigma_f(t,T) (\int_t^T \sigma_f(t,u)du) dt + \sigma_f(t,T) dWt 
\end{equation*}$$

The short rate can now be written as follows :
$$r_t = f(t,t) = f(0,t) + \int_0^t \sigma_f(u,t) (\int_u^t \sigma_f(u,s)ds)du + \int_0^t \sigma_f(u,t)dW_u$$

## Proof of the general LGM SDE
If we define $x_t$ as $r_t = f(0,t)+x_t$, then it follows that
$$x_t = \int_0^t \sigma_f(u,t) (\int_u^t \sigma_f(u,s)ds)du + \int_0^t \sigma_f(u,t)dW_u$$

In order to get a markovian equation of x_t we pose a seperable volatility function $\sigma_f(u,t)=\sigma_u K_{u,t}$ with the help of intermediary notations $K_t = e^{-\int_0^t \kappa_u du}$ and $K_{u,t}=\frac{K_t}{K_u}$.

Using Fubini's theorem, we get the following expression for $x_t$
$$x_t = K_t \int_0^t K_s (\int_0^s \frac{\sigma_u^2}{K_u^2} du) ds  + K_t \int_0^t \frac{\sigma_u}{K_u} dW_u$$

Applying Ito and defining $y_t := Var(x_t) = K_t^2 \int_0^t \frac{\sigma_u^2}{\K_u^2} du$, we get the followind SDE:
$$dx_t = (y_t-\kappa_t x_t) dt + \sigma_t dW_t $$


## Annex 2: Change of measure
## Annex 3: etc
