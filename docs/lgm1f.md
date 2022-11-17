# LGM1F

## definition
Let's assume the short term instantaneous rate follows the following equation:
$$dr_t = ... dt - \kappa_t r_t dt + \sigma_r dW_t $$

Now we can decompose this process into a state variable x_t and a deterministic component enabling the fin to zero-coupon spot curve
$$r_t = x_t + f(0,t)$$

Following the HJM constraint, the dynamic is given by :
$$dx_t = (y_t - \kappa_t x_t) dt + \sigma_r dW_t$$
$$y_t = \int_0^t exp(-2 \int_u^t \kappa_s ds) \sigma_u^2 du$$

## zero coupon prices
We recall the definition of the zero-coupon prices
$$P(t,T) = e^{\int_t^T f_u du}$$

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

Consider a swaption of expiry $T_0$ paying a coupon c at maturities $T_1,..,T_N$. The value at expiry $T_0$ is the following :
$$V(T_0)=(1-P(T_0,T_1)-c \sum_{i=0}^{N-1}{\tau_i P(T_0,T_i)})^+$$

### Swaptions 1: semi analytical formula
Let's define Jamishidian strikes as $K_i=P(T_0, T_i, x*)$ where
$$1 = P(T_0,T_N,x*) + c \sum_0^{N-1}{\tau_i P(T_0,T_{i+1},x*)}$$

Then the Jamishidian trick gives us the following formula :
$$V(T_0) = (K_N - P(T_0,T_N,x_{T_0}))^+ + c \sum_0^N \tau_i (K_i-P(T_0,T_i,x_{T_0}))^+$$

### Swaptions: approximate formula for fast calibration
         
# Application to the constant case

In this section, we will consider the constant case where all parameters are fixed with respect to the time.

The general case formulas still hold,but with simplified expressions :

$$G(t,T)= \int_t^T \exp(-\kappa (u-t)) du = \frac{1-exp(-\kappa (T-t))}{\kappa}= G(0,T-t)= G(T-t)$$

$$y_t = \frac{\sigma^2}{2\kappa} (1-exp(-2\kappa t))$$

For pricing of swaptions, we need to express the quantity v
$$v(t,T) = \int_t^T (\sigma_p(u,T+\tau)-\sigma_p(u,T))^2 du
         = \sigma^2 G(T)^2 \frac{G(2(T-t))}{2}$$


## diffusion of $x_t$ in the constant case

$$exp(\kappa t)x_t = x_0+\int_0^t exp(\kappa u) y_u du + \int_0^t exp(\kappa u) \sigma dW_u$$

$$x_t = e^{-\kappa t} x_0+ 
         \frac{\sigma^2}{2\kappa^2}(1-e^{-kt})^2 +
         \sigma \sqrt{\frac{1-e^{-2kt}}{2\kappa}}\epsilon$$

_________

# Annex
## Annex 1: recall of the HJM

$$ df(t,T)=\sigma_f(t,T) (int_t^T \sigma_f(t,u)du)dt+\sigma_f(t,T)dWt $$

$$ \sigma_f(t,T)= \sigma_r(t) exp(-\int_t^T \kappa_u du)$$

## Annex 2: Change of measure
## Annex 3: etc
