# LGM1F

## definition
Let's definie the short term as :
$$ r_t = x_t + f(0,t) $$
with x_t the stochastic compenent and f(0,t) the deterministic compenent enabling the fit to the zero-coupon spot curve

Following the HJM constraint, the dynamic is given by :
$$ dx_t = (y_t - \kappa_t x_t) dt + \sigma_r dW_t $$
$$ y_t = \int_0^t exp(-2 \int_u^t \kappa_s ds) \sigma_u^2 du $$

zero coupon prices
$$ G(t,T) = \int_t^T exp(-\int_t^u \kappa_s ds) du $$
$$ P(t,T) = \frac{P(0,T)}{P(0,t)} exp(-x_t G(t,T)-0.5 y_t G(t,T)^2 ) $$



## options on zero-coupons bonds

## cap/floors

## swaptions

# Application to the constant case constant case
$ \kappa_t = \kappa $

$$ G(t,T)= \int_t^T exp(-\kappa (u-t)) du = \frac{1-exp(-\kappa (T-t))}{\kappa} $$

$$ y_t = \frac{\sigma^2}{2\kappa} (1-exp(-2\kappa t)) $$

## diffusion of xt

$$ exp(\kappa t)x_t = x_0+\int_0^t exp(\kappa u) y_u du + \int_0^t exp(\kappa u) \sigma dW_u$$

$$ x_t = e^{-\kappa t} x_0+ 
         \frac{\sigma^2}{2\kappa^2}(1-e^{-kt})^2 +
         \sigma \sqrt{\frac{1-e^{-2kt}}{2\kappa}}\epsilon
$$
## Annex 1: recall of the HJM
## Annex 2: Change of measure
## Annex 3: etc
