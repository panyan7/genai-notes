# Diffusion Models

## Itô Diffusion
A **Itô diffusion** is a stochastic differential equation (SDE) of the form

$$ dX_t = b(X_t)dt + \sigma(X_t) dB_t. $$

The **Langevin SDE** is

$$ dx_t = -\nabla f(x_t)dt + \sqrt{2}dB_t. $$

The goal of training a diffusion model is to learn the [**score function**](https://github.com/panyan7/genai-notes/blob/main/score.md)

$$ \min_{s_t \in \mathcal{F}}~\mathbb{E}_{p_t} [\lVert s_t(x) - \nabla_x \log p_t(x)\rVert^2] $$

which is equivalent to given input and noise, predict the noise

$$ \min_{s_t \in \mathcal{F}}~\mathbb{E}\left[\left\lVert s_t(\bar{X}_t) + \frac{1}{\sqrt{1-\exp(-2t)}} Z_t\right\rVert^2\right] $$

where $Z_t \sim N(0, I_d)$.

## Diffusion Models

### DDPM
[[HJA20]][1] considers a sequence of positive noise scales $\beta_1, \dots, \beta_N \in (0,1)$, and for each $x_0 \sim q(x)$, a discrete Markov chain is constructed such that

$$ q(x_i|x_{i-1}) = N(x_i; \sqrt{1-\beta_i} x_{i-1}, \beta_i I). $$

This is called the **forward process** or **diffusion process**, where Gaussian noise is added to the data according to the variance schedule $\beta_1, \dots, \beta_N$.

The joint distribution $p_\theta(x_{0:T})$ is called the **reverse process** and is defined as a Markov chain with learned Gaussian transitions such that

$$ p_\theta (x_{t-1}|x_t) = N(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$

### Diffusion SDE
[[SSK+20]][3] models the diffusion process can be modeled as the solution to an Itô SDE

$$ dx = f(x, t)dt + g(t)dw $$

where $w$ is the standard Wiener process.

The reverse process of a diffusion process is also a diffusion process, given by the SDE

$$ dx = \left(f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right)dt + g(t) d\bar{w} $$

where $\bar{w}$ is a standard Wiener process when time flows backward from $T$ to $0$. Notice that $\nabla_x \log p_t(x)$ is the score function.

## Diffusion Transformers
Diffusion Transformers (DiTs) [[PX22]][2]

[1]: <https://arxiv.org/abs/2006.11239> "[HJA20] Denoising Diffusion Probabilistic Models"
[2]: <https://arxiv.org/abs/2212.09748> "[PX22] Scalable Diffusion Models with Transformers"
[3]: <https://arxiv.org/abs/2011.13456> "[SSK+20] Score-Based Generative Modeling through Stochastic Differential Equations"
