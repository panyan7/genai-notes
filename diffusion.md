# Diffusion Models

## Itô Diffusion
A Itô diffusion is a stochastic differential equation (SDE) of the form

$$ dX_t = b(X_t)dt + \sigma(X_t) dB_t. $$

The Langevin SDE is

$$ dx_t = -\nabla f(x_t)dt + \sqrt{2}dB_t. $$

## Training Objective

The goal of training a diffusion model is to learn the "score function"

$$ \min_{s_t \in \mathcal{F}}~\mathbb{E}_{q_t} [\lVert s_t - \nabla \ln q_t\rVert^2] $$

which is equivalent to given input and noise, predict the noise

$$ \min_{s_t \in \mathcal{F}}~\mathbb{E}\left[\left\lVert s_t(\bar{X}_t) + \frac{1}{\sqrt{1-\exp(-2t)}} Z_t\right\rVert^2\right] $$

where $Z_t \sim N(0, I_d)$.

For video generation, we can just see $\mathcal{F}$ as a distribution over the videos.

## Denoising Diffusion Probabilistic Models
Denoising Diffusion Probabilistic Models (DDPM) [[HJA20]][1]

## Diffusion Transformers
Diffusion Transformers (DiTs) [[PX22]][2]

[1]: <https://arxiv.org/abs/2006.11239> "[HJA20] Denoising Diffusion Probabilistic Models"
[2]: <https://arxiv.org/abs/2212.09748> "[PX22] Scalable Diffusion Models with Transformers"
