# Diffusion Models

## Background on Itô Diffusion
Let $W_t$ denote the standard Werner process. A **Itô diffusion** is a stochastic differential equation (SDE) of the form
$$ dX_t = b(X_t)dt + \sigma(X_t) dW_t. $$

The **Langevin SDE** is
$$ dx_t = -\nabla f(x_t)dt + \sqrt{2}dW_t. $$

## Diffusion Models

### DDPM
[[HJA20]][1] considers a sequence of positive noise scales $\beta_1, \dots, \beta_N \in (0,1)$, and for each $x_0 \sim q(x)$, a discrete Markov chain is constructed such that
$$ q(x_i|x_{i-1}) = \mathcal{N}(x_i; \sqrt{1-\beta_i} x_{i-1}, \beta_i I). $$

This is called the **forward process** or **diffusion process**, where Gaussian noise is added to the data according to the variance schedule $\beta_1, \dots, \beta_N$. Let $\alpha_t := 1-\beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, the distribution of $x_t$ conditional on $x_0$ is
$$ q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I).$$

The joint distribution $p_\theta(x_{0:T})$ is called the **reverse process** and is defined as a Markov chain with learned Gaussian transitions such that
$$ p_\theta (x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)). $$

The fact that the reverse process is also a diffusion process is important, because learning the mean and covariance is much easier than learning the full distribution, and can be modeled as a regression problem. For example, when the target distribution are images, then the regression problem is actually the image denoising objective, which can be solved using many methods such as CNNs.

From the diffusion SDE in the next section, we can see that
```math
\mathbb{E}[x_{t-1}|x_t] \approx x_t + \sigma_t^2 \nabla \log p_t(x_t)
```
where $p_t$ is the marginal distribution of $x_t$, and the $\nabla \log p_t(x_t)$ is again the [**score function**](https://github.com/panyan7/genai-notes/blob/main/score.md).

The training objective is, given $x_0$, sample $t \sim \mathrm{Uniform}([T])$, $\epsilon \sim \mathcal{N}(0, I)$, then
```math
\mathsf{loss}_\theta = \lVert \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)\rVert.
```
From the parametrization before, we can see that the expected value of the noise is the score function, so this is equivalent to the score matching objective. 

The sampling step is, sample $x_T \sim \mathcal{N}(0, I)$, then for $t \gets T, \dots, 1$, sample
```math
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z_t
```
where $z_t \sim \mathcal{N}(0, 1)$. This resembles the **Langevin dynamics** with $\epsilon_\theta$ as a learned gradient of the data density.

### Diffusion SDE
[[SSK+20]][3] models the diffusion process can be modeled as the solution to an Itô SDE

$$ dx = f(x, t)dt + g(t)dw $$

where $w$ is the standard Wiener process.

The reverse process of a diffusion process is also a diffusion process, given by the SDE

$$ dx = \left(f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right)dt + g(t) d\bar{w} $$

where $\bar{w}$ is a standard Wiener process when time flows backward from $T$ to $0$. Notice that $\nabla_x \log p_t(x)$ is the score function.

[[Yang21]][4]'s blog contains some examples and detailed theory for the diffusion SDE.

## Diffusion Transformers
Diffusion Transformers (DiTs) [[PX22]][2]

[1]: <https://arxiv.org/abs/2006.11239> "[HJA20] Denoising Diffusion Probabilistic Models"
[2]: <https://arxiv.org/abs/2212.09748> "[PX22] Scalable Diffusion Models with Transformers"
[3]: <https://arxiv.org/abs/2011.13456> "[SSK+20] Score-Based Generative Modeling through Stochastic Differential Equations"
[4]: <https://yang-song.net/blog/2021/score/> "[Yang21] Generative Modeling by Estimating Gradients of the Data Distribution"