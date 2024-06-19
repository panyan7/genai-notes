# Diffusion Models

## Background on Itô Diffusion
Let $W_t$ denote the standard Wiener process. A **Itô diffusion** is a stochastic differential equation (SDE) of the form

$$ dX_t = b(X_t)dt + \sigma(X_t) dW_t. $$

The **Langevin SDE** is

$$ dx_t = -\nabla f(x_t)dt + \sqrt{2}dW_t. $$

## Diffusion Models
### Diffusion Process
[[HJA20]][1] considers a sequence of positive noise scales $\beta_1, \dots, \beta_N \in (0,1)$, and for each $x_0 \sim q(x)$, a discrete Markov chain is constructed such that

$$ q(x_i|x_{i-1}) = \mathcal{N}(x_i; \sqrt{1-\beta_i} x_{i-1}, \beta_i I). $$

This is called the **forward process** or **diffusion process**, where Gaussian noise is added to the data according to the variance schedule $\beta_1, \dots, \beta_N$, so eventually $x_T \approx \mathcal{N}(0,1)$. Let $\alpha_t := 1-\beta_t$ and $\bar{\alpha}_t = \alpha_1 \cdots \alpha_t$, the distribution of $x_t$ conditional on $x_0$ is

$$ q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I).$$

The joint distribution $p_\theta(x_{0:T})$ is called the **reverse process** and is defined as a Markov chain with learned Gaussian transitions such that

$$ p_\theta (x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)). $$

The fact that the reverse process is also a diffusion process is important, because learning the mean and covariance is much easier than learning the full distribution, and can be modeled as a regression problem. For example, when the target distribution are images, then the regression problem is actually the image denoising objective, which can be solved using many methods such as CNNs [[NBZA24]][5].

[[HJA20]][1] parametrized the mean using a U-Net neural network, and the covariance is set to a fixed schedule $\sigma_t^2 I$. They found that adding small Gaussian noises each step works best, with $\beta_1 = 10^{-4}$ and $\beta_T = 0.02$ to be a linearly increasing sequence and $T = 1000$.

### Score Matching Objective
From the diffusion SDE in the next section, we can see that if $x_t = x_{t-1} + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma_t^2)$, then
```math
\mathbb{E}[x_{t-1}|x_t] \approx x_t + \sigma_t^2 \nabla_x \log p_t(x_t)
```
where $p_t$ is the marginal distribution of $x_t$, and the $\nabla_x \log p_t(x_t)$ is again the [**score function**](https://github.com/panyan7/genai-notes/blob/main/score.md). This is also known as **Tweedie's formula** in Bayesian statistics that suggests a Bayes correction for the MLE of the mean. A proof for the discretized case can be found in [[NBZA24]][5].

The training objective of DDPM is, given $x_0$, sample $t \sim \mathrm{Uniform}([T])$, $\epsilon \sim \mathcal{N}(0, I)$, then
```math
\mathsf{loss}_\theta = \lVert \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)\rVert^2.
```
From the parametrization before, we can see that the expected value of the noise given $x_t$ is the score function, so this is equivalent to the score matching objective. 

The sampling step is, sample $x_T \sim \mathcal{N}(0, I)$, then for $t \gets T, \dots, 1$, sample
```math
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z_t
```
where $z_t \sim \mathcal{N}(0, 1)$. This resembles the **Langevin dynamics** with $\epsilon_\theta$ as a learned gradient of the data density.


### Diffusion SDE
[[SSK+20]][3] models the diffusion process can be modeled as the solution to an Itô SDE

$$ dx = f(x, t)dt + g(t)dw $$

where $w$ is the standard Wiener process. The reverse process of a diffusion process is also a diffusion process, given by

$$ dx = \left(f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right)dt + g(t) d\bar{w} $$

where $\bar{w}$ is a standard Wiener process when time flows backward from $T$ to $0$. Notice that $\nabla_x \log p_t(x)$ is the score function. DDPM is a discretized special case of the diffusion SDE, given as

$$ dx = -\frac{1}{2} \beta(t) x~dt + \sqrt{\beta(t)}~dw $$

where $\beta(t/T) = T \beta_t$ as $T \to \infty$.

The reverse SDE can also be converted into a deterministic ODE, called the **probability flow** ODE, whose trajectories have the same marginals as the reverse-time SDE, given by

$$ \frac{dx}{dt} = f(x,t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x).$$

The score function is generally unknown in generative modeling. However, if we learn the score function, then we can use the learned score function to solve the reverse SDE.

[[Song21]][4]'s blog contains some examples and detailed theory for the diffusion SDE. The derivation for the reverse process SDE can be found in [[Winkler21]][6].

## Conditional Generation
### Classifier Guided Diffusion
[[DN21]][7] trained a classifier $f_\phi(y|x_t, t)$ on noisy images and use its gradients to guide the diffusion sampling process. Suppose we want to generate with conditional information $y$, such as a target class label. We can write the score function for the joint distribution as the following

```math
\begin{align*}
    \nabla_x \log q(x_t, y)
    &= \nabla_x \log q(x_t) + \nabla_x \log q(y | x_t)\\
    &\approx s_\theta(x_t, t) + \nabla_x \log f_\phi(y|x_t).
\end{align*}
```

Hence, a classifier guided diffusion model will look like

```math
\bar{\epsilon}_\theta(x_t, t) = \epsilon_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_x \log f_\phi(y|x_t).
```

### Classifier-Free Guidance
[[HS21]][8] showed that it is still possible to run conditional diffusion without classifiers. Let the unconditional model be parameterized as $\epsilon_\theta(x_t, t, y=\varnothing)$ and let the conditional model be parameterized as $\epsilon_\theta(x_t, t, y)$. The conditonal information is discarded at random during training. The gradient can be represented as

```math
\begin{align*}
    \nabla_x \log p(y|x_t) &= \nabla_x \log p(x_t|y) - \nabla_x p(x_t)\\
    &= -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \left(\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t)\right)
\end{align*}
```
```math
\bar{\epsilon}_\theta(x_t, t, y) = (w+1)\epsilon_\theta(x_t, t, y) - w\epsilon_\theta(x_t, t)
```

## Diffusion Transformers
Diffusion Transformers (DiTs) [[PX22]][2]

[1]: <https://arxiv.org/abs/2006.11239> "[HJA20] Denoising Diffusion Probabilistic Models"
[2]: <https://arxiv.org/abs/2212.09748> "[PX22] Scalable Diffusion Models with Transformers"
[3]: <https://arxiv.org/abs/2011.13456> "[SSK+20] Score-Based Generative Modeling through Stochastic Differential Equations"
[4]: <https://yang-song.net/blog/2021/score/> "[Song21] Generative Modeling by Estimating Gradients of the Data Distribution"
[5]: <https://arxiv.org/abs/2406.08929> "[NBZA24] Step-by-Step Diffusion: An Elementary Tutorial"
[6]: <https://ludwigwinkler.github.io/blog/ReverseTimeAnderson/> "[Winkler21] Reverse Time Stochatic Differential Equations [For Generative Modeling]"
[7]: <https://arxiv.org/abs/2105.05233> "[DN21] Diffusion Models Beat GANs on Image Synthesis"
[8]: <https://arxiv.org/abs/2207.12598> "[HS21] Classifier-Free Diffusion Guidance"