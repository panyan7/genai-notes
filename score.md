# Score-Based Models

## Score Matching

We can train $p_\theta(x)$ using MLE by maximizing the likelihood

$$\max_\theta \sum_{i=1}^n \log p_\theta(x_i).$$

If we want to parametrize using a neural network, a natural idea is to use gradient descent, in which case we need to evaluate $\nabla_\theta \log p_\theta(x)$, which often involves calculating the gradient of the partition function $\nabla_\theta Z_\theta$. This is usually intractable.

If we want to avoid this, we could be fitting the **score function**

$$s_\theta(x) := \nabla_x \log p_\theta(x)$$

instead, which is independent of the partition function $Z_\theta$ as $\nabla_x Z_\theta = 0$.  To train the model, we can minimize the **Fisher divergence** between the model and data distribution, defined as

```math
\min_\theta \mathbb{E}_p\lVert \nabla_x \log p_\theta(x) - \nabla_x \log p(x)\rVert^2
```

where the function $\nabla_x \log p(x)$ is called the *score function*. We can then rewrite the loss slightly

```math
\begin{align*}
    &\quad~\mathbb{E}_p \lVert \nabla_x \log p_\theta(x) - \nabla_x \log p(x)\rVert^2\\
    &= \mathbb{E}_p \lVert \nabla_x \log p_\theta(x)\rVert^2 - 2\mathbb{E}_p\langle \nabla_x \log p_\theta(x), \nabla_x \log p(x)\rangle + \mathsf{const}\\
    &= \mathbb{E}_p \lVert \nabla_x \log p_\theta(x)\rVert^2 + 2\mathbb{E}_p[\mathrm{tr}(\nabla_x^2 \log p_{\theta}(x))] + \mathsf{const}
\end{align*}
```

where we use integration by parts for the last equality. So we can parametrize the score function , and the training objective becomes

$$ \frac{1}{N} \sum_{i=1}^n \lVert s_\theta(x_i)\rVert^2 + 2\mathrm{tr}(D_\theta s_\theta(x_i))$$

which is the score matching objective.

## Langevin Dynamics
**Langevin Dynamics** provides a MCMC procedure to sample from a distribution using only its score function. It first samples $X_0 \sim \pi(x)$ from a prior distribution, then iterates the following

$$X_{i+1} \gets X_i + \varepsilon \nabla_x \log p(X) + \sqrt{2\varepsilon} Z_i$$

where $Z_i \sim N(0, I_d)$. When $\varepsilon \to 0$ and $N \to \infty$, $x_N$ converges to a sample drawn from $p(x)$ under regularity conditions.

## 