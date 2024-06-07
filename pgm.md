# Probabilistic Graphical Models

## Score Matching

For MLE, we want to maximize $\sum_{i=1}^n \log p_\theta(x_i)$. If we want gradient descent, we need $\nabla_\theta \log p_\theta(x)$, which often involves the partition function $Z_\theta$. 

If we want to avoid this, we could be fitting instead

$$ \min_\theta \mathbb{E}_{x \sim p_{\mathrm{data}}}\lVert \nabla_x \log p_\theta(x) - \nabla_x \log p_{\mathrm{data}}(x)\rVert^2$$

where the function $\nabla_x \log p_{\mathrm{data}}(x)$ is called the *score function*. We can then rewrite the loss slightly

```math
\begin{align*}
    &\quad~\mathbb{E}_{x \sim p_{\mathrm{data}}} \lVert \nabla_x \log p_\theta(x) - \nabla_x \log p_{\mathrm{data}}(x)\rVert^2\\
    &= \mathbb{E}_{x \sim p_{\mathrm{data}}} \lVert \nabla_x \log p_\theta(x)\rVert^2 - 2\mathbb{E}_{x \sim p_{\mathrm{data}}}\langle \nabla_x \log p_\theta(x), \nabla_x \log p_{\mathrm{data}}(x)\rangle + \mathsf{const}\\
    &= \mathbb{E}_{x \sim p_{\mathrm{data}}} \lVert \nabla_x \log p_\theta(x)\rVert^2 + 2\mathbb{E}_{x \sim p_{\mathrm{data}}}[\mathrm{tr}(\nabla_x^2 \log p_{\theta}(x))] + \mathsf{const}
\end{align*}
```

where we use integration by parts for the last equality. So we can parametrize the score function $s_\theta(x) := \nabla_x \log p_\theta(x)$, and the training objective becomes

$$ \frac{1}{N} \sum_{i=1}^n \lVert s_\theta(x_i)\rVert^2 + 2\mathrm{tr}(D_\theta s_\theta(x_i))$$

which is the score matching objective.