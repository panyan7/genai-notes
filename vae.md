# Variational Auto-Encoders (VAE)

Suppose we have $Z \sim N(0,I_d)$ and $X|Z \sim N(\mu_\theta(z), \sigma_\theta^2(z)I)$

We have the following **Evidence Lower Bound (ELBO)** for VAE
$$\begin{align*}
    \log p_\theta(x)
    &= \log \int_z p_\theta(x, z)~dz\\
    &= \log \int_z \frac{p_\theta(x, z) q_\phi(z)}{q_\phi(z)}~dz\\
    &= \log \mathbb{E}_z\left[\frac{p_\theta(x, z)}{q_\phi(z)}\right]\\
    &\ge \mathbb{E}_z\left[\log\frac{p_\theta(x, z)}{q_\phi(z)}\right]\\
    &= L(\theta, \phi, x).
\end{align*}$$
Furthermore, we can show that
$$\begin{align*}
    L(\theta, \phi, x)
    &= -\mathbb{E}_z\left[\log \frac{q_\phi (z)}{p_\theta(z|x)}\right] + \log p_\theta(x)\\
    &= \log p_\theta(x) - D_{\mathrm{KL}}(q_\phi (z)\|p_\theta(z|x)).
\end{align*}$$
so the bound gets tighter as the two distributions gets closer.
