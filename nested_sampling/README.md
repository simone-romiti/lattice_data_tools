This folder contains scripts for the analysis of data produced with the Nested Sampling algorithm.

**Tools**:

- `./analyse_single_stream.py`: analysis of a single stream of data
- `./analyse_streams.py`: analysis of multiple streams of data
- `./chunks.py`: analysis of a single chunk of values from $L_\text{low}$ to $L_\text{top}$.

# Theoretical background

Nested Sampling (NS) is a Bayesian technique that allows to compute the evidence $Z$. 
Given the Likelihood $\mathcal{L}$ and the Prior $\pi$, Bayes' theorem states that:

\begin{equation}
Z = \int d\vec{\theta} \, \mathcal{L}(\vec{\theta}) \, \pi(\vec{\theta})
\end{equation}
The above integral can be rephrased as a 1-dimensional one @Ashton:2022grj:
\begin{equation}
\label{eq:ZNestedSamplingXIntegral}
Z = \int_{0}^{1} dX \, \mathcal{L}(X)
\, ,
\end{equation}

where with abuse of notation we denote $\mathcal{L}(X)$ the inverse function of the "phase space volume function":

\begin{equation}
X(\lambda) 
= \int_{\lambda < \mathcal{L}(\vec{\theta})} 
d \vec{\theta} \, \pi(\vec{\theta}) 
= \int d\vec{\theta} \, 
\Theta\left(\mathcal{L}(\vec{\theta}) - \lambda\right) 
\, \pi(\vec{\theta})
\, .
\end{equation}

We remark that:

- In the context of $\mathrm{SU}(N)$ lattice gauge theoreis, $Z$ is the partition function, $\mathcal{L}$ the Botzmann factor $e^{-S}$ and $\pi$ the Haar measure of the gauge group.
- $X$ goes from $0$ to $1$ by construction, as $\pi$ is a normalized probability distribution.
- If $\mathcal{L}$ does not vanish in its domain, $X$ is a differentiable and strictly monotonic ($\to$ invertible in its domain) function of $\lambda$.

The integral of Eq.\eqref{eq:ZNestedSamplingXIntegral} can be approximated using the Nested Sampling algorithm.
The result of an NS run is a list of $N_p$ values of the Likelihood:
\begin{equation}
\mathcal{L}^*_1,
\, \mathcal{L}^*_2, 
\, \ldots,
\, \mathcal{L}^*_{N_p}
\, \,
\end{equation}

Each of them corresponds to a shrinkage $dX$ of the phase space volume $X$. At every step, $X$ is reduced by a compression factor $t$: ${X_{k+1}=t X_{k}}$.
In a run with $n_\text{live}$ points, $t$ is distributed according to Beta distribution, thus:

\begin{equation}
\langle \log{t} \rangle  = 1/n_\text{live} \, .
\end{equation}

Thus, one can approximate the evidence $Z$ as:

\begin{equation}
Z \approx \sum_k w_k \mathcal{L}^{*}_k
\end{equation}

where $dX$ has been approximated by $w_k$.
For instance, one can consider its forward finite difference:

\begin{equation}
w_k = X_{k} - X_{k+1} = X_k (1 - t) 
= t^k (1-t) X_0 = t^k (1-t)
\end{equation}

**Practice**: The above expression is not useful in numerical calculations. The number of points is typically high, and the compressed volume may become too small for floating point precision on a machine. The solution is to **use logarithms throughout**. For the above case for instance:

\begin{align}
&
\log{w_k} = k \log(t) + \log{(1-t)}
\\
&
\log(w_k \cdot \mathcal{L}^*_k) = 
\log(w_k) + \log{\mathcal{L}^*_k} 
\\
&
\log{
    \left[
        w_k \mathcal{L}^*_k + w_{k+1} \mathcal{L}^*_{k+1} 
    \right]
    }
=
\log(
    \left[
        e^{\log{(w_k \mathcal{L}^*_k)}}
        +
        e^{\log{(w_{k+1} \mathcal{L}^*_{k+1})}}
    \right]
    )
\end{align}

where the last equation has to be used iteratively when computing $Z$.

## Processing streams and memory constrains



# References

::: {#refs}
:::