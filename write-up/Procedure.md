### Procedure

For a given trial $i$, let $X^{(c_1, c_2)}=\{(rt_{1}, rt_2);( c_{1}, c_{2}) = \boldsymbol{c}\}$ be $n \times d$ sample matrix fall into the choice category $c$. 

1. Transform $(rt_1, rt_2)$ into log-space ($Z^{(c_1, c_2)}= \{(\text{log}(rt_1), \text{log}(rt_2))\}$)

   * To make sure the samples in log-space are always greater than 0(e.g., $rt_1$ < 1ms, although it might be impossible to happen), we shift these samples by a small constant $\text{log}(rt) = \text{log}(rt + shift)$
   * After the transformation, Jacobian matrix  is required to ensure the probability density remains correct in the original space, that is given by:

   $$
   \hat f_X(\boldsymbol{x}) = \hat f_Z(\boldsymbol{z}) \cdot \left|\frac{\partial \boldsymbol{z}}{\partial x}\right| = \hat f_Z(\log(\boldsymbol{z})) \cdot \begin{vmatrix}
   \frac{1}{x_1+\text{shift}_1} & 0 \\ 0 & \frac{1}{x_2+\text{shift}_2} \end{vmatrix}.
   $$

2. Calculate the covariance matrix $\Sigma$ and bandwidth matrix $H$

   * Covariance matrix $\Sigma$ is calculated by using all these samples in Z

   * Applying either Scott's rule(Eq. 2) or Silverman's rule(Eq. 3) to calculate the bandwidth matrix $H$
     $$
     H = n^{-1/(d+4)} \, \Sigma,
     $$

     $$
     H = \left(\frac{4}{d+2}\right)^{1/(d+4)} \, n^{-1/(d+4)} \, \Sigma.
     $$

   * If numbers of samples is smaller than dimension $n < d$, we returned a identity matrix $I$ and a small constant ε as the likelihood for this trial

3. We apply the multivariate gaussian kernel to these sample to estimate the probability density of the joint reaction time distribution
   $$
   \phi_H(u) = \frac{1}{(2\pi)^{d/2}|H|^{1/2}} \exp\!\left(-\tfrac{1}{2} u^\top H^{-1}u\right).
   $$
   we need to compute the determinant $|H|$ and the inverse matrix $\bold{H}^{-1}$. To do that, we conduct Cholesky decomposition:
   $$
   H = L L^\top
   $$
    Where $L = \begin{bmatrix}\ell_{11} & 0 \\ \ell_{21} & \ell_{22}\end{bmatrix}$. At this step, Cholesky decomposition requires the principal elements on the diagonal to be positive. Only in this case does the matrix can be positive definite. When the matrix doesn't have full rank(i.e., linearly dependent samples), the matrix can be positive semidefinite, which makes the Cholesky decomposition failed. 

   * An extreme case of matrix $H$ would be $\begin{bmatrix}\sigma & 0 \\ 0 & 0\end{bmatrix}$, when the variance of one dimension is $\sigma$ while the other dimension is 0. In this case, the determinant of the matrix will be 0, which makes the density to be infinite, and the inverse matrix is not normal as well.

4. For the human reaction time data point  $(v_1, v_2)$ in this trial to be estimated, we first transform into the log-space following the step 1 and then calcualte the vector $\boldsymbol{u}$ in Eq. 4 by calcualting the difference between this data point and the samples in matrix $\bold{Z}$:
   $$
   \boldsymbol{u} = \boldsymbol{v}^{\text{log}} - \boldsymbol{x_i}
   = \begin{bmatrix} v_1^{\text{log}} - z_{i1} \\ v_2^{\text{log}} - z_{i2} \end{bmatrix}.
   $$
   We can implicitly derive $\bold{H}^{-1}$ using $u^\top H^{-1} u \;=\; u^\top (L L^\top)^{-1} u = u^\top (L^\top)^{-1} L^{-1} u$. Let $t = L^{-1} u$ then
   $$
   u^\top H^{-1} u = t^T t = ||t||^2
   $$
   Thus, we need to solve the equation:
   $$
   \boldsymbol{L} \boldsymbol{t} = \boldsymbol{u}
   $$
   Because $L$ is a lower triangular matrix, it can be easily solved using [forward substitution](https://en.wikipedia.org/wiki/Triangular_matrix). We can derive $t_1 = \frac{u_1}{\ell_{11}},\quad t_2 = \frac{u_2 - \ell_{21} t_1}{\ell_{22}}, \boldsymbol{t} = \begin{bmatrix} \frac{u_1}{\ell_{11}} \\ \frac{u_2 - \ell_{21} t_1}{\ell_{22}}\end{bmatrix}$. 

   For determinant $|H|$, since $|H| = |L| \cdot |L^\top| = |L|^2$, $|H|^{1/2} = |L|$.

5. To makes the value more stable, let's take the log transformation on the Eq. 4, then it becomes:

$$
\text{log}(\phi_H(u)) = -(\frac{d}{2}\text{log}(2\pi) + |L|) - \frac{1}{2}||t||^2
$$

​	  We apply the Eq. 9 to every samples in matrix $Z$ to estimate the likelihood and take the average:
$$
\hat{f}_\mathbf{H}(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n K_\mathbf{H}(\mathbf{x} - \mathbf{x}_i)  =\frac{1}{n} \sum_{i=1}^n \exp(\text{log}(\phi_H(u)) ).
$$
​	  At last, we return the log likelihood
$$
\log \hat f(z) = s_{\max} + \log\!\left(\sum_i \exp(\text{log}(\phi_H(u)) - s_{\max})\right) - \log n.
$$
​	  Here, $s_{max}$ is the maximal likelihood among all these simulated samples $s_{\max} = \max_i \text{log}(\phi_H(u))$

6. Now we have $\hat f_Z(\log(\boldsymbol{z}))$ in Eq. 1, to transform back to the orginal space, we need to multiply the Jacobian matrix again
   $$
   \log f_V(v) = \log f_Z(z) - \log(v_1+\text{shift}_1) - \log(v_2+\text{shift}_2).
   $$
   That gives us the log-likelihood on the original space.





