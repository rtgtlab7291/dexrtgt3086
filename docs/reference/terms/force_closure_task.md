# Force Closure Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/force_closure_task.py`.

## Residual

For world-frame contact point $\mathbf{p}_{w,i}$ and object-surface normal $\mathbf{n}_{w,i}$, the residual is the summed contact wrench:

$$
\mathbf{e}
=
\begin{bmatrix}
\sum_i\mathbf{n}_{w,i}\\
\sum_i\mathbf{p}_{w,i}\times\mathbf{n}_{w,i}
\end{bmatrix}
\in\mathbb{R}^6.
$$

The normals come from the closest points returned by `WarpScene` and are held fixed while differentiating.

## Jacobian

The force block is constant, while the torque block changes with the contact positions:

$$
\mathbf{J}[\cdot,j]
=
\begin{bmatrix}
\mathbf{0}\\
\sum_i
\frac{\partial\mathbf{p}_{w,i}}{\partial q_j}
\times\mathbf{n}_{w,i}
\end{bmatrix}.
$$
