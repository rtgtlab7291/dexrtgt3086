# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
import torch
import warp as wp
from jaxtyping import Float


@wp.func
def hat_func(v: wp.vec3) -> wp.mat33:
    """Warp implementation of hat operator (skew-symmetric matrix)"""
    return wp.mat33(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)


@wp.func
def so3_jac_left_inv_func(theta: wp.vec3, eps: wp.float32) -> wp.mat33:
    """
    Warp implementation of SO(3) left jacobian inverse.
    """
    theta_squared = wp.dot(theta, theta)
    use_taylor = theta_squared < eps

    # Safe versions to avoid division by zero
    theta_squared_safe = wp.where(use_taylor, 1.0, theta_squared)
    theta_safe = wp.sqrt(theta_squared_safe)
    half_theta_safe = theta_safe / 2.0

    # Skew-symmetric matrix and its square
    skew_omega = hat_func(theta)
    skew_omega_squared = skew_omega * skew_omega

    eye = wp.identity(n=3, dtype=theta.dtype)

    # Taylor series
    taylor_result = eye - 0.5 * skew_omega + theta_squared_safe * skew_omega_squared / 12.0

    # Full computation
    cos_half_theta = wp.cos(half_theta_safe)
    sin_half_theta = wp.sin(half_theta_safe)
    coeff = (1.0 - theta_safe * cos_half_theta / (2.0 * sin_half_theta)) / theta_squared_safe

    full_result = eye - 0.5 * skew_omega + coeff * skew_omega_squared
    result = wp.where(use_taylor, taylor_result, full_result)
    return result


@wp.kernel
def so3_jlog_kernel(
    theta: wp.array(dtype=wp.vec3),
    eps: wp.float32,
    result: wp.array(dtype=wp.mat33),
):
    i = wp.tid()

    V_inv = so3_jac_left_inv_func(theta[i], eps)
    V_inv_T = wp.transpose(V_inv)
    result[i] = V_inv_T


# ------------------------ Torch Wrappers ------------------------ #


def get_epsilon(dtype: torch.dtype) -> float:
    """Get appropriate epsilon value for the given dtype."""
    if dtype == torch.float32:
        return 1e-4
    elif dtype == torch.float64:
        return 1e-8
    else:
        return 1e-4  # Default fallback


class SO3Jlog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
        wp.init()

        eps = get_epsilon(theta.dtype)

        # Flatten and convert to warp
        original_shape = theta.shape
        theta_flat = theta.view(-1, 3).contiguous()

        theta_wp = wp.from_torch(theta_flat, dtype=wp.vec3, requires_grad=theta.requires_grad)

        # Create output tensor
        jac_output = torch.empty(
            (theta_flat.shape[0], 3, 3),
            dtype=theta.dtype,
            device=theta.device,
            requires_grad=theta.requires_grad,
        )

        jac_output_wp = wp.from_torch(jac_output, dtype=wp.mat33)

        # Launch kernel
        wp.launch(
            kernel=so3_jlog_kernel,
            dim=(theta_flat.shape[0],),
            inputs=[theta_wp, eps, jac_output_wp],
            device=theta_wp.device,
        )

        result_shape = original_shape[:-1] + (3, 3)
        return jac_output.view(result_shape)

    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, return None (no backward pass implemented)
        # In practice, you would implement the backward pass for the jacobian
        return None
