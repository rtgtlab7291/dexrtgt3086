import numpy as np
import pytest
import torch
import warp as wp

from robokit.utils.tensor_utils import to_numpy, to_torch
from robokit.xform.numpy.rotation_conversions import axis_angle_to_quaternion as axis_angle_to_quaternion_numpy
from robokit.xform.numpy.rotation_conversions import quaternion_to_axis_angle as quaternion_to_axis_angle_numpy
from robokit.xform.numpy.rotation_conversions import quaternion_to_matrix as quaternion_to_matrix_numpy
from robokit.xform.torch.rotation_conversions import axis_angle_to_quaternion as axis_angle_to_quaternion_torch
from robokit.xform.torch.rotation_conversions import quaternion_to_axis_angle as quaternion_to_axis_angle_torch
from robokit.xform.torch.rotation_conversions import quaternion_to_matrix as quaternion_to_matrix_torch
from robokit.xform.warp.torch_wrappers import axis_angle_to_quaternion as axis_angle_to_quaternion_warp
from robokit.xform.warp.torch_wrappers import quaternion_to_axis_angle as quaternion_to_axis_angle_warp
from robokit.xform.warp.torch_wrappers import quaternion_to_matrix as quaternion_to_matrix_warp


class TestQuaternionToMatrix:
    """Test suite for quaternion to matrix conversion functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, quat_torch: torch.Tensor):
        """Helper function to test consistency between implementations."""
        quat_np = to_numpy(quat_torch)

        # Get results from all implementations
        result_warp = quaternion_to_matrix_warp(quat_torch)
        result_torch = quaternion_to_matrix_torch(quat_torch)
        result_numpy = to_torch(quaternion_to_matrix_numpy(quat_np), device=result_warp.device, dtype=result_warp.dtype)

        # Check that all results are close to each other
        torch.testing.assert_close(result_warp, result_torch, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result_warp, result_numpy, atol=1e-6, rtol=1e-6)

        return result_warp

    def test_identity_quaternion(self):
        """Test identity quaternion conversion."""
        # Identity quaternion (w=1, x=0, y=0, z=0)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = self._test_consistency(quat)

        expected = torch.eye(3)
        assert result.shape == (3, 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_90_degree_rotations(self, device):
        """Test 90-degree rotations around each axis."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # 90-degree rotation around X-axis
        quat_x = torch.tensor([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0], dtype=torch.float32).to(device)
        expected_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).to(device)
        result_x = self._test_consistency(quat_x)
        torch.testing.assert_close(result_x, expected_x, atol=1e-6, rtol=1e-6)

        # 90-degree rotation around Y-axis
        quat_y = torch.tensor([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0], dtype=torch.float32).to(device)
        expected_y = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32).to(device)
        result_y = self._test_consistency(quat_y)
        torch.testing.assert_close(result_y, expected_y, atol=1e-6, rtol=1e-6)

        # 90-degree rotation around Z-axis
        quat_z = torch.tensor([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=torch.float32).to(device)
        expected_z = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32).to(device)
        result_z = self._test_consistency(quat_z)
        torch.testing.assert_close(result_z, expected_z, atol=1e-6, rtol=1e-6)

    def test_batch_processing(self):
        """Test batch processing of multiple quaternions."""
        # Batch of 3 quaternions: identity, 90deg around X, 90deg around Z
        quats = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # identity
                [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0],  # 90deg around X
                [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)],  # 90deg around Z
            ],
            dtype=torch.float32,
        )
        result = self._test_consistency(quats)

        expected = torch.stack(
            [
                torch.eye(3),
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
                torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            ]
        )

        assert result.shape == (3, 3, 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_arbitrary_batch_dimensions(self):
        """Test with arbitrary batch dimensions."""
        # 2x3 batch of quaternions
        quats = torch.randn(2, 3, 4)
        # Normalize to unit quaternions
        quats = quats / torch.norm(quats, dim=-1, keepdim=True)

        result = self._test_consistency(quats)

        assert result.shape == (2, 3, 3, 3)

        # Check that all matrices are orthogonal (R @ R.T = I)
        for i in range(2):
            for j in range(3):
                R = result[i, j]
                identity_check = R @ R.T
                torch.testing.assert_close(identity_check, torch.eye(3), atol=1e-5, rtol=1e-5)

    def test_quaternion_normalization(self):
        """Test that non-unit quaternions are handled correctly."""
        # Non-unit quaternion (should be normalized internally by warp)
        quat = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # Will be normalized to [1, 0, 0, 0]
        result = self._test_consistency(quat)

        expected = torch.eye(3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly for torch and warp versions."""
        quat = torch.tensor([1.0, 0.1, 0.1, 0.1], dtype=torch.float32, requires_grad=True)
        quat_normalized = quat / torch.norm(quat)

        # Test torch version
        quat_torch = quat_normalized.clone().detach().requires_grad_(True)
        result_torch = quaternion_to_matrix_torch(quat_torch)
        loss_torch = torch.sum(result_torch**2)
        loss_torch.backward()
        assert quat_torch.grad is not None
        assert quat_torch.grad.shape == quat.shape

        # Test warp version
        quat_warp = quat_normalized.clone().detach().requires_grad_(True)
        result_warp = quaternion_to_matrix_warp(quat_warp)
        loss_warp = torch.sum(result_warp**2)
        loss_warp.backward()
        assert quat_warp.grad is not None
        assert quat_warp.grad.shape == quat.shape

    def test_orthogonality_property(self):
        """Test that resulting matrices are orthogonal."""
        # Random unit quaternions
        quats = torch.randn(5, 4)
        quats = quats / torch.norm(quats, dim=-1, keepdim=True)

        matrices = self._test_consistency(quats)

        for i in range(5):
            R = matrices[i]
            # Check orthogonality: R @ R.T = I
            identity_check = R @ R.T
            torch.testing.assert_close(identity_check, torch.eye(3), atol=1e-5, rtol=1e-5)

            # Check determinant = 1 (proper rotation)
            det = torch.det(R)
            torch.testing.assert_close(det, torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with very small quaternions (near zero)
        small_quat = torch.tensor([1e-10, 1e-10, 1e-10, 1e-10])
        result = self._test_consistency(small_quat)
        assert not torch.any(torch.isnan(result))

        # Test with negative w component
        neg_w_quat = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        result_neg = self._test_consistency(neg_w_quat)
        pos_w_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result_pos = self._test_consistency(pos_w_quat)

        # Should give same rotation matrix (quaternion double cover)
        torch.testing.assert_close(result_neg, result_pos, atol=1e-6, rtol=1e-6)


class TestQuaternionToAxisAngle:
    """Test suite for quaternion to axis-angle conversion functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, quat_torch: torch.Tensor):
        """Helper function to test consistency between implementations."""
        quat_np = to_numpy(quat_torch)

        # Get results from all implementations
        result_warp = quaternion_to_axis_angle_warp(quat_torch)
        result_torch = quaternion_to_axis_angle_torch(quat_torch)
        result_numpy = to_torch(
            quaternion_to_axis_angle_numpy(quat_np), device=result_warp.device, dtype=result_warp.dtype
        )

        # Check that all results are close to each other
        torch.testing.assert_close(result_warp, result_torch, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result_warp, result_numpy, atol=1e-6, rtol=1e-6)

        return result_warp

    def test_identity_quaternion(self):
        """Test identity quaternion conversion."""
        # Identity quaternion (w=1, x=0, y=0, z=0)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = self._test_consistency(quat)

        # Expected result for identity is zero rotation, so axis-angle is [0, 0, 0]
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert result.shape == (3,)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_90_degree_rotations(self, device):
        """Test 90-degree rotations around each axis."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # 90-degree rotation around X-axis (pi/2)
        quat_x = torch.tensor([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0], dtype=torch.float32).to(device)
        expected_x = torch.tensor([np.pi / 2, 0.0, 0.0], dtype=torch.float32).to(device)
        result_x = self._test_consistency(quat_x)
        torch.testing.assert_close(result_x, expected_x, atol=1e-6, rtol=1e-6)

        # 90-degree rotation around Y-axis
        quat_y = torch.tensor([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0], dtype=torch.float32).to(device)
        expected_y = torch.tensor([0.0, np.pi / 2, 0.0], dtype=torch.float32).to(device)
        result_y = self._test_consistency(quat_y)
        torch.testing.assert_close(result_y, expected_y, atol=1e-6, rtol=1e-6)

        # 90-degree rotation around Z-axis
        quat_z = torch.tensor([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=torch.float32).to(device)
        expected_z = torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float32).to(device)
        result_z = self._test_consistency(quat_z)
        torch.testing.assert_close(result_z, expected_z, atol=1e-6, rtol=1e-6)

    def test_batch_processing(self):
        """Test batch processing of multiple quaternions."""
        quats = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # identity
                [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0],  # 90deg around X
                [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)],  # 90deg around Z
            ],
            dtype=torch.float32,
        )
        result = self._test_consistency(quats)

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [np.pi / 2, 0.0, 0.0],
                [0.0, 0.0, np.pi / 2],
            ],
            dtype=torch.float32,
        )

        assert result.shape == (3, 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_arbitrary_batch_dimensions(self):
        """Test with arbitrary batch dimensions."""
        # 2x3 batch of quaternions
        quats = torch.randn(2, 3, 4)
        quats = quats / torch.norm(quats, dim=-1, keepdim=True)
        result = self._test_consistency(quats)

        assert result.shape == (2, 3, 3)

    def test_quaternion_normalization(self):
        """Test that non-unit quaternions are handled correctly."""
        # Non-unit quaternion (should be normalized internally)
        quat = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # Normalized to [1, 0, 0, 0]
        result = self._test_consistency(quat)

        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly for torch and warp versions."""
        quat = torch.tensor([0.9, 0.1, 0.1, 0.1], dtype=torch.float32, requires_grad=True)
        quat_normalized = quat / torch.norm(quat)

        # Test torch version
        quat_torch = quat_normalized.clone().detach().requires_grad_(True)
        result_torch = quaternion_to_axis_angle_torch(quat_torch)
        loss_torch = torch.sum(result_torch**2)
        loss_torch.backward()
        assert quat_torch.grad is not None
        assert quat_torch.grad.shape == quat.shape

        # Test warp version
        quat_warp = quat_normalized.clone().detach().requires_grad_(True)
        result_warp = quaternion_to_axis_angle_warp(quat_warp)
        loss_warp = torch.sum(result_warp**2)
        loss_warp.backward()
        assert quat_warp.grad is not None
        assert quat_warp.grad.shape == quat.shape

    def test_180_degree_rotation(self):
        """Test a 180-degree rotation, which is a special case."""
        # 180-degree rotation around X-axis (w=0)
        quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        result = self._test_consistency(quat)

        # Expected result is [pi, 0, 0]
        expected = torch.tensor([np.pi, 0.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_small_angle_approximation(self):
        """Test conversion for very small rotation angles."""
        angle = 1e-8
        quat = torch.tensor([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0], dtype=torch.float32)
        result = self._test_consistency(quat)

        # Expected result is approximately [angle, 0, 0]
        expected = torch.tensor([angle, 0.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(result, expected, atol=1e-9, rtol=1e-9)


class TestAxisAngleToQuaternion:
    """Test suite for axis-angle to quaternion conversion functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, axis_angle_torch: torch.Tensor):
        """Helper function to test consistency between implementations."""
        axis_angle_np = to_numpy(axis_angle_torch)

        # Get results from all implementations
        result_warp = axis_angle_to_quaternion_warp(axis_angle_torch)
        result_torch = axis_angle_to_quaternion_torch(axis_angle_torch)
        result_numpy = to_torch(
            axis_angle_to_quaternion_numpy(axis_angle_np), device=result_warp.device, dtype=result_warp.dtype
        )

        # Check that all results are close to each other
        torch.testing.assert_close(result_warp, result_torch, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result_warp, result_numpy, atol=1e-6, rtol=1e-6)

        return result_warp

    def test_identity_conversion(self):
        """Test conversion of zero axis-angle (identity rotation)."""
        axis_angle = torch.zeros(3, dtype=torch.float32)
        quat = self._test_consistency(axis_angle)

        # Should produce identity quaternion [1, 0, 0, 0]
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device)
        torch.testing.assert_close(quat, expected, atol=1e-6, rtol=1e-6)

    def test_small_angles(self):
        """Test small angle approximation."""
        # Very small rotations where approximation is used
        small_angles = torch.tensor(
            [[1e-8, 2e-8, 3e-8], [1e-7, 0.0, 0.0], [0.0, 1e-7, 0.0], [0.0, 0.0, 1e-7], [1e-6, 1e-6, 1e-6]],
            dtype=torch.float32,
        )
        self._test_consistency(small_angles)

    def test_standard_rotations(self):
        """Test standard rotations around principal axes."""
        # 90-degree rotations around each axis
        axis_angles = torch.tensor(
            [
                [np.pi / 2, 0.0, 0.0],  # 90° around X
                [0.0, np.pi / 2, 0.0],  # 90° around Y
                [0.0, 0.0, np.pi / 2],  # 90° around Z
                [np.pi, 0.0, 0.0],  # 180° around X
                [0.0, np.pi, 0.0],  # 180° around Y
                [0.0, 0.0, np.pi],  # 180° around Z
            ],
            dtype=torch.float32,
        )
        self._test_consistency(axis_angles)

    def test_batch_dimensions(self):
        """Test various batch dimensions."""
        test_shapes = [(5, 3), (2, 4, 3), (3, 2, 5, 3), (1, 3), (10, 1, 3)]

        for shape in test_shapes:
            axis_angles = torch.randn(shape, dtype=torch.float32)
            quat = self._test_consistency(axis_angles)

            expected_shape = shape[:-1] + (4,)
            assert quat.shape == expected_shape

    def test_quaternion_properties(self):
        """Test that output quaternions have unit norm."""
        axis_angles = torch.randn(50, 3, dtype=torch.float32)
        quat = self._test_consistency(axis_angles)

        # Check unit norm
        norms = torch.norm(quat, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        edge_cases = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Zero rotation
                [2 * np.pi, 0.0, 0.0],  # Full rotation
                [np.pi + 1e-6, 0.0, 0.0],  # Just over π
                [np.pi - 1e-6, 0.0, 0.0],  # Just under π
                [1e-6, 0.0, 0.0],  # Epsilon threshold boundary
                [1e-6 + 1e-9, 0.0, 0.0],  # Just over epsilon
                [1e-6 - 1e-9, 0.0, 0.0],  # Just under epsilon
            ],
            dtype=torch.float32,
        )
        self._test_consistency(edge_cases)

    def test_gradient_consistency(self):
        """Test that gradients are consistent between implementations."""
        axis_angles_torch = torch.randn(10, 3, dtype=torch.float32, requires_grad=True)
        axis_angles_warp = axis_angles_torch.clone().detach().requires_grad_(True)

        # Forward pass
        quat_torch = axis_angle_to_quaternion_torch(axis_angles_torch)
        quat_warp = axis_angle_to_quaternion_warp(axis_angles_warp)

        # Backward pass
        loss_torch = quat_torch.sum()
        loss_warp = quat_warp.sum()

        loss_torch.backward()
        loss_warp.backward()

        # Compare gradients
        torch.testing.assert_close(
            axis_angles_torch.grad,
            axis_angles_warp.grad,
            atol=1e-4,
            rtol=1e-4,  # Slightly relaxed for GPU differences
        )

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large angles
        large_angles = torch.tensor(
            [
                [100.0, 0.0, 0.0],
                [0.0, 1000.0, 0.0],
                [0.0, 0.0, 50.0],
            ],
            dtype=torch.float32,
        )
        quat = self._test_consistency(large_angles)

        # Results should still be valid quaternions (unit norm)
        assert not torch.any(torch.isnan(quat))
        assert not torch.any(torch.isinf(quat))

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_device_consistency(self, device):
        """Test that results are consistent across CPU/GPU."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        axis_angles = torch.randn(20, 3, dtype=torch.float32).to(device)
        self._test_consistency(axis_angles)

    def test_inverse_consistency(self):
        """Test that axis_angle -> quaternion -> axis_angle is consistent."""

        original_axis_angles = torch.randn(20, 3, dtype=torch.float32)

        # Convert to quaternion and back
        quat = self._test_consistency(original_axis_angles)
        recovered_aa = quaternion_to_axis_angle_warp(quat)

        # Due to quaternion double cover, we need to handle ±2π equivalence
        def normalize_axis_angle(aa):
            angles = torch.norm(aa, dim=-1, keepdim=True)
            # Wrap angles to [-π, π]
            angles_wrapped = ((angles + np.pi) % (2 * np.pi)) - np.pi
            directions = aa / (angles + 1e-8)  # Avoid division by zero
            return directions * angles_wrapped

        original_normalized = normalize_axis_angle(original_axis_angles)
        recovered_normalized = normalize_axis_angle(recovered_aa)

        torch.testing.assert_close(original_normalized, recovered_normalized, atol=1e-4, rtol=1e-4)
