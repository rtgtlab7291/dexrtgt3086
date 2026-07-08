import pytest
import torch
import warp as wp

from robokit.xform.torch.transforms import rot_tl_to_tf_mat as rot_tl_tf_mat_torch
from robokit.xform.torch.transforms import rotate_points as rotate_points_torch
from robokit.xform.torch.transforms import transform_points as transform_points_torch
from robokit.xform.warp.torch_wrappers import rot_tl_to_tf_mat as rot_tl_to_tf_mat_warp
from robokit.xform.warp.torch_wrappers import rotate_points as rotate_points_warp
from robokit.xform.warp.torch_wrappers import transform_points as transform_points_warp


class TestTransforms:
    """Test suite for warp transform functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    @pytest.fixture
    def identity_transform(self):
        """4x4 identity transformation matrix."""
        return torch.eye(4, dtype=torch.float32)

    @pytest.fixture
    def sample_points(self):
        """Sample 3D points for testing."""
        return torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -3.0]], dtype=torch.float32
        )

    @pytest.fixture
    def translation_matrix(self):
        """Translation-only transformation matrix."""
        T = torch.eye(4, dtype=torch.float32)
        T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        return T

    @pytest.fixture
    def rotation_90z(self):
        """90-degree rotation around Z-axis."""
        return torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


class TestTransformPoints(TestTransforms):
    """Tests for transform_points function."""

    def _test_consistency(self, pts: torch.Tensor, tf_mat: torch.Tensor) -> torch.Tensor:
        """Compare torch and warp versions and return warp result."""
        result_warp = transform_points_warp(pts, tf_mat)
        result_torch = transform_points_torch(pts, tf_mat)
        torch.testing.assert_close(result_warp, result_torch, atol=1e-6, rtol=1e-6)
        return result_warp

    def test_identity_transformation(self, sample_points, identity_transform):
        """Test identity transformation leaves points unchanged."""
        result = self._test_consistency(sample_points, identity_transform)
        torch.testing.assert_close(result, sample_points)

    def test_translation_only(self, sample_points, translation_matrix):
        """Test pure translation transformation."""
        result = self._test_consistency(sample_points, translation_matrix)
        expected = sample_points + torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_rotation_90z_transformation(self, sample_points):
        """Test 90-degree rotation around Z-axis."""
        # Create 4x4 transformation matrix with 90-degree Z rotation
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        result = self._test_consistency(sample_points, T)

        # For 90-degree Z rotation: [x, y, z] -> [-y, x, z]
        expected = torch.tensor(
            [
                [-2.0, 1.0, 3.0],  # [1, 2, 3] -> [-2, 1, 3]
                [-5.0, 4.0, 6.0],  # [4, 5, 6] -> [-5, 4, 6]
                [0.0, 0.0, 0.0],  # [0, 0, 0] -> [0, 0, 0]
                [2.0, -1.0, -3.0],  # [-1, -2, -3] -> [2, -1, -3]
            ]
        )

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_device_compatibility(self, sample_points, identity_transform, device):
        """Test function works on different devices."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pts = sample_points.to(device)
        tf_mat = identity_transform.to(device)

        result = self._test_consistency(pts, tf_mat)

        assert result.device == pts.device
        torch.testing.assert_close(result, pts)

    def test_batch_processing(self):
        """Test batch processing with multiple transformations."""
        # Batch of 3 transformations and 2 points each
        pts = torch.randn(3, 2, 3, dtype=torch.float32)
        tf_mats = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        # Add different translations to each batch
        tf_mats[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
        tf_mats[1, :3, 3] = torch.tensor([0.0, 1.0, 0.0])
        tf_mats[2, :3, 3] = torch.tensor([0.0, 0.0, 1.0])

        result = self._test_consistency(pts, tf_mats)

        assert result.shape == pts.shape
        # Check that each batch was transformed correctly
        torch.testing.assert_close(result[0], pts[0] + torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(result[1], pts[1] + torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(result[2], pts[2] + torch.tensor([0.0, 0.0, 1.0]))

    def test_broadcasting(self):
        """Test broadcasting between points and transformation matrices."""
        # Single transformation, multiple point sets
        pts = torch.randn(3, 4, 3, dtype=torch.float32)  # 3 batches of 4 points
        tf_mat = torch.eye(4, dtype=torch.float32)  # Single transformation

        result = self._test_consistency(pts, tf_mat)
        assert result.shape == pts.shape

        # Multiple transformations, single point set
        pts_single = torch.randn(4, 3, dtype=torch.float32)  # Single batch of 4 points
        tf_mats = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)  # 2 transformations

        result = self._test_consistency(pts_single, tf_mats)
        assert result.shape == (2, 4, 3)

    def test_gradient_computation(self):
        """Test gradient computation for both points and transformation matrix."""
        pts = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
        tf_mat = torch.eye(4, dtype=torch.float32, requires_grad=True)

        # Test that gradients can be computed (but don't verify correctness due to Warp non-determinism)
        def func(p, t):
            return transform_points_warp(p, t)

        result = func(pts, tf_mat)
        loss = result.sum()

        # Check that backward pass completes without error
        try:
            loss.backward()
            # Verify gradients exist and have correct shapes
            assert pts.grad is not None
            assert tf_mat.grad is not None
            assert pts.grad.shape == pts.shape
            assert tf_mat.grad.shape == tf_mat.shape
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")

    def test_device_mismatch_error(self):
        """Test error when points and transformation matrix are on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")

        pts_cpu = torch.randn(2, 3, dtype=torch.float32)
        tf_mat_gpu = torch.eye(4, dtype=torch.float32).cuda()

        with pytest.raises(ValueError, match="must be on the same device"):
            transform_points_warp(pts_cpu, tf_mat_gpu)


class TestRotatePoints(TestTransforms):
    """Tests for rotate_points function."""

    def _test_consistency(self, pts: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
        """Compare torch and warp versions and return warp result."""
        result_warp = rotate_points_warp(pts, rot_mat)
        result_torch = rotate_points_torch(pts, rot_mat)
        torch.testing.assert_close(result_warp, result_torch, atol=1e-6, rtol=1e-6)
        return result_warp

    def test_identity_rotation(self, sample_points):
        """Test identity rotation leaves points unchanged."""
        identity_rot = torch.eye(3, dtype=torch.float32)
        result = self._test_consistency(sample_points, identity_rot)
        torch.testing.assert_close(result, sample_points)

    def test_90_degree_z_rotation(self, sample_points, rotation_90z):
        """Test 90-degree rotation around Z-axis."""
        result = self._test_consistency(sample_points, rotation_90z)

        # For 90-degree Z rotation: [x, y, z] -> [-y, x, z]
        expected = torch.tensor(
            [[-2.0, 1.0, 3.0], [-5.0, 4.0, 6.0], [0.0, 0.0, 0.0], [2.0, -1.0, -3.0]], dtype=torch.float32
        )

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_orthogonality_preservation(self):
        """Test that rotation matrices preserve orthogonality."""
        pts = torch.randn(10, 3, dtype=torch.float32)
        rot_mat = torch.randn(3, 3, dtype=torch.float32)
        # Make it orthogonal using QR decomposition
        rot_mat, _ = torch.qr(rot_mat)

        result = self._test_consistency(pts, rot_mat)

        # Check that distances from origin are preserved
        original_norms = torch.norm(pts, dim=-1)
        result_norms = torch.norm(result, dim=-1)
        torch.testing.assert_close(original_norms, result_norms, atol=1e-6, rtol=1e-6)

    def test_batch_rotation(self):
        """Test batch processing with multiple rotation matrices."""
        pts = torch.randn(3, 5, 3, dtype=torch.float32)
        rot_mats = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        # Apply different rotations to each batch
        angle = torch.tensor(torch.pi / 4, dtype=torch.float32)  # 45 degrees
        rot_mats[1] = torch.tensor(
            [[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0], [0, 0, 1]]
        )

        result = self._test_consistency(pts, rot_mats)
        assert result.shape == pts.shape

    def test_gradient_computation(self):
        """Test gradient computation for rotation."""
        pts = torch.randn(3, 3, dtype=torch.float32, requires_grad=True)
        rot_mat = torch.eye(3, dtype=torch.float32, requires_grad=True)

        def func(p, r):
            return rotate_points_warp(p, r)

        result = func(pts, rot_mat)
        loss = result.sum()

        try:
            loss.backward()
            # Verify gradients exist and have correct shapes
            assert pts.grad is not None
            assert rot_mat.grad is not None
            assert pts.grad.shape == pts.shape
            assert rot_mat.grad.shape == rot_mat.shape
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")


class TestRotTlToTfMat(TestTransforms):
    """Tests for rot_tl_to_tf_mat function."""

    def _test_consistency(self, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """Compare torch and warp versions and return warp result."""
        result_warp = rot_tl_to_tf_mat_warp(rotation, translation)
        result_torch = rot_tl_tf_mat_torch(rotation, translation)
        torch.testing.assert_close(result_warp, result_torch, atol=1e-6, rtol=1e-6)
        return result_warp

    def test_identity_transformation_matrix(self):
        """Test creating identity transformation matrix."""
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)

        result = self._test_consistency(rotation, translation)
        expected = torch.eye(4, dtype=torch.float32)

        torch.testing.assert_close(result, expected)

    def test_translation_only_matrix(self):
        """Test creating translation-only transformation matrix."""
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        result = self._test_consistency(rotation, translation)

        expected = torch.eye(4, dtype=torch.float32)
        expected[:3, 3] = translation

        torch.testing.assert_close(result, expected)

    def test_rotation_only_matrix(self, rotation_90z):
        """Test creating rotation-only transformation matrix."""
        translation = torch.zeros(3, dtype=torch.float32)

        result = self._test_consistency(rotation_90z, translation)

        expected = torch.eye(4, dtype=torch.float32)
        expected[:3, :3] = rotation_90z

        torch.testing.assert_close(result, expected)

    def test_combined_rotation_translation(self, rotation_90z):
        """Test combined rotation and translation matrix."""
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        result = self._test_consistency(rotation_90z, translation)

        expected = torch.eye(4, dtype=torch.float32)
        expected[:3, :3] = rotation_90z
        expected[:3, 3] = translation

        torch.testing.assert_close(result, expected)

    def test_batch_processing(self):
        """Test batch processing of multiple rotation-translation pairs."""
        batch_size = 3
        rotations = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        translations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        result = self._test_consistency(rotations, translations)

        assert result.shape == (batch_size, 4, 4)

        # Check each batch
        for i in range(batch_size):
            expected = torch.eye(4, dtype=torch.float32)
            expected[:3, 3] = translations[i]
            torch.testing.assert_close(result[i], expected)

    def test_broadcasting(self):
        """Test broadcasting between rotation and translation."""
        # Single rotation, multiple translations
        rotation = torch.eye(3, dtype=torch.float32)
        translations = torch.randn(3, 3)

        result = self._test_consistency(rotation, translations)
        assert result.shape == (3, 4, 4)

        # Multiple rotations, single translation
        rotations = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        result = self._test_consistency(rotations, translation)
        assert result.shape == (2, 4, 4)

    def test_gradient_computation(self):
        """Test gradient computation for both rotation and translation."""
        rotation = torch.eye(3, dtype=torch.float32, requires_grad=True)
        translation = torch.zeros(3, dtype=torch.float32, requires_grad=True)

        result = rot_tl_to_tf_mat_warp(rotation, translation)
        loss = result.sum()

        # Verify backward pass completes and produces gradients
        try:
            loss.backward()

            # Verify gradients exist and have correct shapes
            assert rotation.grad is not None
            assert translation.grad is not None
            assert rotation.grad.shape == rotation.shape
            assert translation.grad.shape == translation.shape

            # Verify gradients are not NaN or infinite
            assert not torch.isnan(rotation.grad).any()
            assert not torch.isnan(translation.grad).any()
            assert torch.isfinite(rotation.grad).all()
            assert torch.isfinite(translation.grad).all()

        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")

    def test_consistency_with_transform_points(self):
        """Test that rot_tl_to_tf_mat is consistent with transform_points."""
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        points = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

        # Using rot_tl_to_tf_mat + transform_points
        tf_mat = rot_tl_to_tf_mat_warp(rotation, translation)
        result1 = transform_points_warp(points, tf_mat)

        # Manual computation
        rotated = rotate_points_warp(points, rotation)
        result2 = rotated + translation.unsqueeze(0)

        torch.testing.assert_close(result1, result2, atol=1e-6, rtol=1e-6)

    def test_device_mismatch_error(self):
        """Test error when rotation and translation are on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")

        rotation_cpu = torch.eye(3, dtype=torch.float32)
        translation_gpu = torch.zeros(3, dtype=torch.float32).cuda()

        with pytest.raises(ValueError, match="must be on the same device"):
            rot_tl_to_tf_mat_warp(rotation_cpu, translation_gpu)

    def test_homogeneous_coordinate_format(self):
        """Test that the resulting matrix has correct homogeneous coordinate format."""
        rotation = torch.randn(3, 3, dtype=torch.float32)
        translation = torch.randn(3, dtype=torch.float32)

        result = self._test_consistency(rotation, translation)

        # Check bottom row is [0, 0, 0, 1]
        expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0])
        torch.testing.assert_close(result[3, :], expected_bottom)

        # Check top-left 3x3 is the rotation matrix
        torch.testing.assert_close(result[:3, :3], rotation)

        # Check top-right 3x1 is the translation vector
        torch.testing.assert_close(result[:3, 3], translation)


class TestIntegration(TestTransforms):
    """Integration tests combining multiple functions."""

    def test_transform_pipeline_consistency(self):
        """Test consistency of the complete transformation pipeline."""
        # Create test data
        points = torch.randn(5, 3, dtype=torch.float32)
        rotation = torch.randn(3, 3, dtype=torch.float32)
        rotation, _ = torch.qr(rotation)  # Make orthogonal
        translation = torch.randn(3, dtype=torch.float32)

        # Method 1: Use rot_tl_to_tf_mat + transform_points
        tf_mat = rot_tl_to_tf_mat_warp(rotation, translation)
        result1 = transform_points_warp(points, tf_mat)

        # Method 2: Use rotate_points + manual translation
        rotated = rotate_points_warp(points, rotation)
        result2 = rotated + translation.unsqueeze(0)

        torch.testing.assert_close(result1, result2, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_performance_scaling(self, batch_size):
        """Test performance with different batch sizes."""
        points = torch.randn(batch_size, 100, 3, dtype=torch.float32)
        tf_mats = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)

        result = transform_points_warp(points, tf_mats)
        assert result.shape == points.shape
