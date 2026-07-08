# type: ignore[attr-defined]
import numpy as np
import pytest
import torch
import warp as wp
from torch.testing import assert_close

from robokit.lie.pinocchio_se3 import PinocchioSE3
from robokit.lie.se3 import SE3
from robokit.lie.torch_se3 import TorchSE3
from robokit.lie.warp_se3 import WarpSE3
from robokit.utils.tensor_utils import to_numpy, to_torch
from robokit.utils.warp_utils import wp_vec6, wp_vec7
from robokit.xform.torch import random_quaternions, standardize_quaternion


class SE3Adapter:
    """Adapter to handle implementation-specific differences for SE3."""

    def __init__(self, impl_class):
        self.impl_class = impl_class
        self.name = impl_class.__name__
        self.is_pinocchio = self.name == "PinocchioSE3"
        self.is_warp = self.name == "WarpSE3"
        self.is_torch = self.name == "TorchSE3"
        self._was_single_element: bool = False

    def _to_warp(self, tensor, dtype):
        wp.init()
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        # Track if input was a single element based on dtype and tensor dimensions
        if dtype == wp_vec7:
            self._was_single_element = tensor.ndim == 1
        elif dtype == wp_vec6:
            self._was_single_element = tensor.ndim == 1
        elif dtype == wp.vec3:
            self._was_single_element = tensor.ndim == 1
        elif dtype == wp.mat44:
            self._was_single_element = tensor.ndim == 2
        else:
            self._was_single_element = False
        return wp.from_numpy(arr, dtype=dtype)

    def create(self, xyz_wxyz: torch.Tensor):
        """Create SE3 instance, handling numpy conversion if needed."""
        if self.is_pinocchio:
            return self.impl_class(to_numpy(xyz_wxyz)) if xyz_wxyz.ndim == 1 else None
        if self.is_warp:
            return self.impl_class(self._to_warp(xyz_wxyz, wp_vec7))
        return self.impl_class(xyz_wxyz)

    def from_matrix(self, matrix: torch.Tensor):
        """Construct SE3 from matrix for each backend."""
        if self.is_pinocchio:
            import pinocchio as pin

            return self.impl_class(pin.SE3(matrix.detach().cpu().numpy()))
        if self.is_warp:
            return self.impl_class.from_matrix(self._to_warp(matrix, wp.mat44))
        if hasattr(self.impl_class, "from_matrix"):
            return self.impl_class.from_matrix(matrix)
        # Fallback via torch implementation
        return TorchSE3.from_matrix(matrix)

    def exp(self, omega: torch.Tensor):
        """Exponential map."""
        if self.is_pinocchio:
            return self.impl_class.exp(to_numpy(omega))
        if self.is_warp:
            return self.impl_class.exp(self._to_warp(omega, wp_vec6))
        return self.impl_class.exp(omega)

    def convert_input(self, tensor, warp_dtype):
        """Convert input tensor to appropriate format."""
        if self.is_pinocchio:
            return to_numpy(tensor)
        elif self.is_warp:
            return self._to_warp(tensor, warp_dtype)
        return tensor

    def to_tensor(self, result):
        """Convert result to torch tensor."""
        if self.is_warp and isinstance(result, wp.array):
            result_np = result.numpy()
            # Squeeze the first dimension if input was a single element
            if self._was_single_element and result_np.shape[0] == 1:
                result_np = result_np[0]
            return torch.from_numpy(result_np).float()
        return to_torch(result, dtype=torch.float32)

    def as_matrix_tensor(self, se3_obj) -> torch.Tensor:
        mat = se3_obj.as_matrix()
        return self.to_tensor(mat)

    def apply_points(self, se3_obj, points: torch.Tensor) -> torch.Tensor:
        """Apply transform to points, using matrix multiply for universal support."""
        mat = self.as_matrix_tensor(se3_obj)  # (..., 4, 4)
        if points.ndim == 1:
            p_h = torch.cat([points, torch.ones(1, dtype=points.dtype, device=points.device)])
            out = mat @ p_h
            return out[:3]
        else:
            ones = torch.ones(points.shape[:-1] + (1,), dtype=points.dtype, device=points.device)
            p_h = torch.cat([points, ones], dim=-1)
            out = (mat @ p_h.unsqueeze(-1)).squeeze(-1)
            return out[..., :3]

    def multiply(self, a, b):
        return a * b

    def inverse(self, se3_obj):
        return se3_obj.inverse()


@pytest.mark.parametrize(
    "impl_spec",
    [
        SE3,
        TorchSE3,
        WarpSE3,
        PinocchioSE3,
    ],
    ids=["SE3", "TorchSE3", "WarpSE3", "PinocchioSE3"],
)
class TestSE3Interface:
    """Test suite for SE3 interface - tests all implementations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        np.random.seed(42)

    @pytest.fixture
    def adapter(self, impl_spec):
        return SE3Adapter(impl_spec)

    @pytest.fixture
    def sample_transforms(self):
        return torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Identity
                [1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0],  # 90° X with translation
                [0.5, -1.0, 2.0, 0.7071, 0.0, 0.7071, 0.0],  # 90° Y
                [2.0, 1.0, -0.5, 0.7071, 0.0, 0.0, 0.7071],  # 90° Z
            ],
            dtype=torch.float32,
        )

    def test_initialization_single(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        se3 = adapter.create(xyz_wxyz)
        result = adapter.to_tensor(se3.xyz_wxyz) if adapter.is_pinocchio or adapter.is_warp else se3.xyz_wxyz
        assert_close(result, xyz_wxyz, atol=1e-6, rtol=1e-6)

    def test_initialization_batch(self, adapter: SE3Adapter, sample_transforms: torch.Tensor):
        if adapter.is_pinocchio:
            for x in sample_transforms:
                se3 = adapter.create(x)
                result = adapter.to_tensor(se3.xyz_wxyz)
                assert_close(result, x, atol=1e-4, rtol=1e-4)
        else:
            se3 = adapter.create(sample_transforms)
            result = adapter.to_tensor(se3.xyz_wxyz) if adapter.is_warp else se3.xyz_wxyz
            assert_close(result, sample_transforms, atol=1e-4, rtol=1e-4)

    def test_as_matrix_identity(self, adapter: SE3Adapter):
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        se3 = adapter.create(identity)
        mat = adapter.as_matrix_tensor(se3)
        expected = torch.eye(4)
        assert_close(mat, expected, atol=1e-6, rtol=1e-6)

    def test_matrix_roundtrip(self, adapter: SE3Adapter):
        t = torch.randn(3)
        q = random_quaternions(1)[0]
        se3 = adapter.create(torch.cat([t, q]))

        original_matrix = adapter.as_matrix_tensor(se3)
        recovered = adapter.from_matrix(original_matrix)
        recovered_matrix = adapter.as_matrix_tensor(recovered)
        assert_close(original_matrix, recovered_matrix, atol=1e-5, rtol=1e-5)

    def test_apply_identity(self, adapter: SE3Adapter):
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        p = torch.tensor([1.0, 2.0, 3.0])
        se3 = adapter.create(identity)
        result = adapter.apply_points(se3, p)
        assert_close(result, p, atol=1e-6, rtol=1e-6)

    def test_apply_translation_and_rotation(self, adapter: SE3Adapter):
        # 90° around Z: rotate (1,0,0)->(0,1,0) and add translation
        se3 = adapter.create(torch.tensor([1.0, 2.0, 3.0, 0.7071, 0.0, 0.0, 0.7071]))
        p = torch.tensor([1.0, 0.0, 0.0])
        result = adapter.apply_points(se3, p)
        expected = torch.tensor([1.0, 3.0, 3.0])
        assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_multiply_identity(self, adapter: SE3Adapter):
        t = torch.randn(3)
        q = random_quaternions(1)[0]
        x = torch.cat([t, q])
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        a = adapter.create(x)
        ident = adapter.create(identity)

        result = adapter.multiply(a, ident)
        result_w = adapter.to_tensor(result.xyz_wxyz)
        assert_close(
            standardize_quaternion(result_w[..., 3:]), standardize_quaternion(x[..., 3:]), atol=1e-6, rtol=1e-6
        )
        assert_close(result_w[..., :3], x[..., :3], atol=1e-6, rtol=1e-6)

        result = adapter.multiply(ident, a)
        result_w = adapter.to_tensor(result.xyz_wxyz)
        assert_close(
            standardize_quaternion(result_w[..., 3:]), standardize_quaternion(x[..., 3:]), atol=1e-6, rtol=1e-6
        )
        assert_close(result_w[..., :3], x[..., :3], atol=1e-6, rtol=1e-6)

    def test_inverse_identity(self, adapter: SE3Adapter):
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        se3 = adapter.create(identity)
        inv = adapter.inverse(se3)
        inv_w = adapter.to_tensor(inv.xyz_wxyz)
        assert_close(inv_w, identity, atol=1e-6, rtol=1e-6)

    def test_inverse_property(self, adapter: SE3Adapter):
        t = torch.randn(3)
        q = random_quaternions(1)[0]
        a = adapter.create(torch.cat([t, q]))
        inv = adapter.inverse(a)
        result = adapter.multiply(a, inv)
        result_w = adapter.to_tensor(result.xyz_wxyz)
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        result_w[..., 3:] = standardize_quaternion(result_w[..., 3:])
        assert_close(result_w, identity, atol=1e-5, rtol=1e-5)

    def test_exp_identity(self, adapter: SE3Adapter):
        zero = torch.zeros(6)
        se3 = adapter.exp(zero)
        result = adapter.to_tensor(se3.xyz_wxyz) if adapter.is_pinocchio or adapter.is_warp else se3.xyz_wxyz
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert_close(standardize_quaternion(result[3:]), expected[3:], atol=1e-6, rtol=1e-6)
        assert_close(result[:3], expected[:3], atol=1e-6, rtol=1e-6)

    def test_log_identity(self, adapter: SE3Adapter):
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        se3 = adapter.create(identity)
        omega = adapter.to_tensor(se3.log())
        expected = torch.zeros(6)
        assert_close(omega, expected, atol=1e-6, rtol=1e-6)

    def test_exp_log_consistency(self, adapter: SE3Adapter):
        # Use small angles to avoid wrapping issues
        small = torch.tensor([0.1, -0.2, 0.3, 0.05, -0.04, 0.02])  # [t, w]
        se3 = adapter.exp(small)
        recovered = adapter.to_tensor(se3.log())
        assert_close(recovered, small, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self, adapter: SE3Adapter):
        if adapter.is_pinocchio or adapter.is_warp:
            pytest.skip(f"{adapter.name} doesn't support CUDA")
        t = torch.randn(3, device="cuda")
        q = random_quaternions(1, device="cuda")[0]
        se3 = adapter.create(torch.cat([t, q]))
        mat = se3.as_matrix()
        assert mat.device == t.device
        log = se3.log()
        assert log.device == t.device


class TestSE3Consistency:
    """Test consistency between different SE3 implementations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def _get_results(self, input_data, method_fn):
        results = {}
        for name, impl in [
            ("torch", TorchSE3),
            ("warp", WarpSE3),
            ("pinocchio", PinocchioSE3),
        ]:
            adapter = SE3Adapter(impl)
            results[name] = method_fn(adapter)
        return results

    def test_matrix_conversion_consistency(self):
        t = torch.randn(3)
        q = random_quaternions(1)[0]
        xyz_wxyz = torch.cat([t, q])

        matrices = self._get_results(xyz_wxyz, lambda a: a.as_matrix_tensor(a.create(xyz_wxyz)))

        assert_close(matrices["torch"], matrices["warp"], atol=1e-6, rtol=1e-6)
        assert_close(matrices["torch"], matrices["pinocchio"], atol=1e-6, rtol=1e-6)

    def test_exp_map_singularity(self):
        """Test exponential map behavior near zero (singularity)."""
        # Exactly zero
        zero = torch.zeros(6)

        # Very small values
        small = torch.tensor([1e-6, -1e-6, 1e-7, -1e-7, 1e-8, -1e-8])

        for v in [zero, small]:
            transforms = {}
            for name, impl in [
                ("torch", TorchSE3),
                ("warp", WarpSE3),
                ("pinocchio", PinocchioSE3),
            ]:
                adapter = SE3Adapter(impl)
                se3 = adapter.exp(v)
                transforms[name] = (
                    adapter.to_tensor(se3.xyz_wxyz) if adapter.is_pinocchio or adapter.is_warp else se3.xyz_wxyz
                )
                transforms[name][..., 3:] = standardize_quaternion(transforms[name][..., 3:])

            assert_close(transforms["torch"], transforms["warp"], atol=1e-6, rtol=1e-6)
            assert_close(transforms["torch"], transforms["pinocchio"], atol=1e-6, rtol=1e-6)

    def test_multiply_consistency(self):
        x1 = torch.cat([torch.randn(3), random_quaternions(1)[0]])
        x2 = torch.cat([torch.randn(3), random_quaternions(1)[0]])

        results = {}
        for name, impl in [
            ("torch", TorchSE3),
            ("warp", WarpSE3),
            ("pinocchio", PinocchioSE3),
        ]:
            adapter = SE3Adapter(impl)
            a = adapter.create(x1)
            b = adapter.create(x2)
            c = adapter.multiply(a, b)
            results[name] = adapter.to_tensor(c.xyz_wxyz) if adapter.is_pinocchio or adapter.is_warp else c.xyz_wxyz
            results[name][..., 3:] = standardize_quaternion(results[name][..., 3:])

        assert_close(results["torch"], results["warp"], atol=1e-6, rtol=1e-6)
        assert_close(results["torch"], results["pinocchio"], atol=1e-6, rtol=1e-6)

    def test_exp_log_consistency(self):
        v = torch.tensor([0.5, -0.3, 0.2, 0.1, -0.2, 0.05])

        transforms = {}
        for name, impl in [
            ("torch", TorchSE3),
            ("warp", WarpSE3),
            ("pinocchio", PinocchioSE3),
        ]:
            adapter = SE3Adapter(impl)
            se3 = adapter.exp(v)
            transforms[name] = (
                adapter.to_tensor(se3.xyz_wxyz) if adapter.is_pinocchio or adapter.is_warp else se3.xyz_wxyz
            )
            transforms[name][..., 3:] = standardize_quaternion(transforms[name][..., 3:])

        assert_close(transforms["torch"], transforms["warp"], atol=1e-6, rtol=1e-6)
        assert_close(transforms["torch"], transforms["pinocchio"], atol=1e-6, rtol=1e-6)

        # Now test log consistency
        x = transforms["torch"].detach()
        logs = {}
        for name, impl in [
            ("torch", TorchSE3),
            ("warp", WarpSE3),
            ("pinocchio", PinocchioSE3),
        ]:
            adapter = SE3Adapter(impl)
            se3 = adapter.create(x)
            logs[name] = adapter.to_tensor(se3.log())

        assert_close(logs["torch"], logs["warp"], atol=1e-6, rtol=1e-6)
        assert_close(logs["torch"], logs["pinocchio"], atol=1e-6, rtol=1e-6)

    def test_exp_to_matrix_consistency(self):
        """Test that exp_to_matrix is consistent with exp + as_matrix."""
        v = torch.tensor([0.5, -0.3, 0.2, 0.1, -0.2, 0.05])

        # WarpSE3 direct
        adapter = SE3Adapter(WarpSE3)
        v_wp = adapter.convert_input(v, wp_vec6)

        # Method 1: exp then as_matrix
        se3 = WarpSE3.exp(v_wp)
        mat1 = adapter.to_tensor(se3.as_matrix())

        # Method 2: fused exp_to_matrix
        mat2 = adapter.to_tensor(WarpSE3.exp_to_matrix(v_wp))

        assert_close(mat1, mat2, atol=1e-6, rtol=1e-6)

        # Also test Torch wrapper if available/relevant
        from robokit.lie.warp_se3_kernels import SE3ExpMapToMatrix

        mat3 = SE3ExpMapToMatrix.apply(v.unsqueeze(0)).squeeze(0)
        assert_close(mat1, mat3, atol=1e-6, rtol=1e-6)

    def test_adjoint_consistency(self):
        """Test adjoint consistency across TorchSE3, PinocchioSE3, and WarpSE3."""
        test_transforms = [
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # Identity
            torch.tensor([1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0]),  # 90° X with translation
            torch.tensor([0.5, -1.0, 2.0, 0.7071, 0.0, 0.7071, 0.0]),  # 90° Y with translation
            torch.tensor([2.0, 1.0, -0.5, 0.7071, 0.0, 0.0, 0.7071]),  # 90° Z with translation
            torch.tensor([0.1, 0.2, 0.3, 0.9659, 0.0, 0.2588, 0.0]),  # Small rotation
        ]
        test_transforms.extend(_random_transforms(5, test_transforms[0].device))

        for xyz_wxyz in test_transforms:
            adjoints = {}

            for name, impl in [
                ("torch", TorchSE3),
                ("warp", WarpSE3),
                ("pinocchio", PinocchioSE3),
            ]:
                adapter = SE3Adapter(impl)
                se3 = adapter.create(xyz_wxyz)
                adjoints[name] = adapter.to_tensor(se3.adjoint())
            assert_close(adjoints["torch"], adjoints["warp"], atol=1e-4, rtol=1e-4)
            assert_close(adjoints["torch"], adjoints["pinocchio"], atol=1e-4, rtol=1e-4)

        # Batch consistency
        batch_transforms = torch.stack(test_transforms)
        batch_se3 = TorchSE3(batch_transforms)
        batch_adjoint = batch_se3.adjoint()

        for i in range(len(test_transforms)):
            individual_se3 = TorchSE3(test_transforms[i])
            individual_adjoint = individual_se3.adjoint()
            assert_close(batch_adjoint[i], individual_adjoint, atol=1e-6, rtol=1e-6)

        # adjoint of inverse
        xyz_wxyz = torch.tensor([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0])
        se3 = TorchSE3(xyz_wxyz)
        se3_inv = se3.inverse()

        Ad = se3.adjoint()
        Ad_inv = se3_inv.adjoint()
        Ad_inv_expected = torch.linalg.inv(Ad)
        assert_close(Ad_inv, Ad_inv_expected, atol=1e-4, rtol=1e-4)

    def test_jlog_consistency(self):
        """Test jlog consistency across TorchSE3, PinocchioSE3, and WarpSE3."""
        test_transforms = [
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # Identity
            torch.tensor([0.1, 0.2, 0.3, 0.9987, 0.0314, 0.0314, 0.0314]),  # Small transformation
            torch.tensor([1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0]),  # 90° X with translation
            torch.tensor([0.5, -1.0, 2.0, 0.8660, 0.0, 0.5, 0.0]),  # 60° Y with translation
        ]
        test_transforms.extend(_random_transforms(5, test_transforms[0].device))

        for xyz_wxyz in test_transforms:
            jlogs = {}
            for name, impl in [
                ("torch", TorchSE3),
                ("warp", WarpSE3),
                ("pinocchio", PinocchioSE3),
            ]:
                adapter = SE3Adapter(impl)
                se3 = adapter.create(xyz_wxyz)
                jlogs[name] = adapter.to_tensor(se3.jlog())
            assert_close(jlogs["torch"], jlogs["warp"], atol=1e-5, rtol=1e-5)
            assert_close(jlogs["torch"], jlogs["pinocchio"], atol=1e-4, rtol=1e-4)

        # Batch consistency
        batch_transforms = torch.stack(test_transforms)
        batch_se3 = TorchSE3(batch_transforms)
        batch_jlog = batch_se3.jlog()

        for i in range(len(test_transforms)):
            individual_se3 = TorchSE3(test_transforms[i])
            individual_jlog = individual_se3.jlog()
            assert_close(batch_jlog[i], individual_jlog, atol=1e-6, rtol=1e-6)

        # small transformations should have jlog close to identity
        small_xyz_wxyz = torch.tensor([0.1, 0.05, -0.02, 0.9988, 0.0349, -0.0175, 0.0262])
        se3_small = TorchSE3(small_xyz_wxyz)
        J_log = se3_small.jlog()
        assert torch.norm(J_log - torch.eye(6)) < 0.5

        # Device consistency (if CUDA available)
        if torch.cuda.is_available():
            cuda_xyz_wxyz = test_transforms[0].cuda()
            cuda_se3 = TorchSE3(cuda_xyz_wxyz)
            cuda_jlog = cuda_se3.jlog()

            assert cuda_jlog.device.type == "cuda"
            assert cuda_jlog.shape == (6, 6)


def _random_transforms(n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generate random SE(3) transforms [x, y, z, qw, qx, qy, qz]."""
    # Random translations in range [-5, 5]
    translations = torch.rand(n, 3, device=device) * 10 - 5

    # Random unit quaternions
    quaternions = random_quaternions(n, device=device)

    return torch.cat([translations, quaternions], dim=-1)


@pytest.mark.parametrize("impl_spec", [TorchSE3, WarpSE3], ids=["TorchSE3", "WarpSE3"])
class TestSE3BatchShape:
    """Test suite for SE3 batch shape handling - tests TorchSE3 and WarpSE3."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        np.random.seed(42)

    @pytest.fixture
    def adapter(self, impl_spec):
        return SE3Adapter(impl_spec)

    def test_initialization_batch_1xn(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0],
                    [0.5, -1.0, 2.0, 0.7071, 0.0, 0.7071, 0.0],
                ]
            ]
        )  # shape (1, 3, 7)
        se3 = adapter.create(xyz_wxyz)
        result = adapter.to_tensor(se3.xyz_wxyz) if adapter.is_warp else se3.xyz_wxyz
        assert_close(result, xyz_wxyz, atol=1e-4, rtol=1e-4)

    def test_from_matrix_batch_1xn(self, adapter: SE3Adapter):
        matrices = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)  # shape (2, 4, 4)
        se3 = adapter.from_matrix(matrices)
        result = adapter.to_tensor(se3.xyz_wxyz) if adapter.is_warp else se3.xyz_wxyz
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_as_matrix_batch_1xn(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 0.7071, 0.0, 0.0, 0.7071],
                ]
            ]
        )  # shape (1, 2, 7)
        se3 = adapter.create(xyz_wxyz)
        result = adapter.to_tensor(se3.as_matrix())
        assert result.shape == (1, 2, 4, 4)

    def test_apply_batch_1xn(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 0.7071, 0.0, 0.0, 0.7071],
                ]
            ]
        )  # shape (1, 2, 7)
        points = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])  # shape (1, 2, 3)
        se3 = adapter.create(xyz_wxyz)
        result = adapter.to_tensor(se3.apply(adapter.convert_input(points, wp.vec3)))
        expected = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 3.0, 3.0]]])
        assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_multiply_batch_1xn(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 0.7071, 0.0, 0.0, 0.7071],
                ]
            ]
        )  # shape (1, 2, 7)
        se3_a = adapter.create(xyz_wxyz)
        se3_b = adapter.create(xyz_wxyz)
        result = adapter.to_tensor((se3_a * se3_b).xyz_wxyz)
        assert result.shape == (1, 2, 7)

    def test_inverse_batch_1xn(self, adapter: SE3Adapter):
        sqrt2_2 = 0.7071067811865476
        xyz_wxyz = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, sqrt2_2, 0.0, 0.0, sqrt2_2],
                ]
            ]
        )  # shape (1, 2, 7)
        se3 = adapter.create(xyz_wxyz)
        result = se3 * se3.inverse()
        result_wxyz = adapter.to_tensor(result.xyz_wxyz)
        # Check shape is preserved
        assert result_wxyz.shape == (1, 2, 7)
        # Check translation is near zero
        assert_close(result_wxyz[..., :3], torch.zeros(1, 2, 3), atol=1e-5, rtol=1e-5)
        # Check quaternion is near identity
        result_quat = standardize_quaternion(result_wxyz[..., 3:])
        expected_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
        assert_close(result_quat, expected_quat, atol=1e-5, rtol=1e-5)

    def test_exp_batch_1xn(self, adapter: SE3Adapter):
        omegas = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.3, 0.05, -0.04, 0.02],
                ]
            ]
        )  # shape (1, 2, 6)
        se3 = adapter.exp(omegas)
        result = adapter.to_tensor(se3.xyz_wxyz) if adapter.is_warp else se3.xyz_wxyz
        assert result.shape == (1, 2, 7)

    def test_log_batch_1xn(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.3, 0.9987, 0.0314, 0.0314, 0.0314],
                ]
            ]
        )  # shape (1, 2, 7)
        se3 = adapter.create(xyz_wxyz)
        result = adapter.to_tensor(se3.log())
        assert result.shape == (1, 2, 6)

    def test_adjoint_batch_1xn(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0],
                ]
            ]
        )  # shape (1, 2, 7)
        se3 = adapter.create(xyz_wxyz)
        result = adapter.to_tensor(se3.adjoint())
        assert result.shape == (1, 2, 6, 6)

    def test_jlog_batch_1xn(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.3, 0.9987, 0.0314, 0.0314, 0.0314],
                ]
            ]
        )  # shape (1, 2, 7)
        se3 = adapter.create(xyz_wxyz)
        result = adapter.to_tensor(se3.jlog())
        assert result.shape == (1, 2, 6, 6)

    def test_xyz_quat_properties_batch_1xn(self, adapter: SE3Adapter):
        xyz_wxyz = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                    [4.0, 5.0, 6.0, 0.7071, 0.7071, 0.0, 0.0],
                ]
            ]
        )  # shape (1, 2, 7)
        se3 = adapter.create(xyz_wxyz)
        xyz = adapter.to_tensor(se3.xyz)
        quat_wxyz = adapter.to_tensor(se3.quat_wxyz)
        assert xyz.shape == (1, 2, 3)
        assert quat_wxyz.shape == (1, 2, 4)
        expected_xyz = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        expected_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]]])
        assert_close(xyz, expected_xyz, atol=1e-4, rtol=1e-4)
        assert_close(quat_wxyz, expected_quat, atol=1e-4, rtol=1e-4)


@wp.kernel
def _write_vec3_kernel(output: wp.array(dtype=wp.vec3), value: wp.vec3):
    tid = wp.tid()
    output[tid] = value


@wp.kernel
def _write_vec4_kernel(output: wp.array(dtype=wp.vec4), value: wp.vec4):
    tid = wp.tid()
    output[tid] = value


@wp.kernel
def _sum_vec3_kernel(src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    v = src[tid]
    dst[tid] = v[0] + v[1] + v[2]


@wp.kernel
def _sum_vec4_kernel(src: wp.array(dtype=wp.vec4), dst: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    v = src[tid]
    dst[tid] = v[0] + v[1] + v[2] + v[3]


class TestWarpSE3ComponentView:
    """Tests for WarpSE3 zero-copy component view mechanism (_component_view, xyz, quat_wxyz)."""

    def test_read_xyz(self):
        data = np.array(
            [
                [1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0],
                [4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        se3 = WarpSE3(wp.from_numpy(data, dtype=wp_vec7))
        xyz = torch.from_numpy(se3.xyz.numpy())
        expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert xyz.shape == (2, 3)
        assert_close(xyz, expected, atol=1e-6, rtol=1e-6)

    def test_read_quat_wxyz(self):
        data = np.array(
            [
                [1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0],
                [4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        se3 = WarpSE3(wp.from_numpy(data, dtype=wp_vec7))
        quat = torch.from_numpy(se3.quat_wxyz.numpy())
        expected = torch.tensor([[0.7071, 0.7071, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        assert quat.shape == (2, 4)
        assert_close(quat, expected, atol=1e-6, rtol=1e-6)

    def test_write_through_xyz_view(self):
        se3 = WarpSE3.identity(2)
        xyz_view = se3.xyz
        wp.launch(_write_vec3_kernel, dim=2, inputs=[xyz_view, wp.vec3(10.0, 20.0, 30.0)])
        result = torch.from_numpy(se3.xyz_wxyz.numpy())
        expected = torch.tensor(
            [
                [10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0],
                [10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_write_through_quat_wxyz_view(self):
        se3 = WarpSE3.identity(2)
        quat_view = se3.quat_wxyz
        wp.launch(_write_vec4_kernel, dim=2, inputs=[quat_view, wp.vec4(0.5, 0.5, 0.5, 0.5)])
        result = torch.from_numpy(se3.xyz_wxyz.numpy())
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            ]
        )
        assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_gradient_through_xyz_view(self):
        batch_size = 2
        data = np.array(
            [
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                [4.0, 5.0, 6.0, 0.7071, 0.7071, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        xyz_wxyz = wp.from_numpy(data, dtype=wp_vec7, requires_grad=True)
        se3 = WarpSE3(xyz_wxyz)
        xyz_view = se3.xyz
        output = wp.zeros(batch_size, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_sum_vec3_kernel, dim=batch_size, inputs=[xyz_view, output])

        grad_seed = wp.ones(batch_size, dtype=wp.float32)
        tape.backward(grads={output: grad_seed})

        grad = torch.from_numpy(se3.xyz_wxyz.grad.numpy())
        expected_grad = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert_close(grad, expected_grad, atol=1e-6, rtol=1e-6)

    def test_gradient_through_quat_wxyz_view(self):
        batch_size = 2
        data = np.array(
            [
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                [4.0, 5.0, 6.0, 0.7071, 0.7071, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        xyz_wxyz = wp.from_numpy(data, dtype=wp_vec7, requires_grad=True)
        se3 = WarpSE3(xyz_wxyz)
        quat_view = se3.quat_wxyz
        output = wp.zeros(batch_size, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_sum_vec4_kernel, dim=batch_size, inputs=[quat_view, output])

        grad_seed = wp.ones(batch_size, dtype=wp.float32)
        tape.backward(grads={output: grad_seed})

        grad = torch.from_numpy(se3.xyz_wxyz.grad.numpy())
        expected_grad = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        assert_close(grad, expected_grad, atol=1e-6, rtol=1e-6)
