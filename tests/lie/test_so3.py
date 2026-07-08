import numpy as np
import pytest
import torch
import warp as wp
from torch.testing import assert_close

from robokit.lie.pinocchio_so3 import PinocchioSO3
from robokit.lie.so3 import SO3
from robokit.lie.torch_so3 import TorchSO3
from robokit.lie.warp_so3 import WarpSO3
from robokit.utils.tensor_utils import to_numpy, to_torch
from robokit.xform.torch import random_quaternions, standardize_quaternion


IDENTITY_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


class SO3Adapter:
    def __init__(self, impl_class):
        self.impl_class = impl_class
        self.name = impl_class.__name__
        self.is_pinocchio = self.name == "PinocchioSO3"
        self.is_warp = self.name == "WarpSO3"
        self.is_torch = self.name == "TorchSO3"
        self._was_single_element: bool = False

    @property
    def needs_conversion(self) -> bool:
        return self.is_pinocchio or self.is_warp

    def _to_warp(self, tensor, dtype):
        wp.init()
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        if dtype in (wp.vec3, wp.vec4):
            self._was_single_element = tensor.ndim == 1
        elif dtype == wp.mat33:
            self._was_single_element = tensor.ndim == 2
        else:
            self._was_single_element = False
        return wp.from_numpy(arr, dtype=dtype)

    def create(self, wxyz):
        if self.is_pinocchio:
            return self.impl_class(to_numpy(wxyz)) if wxyz.ndim == 1 else None
        if self.is_warp:
            return self.impl_class(self._to_warp(wxyz, wp.vec4))
        return self.impl_class(wxyz)

    def from_matrix(self, matrix):
        if self.is_pinocchio:
            return self.impl_class.from_matrix(to_numpy(matrix))
        if self.is_warp:
            return self.impl_class.from_matrix(self._to_warp(matrix, wp.mat33))
        return self.impl_class.from_matrix(matrix)

    def exp(self, omega):
        if self.is_pinocchio:
            return self.impl_class.exp(to_numpy(omega))
        if self.is_warp:
            return self.impl_class.exp(self._to_warp(omega, wp.vec3))
        return self.impl_class.exp(omega)

    def convert_input(self, tensor, warp_dtype):
        if self.is_pinocchio:
            return to_numpy(tensor)
        if self.is_warp:
            return self._to_warp(tensor, warp_dtype)
        return tensor

    def to_tensor(self, result):
        if isinstance(result, torch.Tensor):
            return result
        if self.is_warp and isinstance(result, wp.array):
            arr = torch.from_numpy(result.numpy()).float()
            if self._was_single_element and result.shape == (1,):
                arr = arr.squeeze(0)
            return arr
        return to_torch(result, dtype=torch.float32)

    def get_wxyz(self, so3) -> torch.Tensor:
        return self.to_tensor(so3.wxyz)


@pytest.mark.parametrize(
    "impl_spec",
    [SO3, TorchSO3, WarpSO3, PinocchioSO3],
    ids=["SO3", "TorchSO3", "WarpSO3", "PinocchioSO3"],
)
class TestSO3Interface:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        np.random.seed(42)

    @pytest.fixture
    def adapter(self, impl_spec):
        return SO3Adapter(impl_spec)

    def test_initialization_single(self, adapter):
        quat = IDENTITY_QUAT
        so3 = adapter.create(quat)
        assert_close(adapter.get_wxyz(so3), quat, atol=1e-6, rtol=1e-6)

    def test_initialization_batch(self, adapter):
        quats = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.7071, 0.7071, 0.0, 0.0],
                [0.7071, 0.0, 0.7071, 0.0],
                [0.7071, 0.0, 0.0, 0.7071],
            ]
        )
        if adapter.is_pinocchio:
            for quat in quats:
                so3 = adapter.create(quat)
                assert_close(adapter.get_wxyz(so3), quat, atol=1e-6, rtol=1e-6)
        else:
            so3 = adapter.create(quats)
            assert_close(adapter.get_wxyz(so3), quats, atol=1e-6, rtol=1e-6)

    def test_from_matrix_identity(self, adapter):
        so3 = adapter.from_matrix(torch.eye(3))
        assert_close(adapter.get_wxyz(so3), IDENTITY_QUAT, atol=1e-6, rtol=1e-6)

    def test_as_matrix_identity(self, adapter):
        so3 = adapter.create(IDENTITY_QUAT)
        assert_close(adapter.to_tensor(so3.as_matrix()), torch.eye(3), atol=1e-6, rtol=1e-6)

    def test_matrix_roundtrip(self, adapter):
        random_quat = random_quaternions(1)[0]
        so3 = adapter.create(random_quat)
        original_matrix = adapter.to_tensor(so3.as_matrix())

        recovered_so3 = adapter.from_matrix(original_matrix)
        recovered_matrix = adapter.to_tensor(recovered_so3.as_matrix())
        assert_close(original_matrix, recovered_matrix, atol=1e-5, rtol=1e-5)

    def test_apply_identity(self, adapter):
        point = torch.tensor([1.0, 0.0, 0.0])
        so3 = adapter.create(IDENTITY_QUAT)
        result = adapter.to_tensor(so3.apply(adapter.convert_input(point, wp.vec3)))
        assert_close(result, point, atol=1e-6, rtol=1e-6)

    def test_apply_90_degree_rotation(self, adapter):
        rot_z_90 = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
        point = torch.tensor([1.0, 0.0, 0.0])
        so3 = adapter.create(rot_z_90)
        result = adapter.to_tensor(so3.apply(adapter.convert_input(point, wp.vec3)))
        assert_close(result, torch.tensor([0.0, 1.0, 0.0]), atol=1e-4, rtol=1e-4)

    def test_multiply_identity(self, adapter):
        quat = random_quaternions(1)[0]
        so3 = adapter.create(quat)
        so3_identity = adapter.create(IDENTITY_QUAT)

        assert_close(adapter.get_wxyz(so3 * so3_identity), quat, atol=1e-6, rtol=1e-6)
        assert_close(adapter.get_wxyz(so3_identity * so3), quat, atol=1e-6, rtol=1e-6)

    def test_inverse_identity(self, adapter):
        so3 = adapter.create(IDENTITY_QUAT)
        assert_close(adapter.get_wxyz(so3.inverse()), IDENTITY_QUAT, atol=1e-6, rtol=1e-6)

    def test_inverse_property(self, adapter):
        quat = random_quaternions(1)[0]
        so3 = adapter.create(quat)
        result_wxyz = standardize_quaternion(adapter.get_wxyz(so3 * so3.inverse()))
        expected = standardize_quaternion(IDENTITY_QUAT)
        assert_close(result_wxyz, expected, atol=1e-5, rtol=1e-5)

    def test_exp_identity(self, adapter):
        so3 = adapter.exp(torch.zeros(3))
        assert_close(adapter.get_wxyz(so3), IDENTITY_QUAT, atol=1e-6, rtol=1e-6)

    def test_log_identity(self, adapter):
        so3 = adapter.create(IDENTITY_QUAT)
        assert_close(adapter.to_tensor(so3.log()), torch.zeros(3), atol=1e-6, rtol=1e-6)

    def test_exp_log_consistency(self, adapter):
        omega = torch.tensor([0.1, 0.2, 0.3])
        so3 = adapter.exp(omega)
        recovered_omega = adapter.to_tensor(so3.log())
        assert_close(omega, recovered_omega, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self, adapter):
        if adapter.needs_conversion:
            pytest.skip(f"{adapter.name} doesn't support CUDA")

        quat = random_quaternions(1)[0].cuda()
        so3 = adapter.create(quat)
        assert so3.as_matrix().device == quat.device
        assert so3.log().device == quat.device


class TestSO3Consistency:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def _collect_results(self, method_fn):
        results = {}
        for name, impl in [("torch", TorchSO3), ("warp", WarpSO3), ("pinocchio", PinocchioSO3)]:
            adapter = SO3Adapter(impl)
            results[name] = method_fn(adapter)
        return results

    def _assert_all_close(self, results, atol=1e-6, rtol=1e-6):
        assert_close(results["torch"], results["warp"], atol=atol, rtol=rtol)
        assert_close(results["torch"], results["pinocchio"], atol=atol, rtol=rtol)

    def test_matrix_conversion_consistency(self):
        random_quat = random_quaternions(1)[0]

        matrices = self._collect_results(lambda a: a.to_tensor(a.create(random_quat).as_matrix()))
        self._assert_all_close(matrices)

        matrix = matrices["torch"]
        quats = self._collect_results(lambda a: standardize_quaternion(a.get_wxyz(a.from_matrix(matrix))))
        self._assert_all_close(quats)

    def test_apply_consistency(self):
        random_quat = random_quaternions(1)[0]
        point = torch.tensor([1.0, 2.0, 3.0])

        results = self._collect_results(
            lambda a: a.to_tensor(a.create(random_quat).apply(a.convert_input(point, wp.vec3)))
        )
        self._assert_all_close(results)

    def test_exp_log_consistency(self):
        omega = torch.tensor([0.5, -0.3, 0.2])

        quats = self._collect_results(lambda a: standardize_quaternion(a.get_wxyz(a.exp(omega))))
        self._assert_all_close(quats)

        quat = quats["torch"]
        omegas = self._collect_results(lambda a: a.to_tensor(a.create(quat).log()))
        self._assert_all_close(omegas)

    def test_jlog_consistency(self):
        test_quats = [
            IDENTITY_QUAT,
            torch.tensor([0.7071068, 0.7071068, 0.0, 0.0]),
            torch.tensor([0.7071068, 0.0, 0.7071068, 0.0]),
            torch.tensor([0.7071068, 0.0, 0.0, 0.7071068]),
            random_quaternions(1)[0],
        ]

        for quat in test_quats:
            quat = standardize_quaternion(quat)
            jlogs = {}
            for name, impl in [("torch", TorchSO3), ("warp", WarpSO3)]:
                adapter = SO3Adapter(impl)
                so3 = adapter.create(quat)
                assert so3 is not None
                jlogs[name] = adapter.to_tensor(so3.jlog())

            assert_close(jlogs["torch"], jlogs["warp"], atol=1e-6, rtol=1e-6)
            assert jlogs["torch"].shape == (3, 3)

            if torch.allclose(quat, IDENTITY_QUAT, atol=1e-6):
                assert_close(jlogs["torch"], torch.eye(3), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("impl_spec", [TorchSO3, WarpSO3], ids=["TorchSO3", "WarpSO3"])
class TestSO3BatchShape:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        np.random.seed(42)

    @pytest.fixture
    def adapter(self, impl_spec):
        return SO3Adapter(impl_spec)

    def test_initialization_batch_1xn(self, adapter):
        quats = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.7071, 0.7071, 0.0, 0.0],
                    [0.7071, 0.0, 0.7071, 0.0],
                ]
            ]
        )  # shape (1, 3, 4)
        so3 = adapter.create(quats)
        assert_close(adapter.get_wxyz(so3), quats, atol=1e-4, rtol=1e-4)

    def test_from_matrix_batch_1xn(self, adapter):
        matrices = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # shape (2, 3, 3)
        so3 = adapter.from_matrix(matrices)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        assert_close(adapter.get_wxyz(so3), expected, atol=1e-6, rtol=1e-6)

    def test_as_matrix_batch_1xn(self, adapter):
        quats = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.7071, 0.0, 0.0, 0.7071],
                ]
            ]
        )  # shape (1, 2, 4)
        so3 = adapter.create(quats)
        assert adapter.to_tensor(so3.as_matrix()).shape == (1, 2, 3, 3)

    def test_apply_batch_1xn(self, adapter):
        quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.0, 0.7071]]])
        points = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        so3 = adapter.create(quats)
        result = adapter.to_tensor(so3.apply(adapter.convert_input(points, wp.vec3)))
        expected = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_multiply_batch_1xn(self, adapter):
        quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.0, 0.7071]]])
        so3_a = adapter.create(quat)
        so3_b = adapter.create(quat)
        assert adapter.get_wxyz(so3_a * so3_b).shape == (1, 2, 4)

    def test_inverse_batch_1xn(self, adapter):
        quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.0, 0.7071]]])
        so3 = adapter.create(quats)
        result_wxyz = standardize_quaternion(adapter.get_wxyz(so3 * so3.inverse()))
        expected = standardize_quaternion(torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]))
        assert_close(result_wxyz, expected, atol=1e-5, rtol=1e-5)

    def test_exp_batch_1xn(self, adapter):
        omegas = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]])
        so3 = adapter.exp(omegas)
        assert adapter.get_wxyz(so3).shape == (1, 2, 4)

    def test_log_batch_1xn(self, adapter):
        quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.9553, 0.0475, 0.0951, 0.1426]]])
        so3 = adapter.create(quats)
        assert adapter.to_tensor(so3.log()).shape == (1, 2, 3)

    def test_jlog_batch_1xn(self, adapter):
        quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]]])
        so3 = adapter.create(quats)
        assert adapter.to_tensor(so3.jlog()).shape == (1, 2, 3, 3)
