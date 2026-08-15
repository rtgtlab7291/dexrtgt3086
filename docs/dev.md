# Development Guide

## Formatting, Linting, and Type Checking

- Use `ruff` for formatting and linting:

  ```shell
  uvx ruff check
  uvx ruff format
  ```

- Use `pyright` for type checking:

  ```shell
  uv run pyright
  ```

## Function Naming

Start function names with one of these verbs:

- `solve`: run an iterative optimization; may fail.
- `compute`: direct evaluation; always succeeds.
- `build`: construct objects, buffers, or kernels.
- `get`: cached lookup (build on miss).
- `set` / `clear`: mutate or drop state.

Warp kernels and functions additionally:

- Suffix by type: `*_kernel` for `@wp.kernel`, `*_func` for `@wp.func`.
- Verbs also include `transform` / `integrate` / `accept`.
- Variant tags before the suffix, fixed order: representation (`matrix`), algorithm (`sequential` / `local` / `accum`), then `backward`.

## Comments and Docstrings

Use single-line separators for sections, the short bracket form for class sections and
substantial phases in long functions.

```python
# --- kernels ----------------------------------------------------------------
# --- build solver ---
```

Use lowercase fragments for section labels and short code comments:

```python
# skip the normal/closest math (~2x faster)
```

Use normal sentence capitalization and punctuation for complete explanatory prose. Preserve proper names,
acronyms, and identifiers. Write section labels using full words.

Use Google-style docstrings (`Args:` / `Returns:` / `Raises:` sections) with single-backtick inline code.

## Testing

### Doctest

Ensure public functions have corresponding doctests:

```python
def quaternion_multiply(
    q_wxyz_1: Float[torch.Tensor, "... 4"], q_wxyz_2: Float[torch.Tensor, "... 4"]
) -> Float[torch.Tensor, "... 4"]:
    """
    Multiply two quaternions.

    Example:
        >>> q1 = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
        >>> q2 = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
        >>> expected = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
        >>> torch.allclose(quaternion_multiply(q1, q2), expected, atol=1e-6)
        True
    """
    return standardize_quaternion(quaternion_raw_multiply(q_wxyz_1, q_wxyz_2))
```

### Pytest

Aggregate test cases into a single class, and compare results across different implementations:

```python
class TestQuaternionToMatrix:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, quat_torch: torch.Tensor):
        """Helper function to test consistency between implementations."""
        quat_np = to_numpy(quat_torch)

        result_warp = quaternion_to_matrix_warp(quat_torch)
        result_torch = quaternion_to_matrix_torch(quat_torch)
        result_numpy = to_torch(quaternion_to_matrix_numpy(quat_np)).to(result_warp.device)

        torch.testing.assert_close(result_warp, result_torch, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result_warp, result_numpy, atol=1e-6, rtol=1e-6)

        return result_warp

    def test_identity_quaternion(self):
        """Test identity quaternion conversion."""
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = self._test_consistency(quat)
        expected = torch.eye(3)
        assert result.shape == (3, 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
```
