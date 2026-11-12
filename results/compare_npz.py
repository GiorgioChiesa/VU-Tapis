"""Compare two infer_untrimmed .npz files array by array (aggregate numbers only)."""
import sys
import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert sorted(a.files) == sorted(b.files), (a.files, b.files)
for k in sorted(a.files):
    x, y = a[k], b[k]
    assert x.shape == y.shape, (k, x.shape, y.shape)
    if x.dtype.kind in "iu":
        print(f"{k:14s} shape={x.shape} equal={np.array_equal(x, y)}")
        continue
    x, y = x.astype(np.float64), y.astype(np.float64)
    d = np.abs(x - y)
    print(f"{k:14s} shape={x.shape} dtype={a[k].dtype} max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
          f"bit_equal={np.array_equal(a[k], b[k])} allclose(1e-2)={np.allclose(x, y, rtol=1e-2, atol=1e-2)} "
          f"|ref|mean={np.abs(y).mean():.3e}")
