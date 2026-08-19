"""
Benchmarks the Cython kernels in larp/field_cy against the current pure
NumPy implementations in larp/field.py, on obstacle/query counts
representative of the city scenarios in benchmark/cities.toml (hundreds to
thousands of obstacles, grid evaluations up to ~160k points for to_image).

Run (after building the extension with `python setup.py build_ext --inplace`
from the repo root):

    .venv/bin/python benchmark/bench_field_cy.py
"""
import time
import numpy as np

import larp.field as lf
import larp.field_cy as fc

rng = np.random.default_rng(0)


def timeit(fn, repeats=5):
    # warmup
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def make_city_multipolygon(n_buildings=800, sides=6, extent=700.0, seed=0):
    """Synthetic city of small convex polygon 'buildings', similar scale to
    benchmark/cities.toml (dist up to 700m)."""
    r = np.random.default_rng(seed)
    coords = []
    centers = r.uniform(-extent, extent, size=(n_buildings, 2))
    for c in centers:
        radius = r.uniform(3.0, 12.0)
        angles = np.sort(r.uniform(0, 2 * np.pi, size=sides))
        ring = c + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        coords.append(ring)
    return lf.MultiPolygonRGJ(coords)


def bench_polygon_repulsion():
    mp = make_city_multipolygon(n_buildings=800, sides=6)
    n_points = 4000
    pts = rng.uniform(-700, 700, size=(n_points, 2))

    def numpy_version():
        return mp.repulsion_vector(pts, min_dist_select=True)

    metric = mp.inv_repulsion

    def cython_version():
        return fc.segment_repulsion_vectors(pts, mp._p1, mp._v, mp._v_dot_v, metric)

    t_np = timeit(numpy_version)
    t_cy = timeit(cython_version)

    out_np = numpy_version()
    out_cy = cython_version()
    max_err = np.max(np.abs(out_np - out_cy))

    n_segments = mp._S
    print(f"[Polygon repulsion_vector] buildings=800 segments={n_segments} points={n_points}")
    print(f"  numpy : {t_np*1000:8.2f} ms")
    print(f"  cython: {t_cy*1000:8.2f} ms   speedup: {t_np/t_cy:5.2f}x   max_abs_err={max_err:.2e}")
    print()


def bench_point_in_polygon():
    # one large polygon with many edges (e.g. a detailed building/park outline)
    n_edges = 2000
    angles = np.linspace(0, 2 * np.pi, n_edges, endpoint=False)
    noise = 1.0 + 0.05 * np.sin(angles * 17)
    outer = 500 * noise[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    poly = lf.PolygonRGJ([outer])

    n_points = 20000
    pts = rng.uniform(-600, 600, size=(n_points, 2))

    def numpy_version():
        return poly._point_in_polygon_vectorized(pts)

    outer_ring = poly.rings[0]

    def cython_version():
        return fc.point_in_rings(pts, outer_ring, [])

    t_np = timeit(numpy_version)
    t_cy = timeit(cython_version)

    out_np = numpy_version()
    out_cy = cython_version().astype(bool)
    mismatch = np.sum(out_np != out_cy)

    print(f"[Point-in-polygon] edges={n_edges} points={n_points}")
    print(f"  numpy : {t_np*1000:8.2f} ms")
    print(f"  cython: {t_cy*1000:8.2f} ms   speedup: {t_np/t_cy:5.2f}x   mismatches={mismatch}/{n_points}")
    print()


def bench_field_eval_point_obstacles():
    n_obstacles = 3000
    centers = rng.uniform(-700, 700, size=(n_obstacles, 2))
    repulsions = np.tile(np.eye(2) * rng.uniform(5, 30, size=(n_obstacles, 1, 1)), (1, 1, 1))

    rgjs = [lf.PointRGJ(coordinates=c, repulsion=repulsions[i]) for i, c in enumerate(centers)]
    field = lf.RiskField(rgjs=rgjs, center_point=[0.0, 0.0], size=1600.0)

    resolution = 400
    xs = np.linspace(-700, 700, resolution)
    ys = np.linspace(-700, 700, resolution)
    xg, yg = np.meshgrid(xs, ys)
    grid_pts = np.stack([xg.ravel(), yg.ravel()], axis=1)

    def numpy_version():
        return field.eval(grid_pts)

    inv_reps = np.array([rgj.inv_repulsion for rgj in rgjs])

    def cython_version():
        return fc.point_obstacles_eval_max(grid_pts, centers, inv_reps)

    t_np = timeit(numpy_version, repeats=3)
    t_cy = timeit(cython_version, repeats=3)

    out_np = numpy_version()
    out_cy = cython_version()
    max_err = np.max(np.abs(out_np - out_cy))

    print(f"[RiskField.eval over Point obstacles] obstacles={n_obstacles} grid_points={len(grid_pts)} ({resolution}x{resolution})")
    print(f"  numpy (per-rgj python loop): {t_np*1000:8.2f} ms")
    print(f"  cython (fused kernel)      : {t_cy*1000:8.2f} ms   speedup: {t_np/t_cy:5.2f}x   max_abs_err={max_err:.2e}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Cython vs NumPy benchmark (larp.field_cy vs larp.field)")
    print("=" * 70)
    bench_polygon_repulsion()
    bench_point_in_polygon()
    bench_field_eval_point_obstacles()
