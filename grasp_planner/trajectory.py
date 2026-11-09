import numpy as np
from scipy.spatial.transform import Rotation
from .models import pose, transform, check


def timed_path(points, R, speed, accel, step):
    rows = []
    time0 = 0.0
    for a, b in zip(points[:-1], points[1:]):
        a = np.asarray(a)
        b = np.asarray(b)
        d = np.linalg.norm(b - a)
        duration = max(1.875 * d / speed, np.sqrt(5.774 * d / accel), 0.02)
        n = max(3, int(np.ceil(d / step)) * 2 + 1)
        for j, u in enumerate(np.linspace(0, 1, n)):
            if rows and j == 0:
                continue
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
            dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
            rows.append(
                {
                    "time_s": float(time0 + u * duration),
                    "position_world_m": (a + s * (b - a)).tolist(),
                    "velocity_world_m_s": (ds * (b - a)).tolist(),
                    "acceleration_world_m_s2": (dds * (b - a)).tolist(),
                }
            )
        time0 += duration
    return rows

