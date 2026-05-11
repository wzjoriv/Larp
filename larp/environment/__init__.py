"""
larp/env  —  Robot + Environment + Visualiser
=============================================
Backwards-compatible re-exports: code that did
``from larp.env import QuadcopterPerspective, ZoomedCityVisualizer``
continues to work unchanged.

New structure
-------------
larp/env/robots.py          Robot ABC, WMR, Car, Quadcopter, QuadcopterPerspective
larp/env/environments.py    Environment ABC, CityEnvironment, FieldHeatmapEnvironment
larp/env/visualizers.py     Visualizer, FieldTrajectoryVisualizer,
                            CityVisualizer, ZoomedCityVisualizer
"""

from larp.environment.twin import (
    Robot,
    WMR,
    Car,
    Quadcopter,
    QuadcopterPerspective,
    SmallFixedWing,
    SmallFixedWingPerspective,
    hex_to_rgb,
    interpolate_colors,
)

from larp.environment.environments import (
    Environment,
    CityEnvironment,
    FieldHeatmapEnvironment,
)

from larp.environment.visualizers import (
    Visualizer,
    FieldTrajectoryVisualizer,
    CityVisualizer,
    ZoomedCityVisualizer,
)

__all__ = [
    # robots
    "Robot", "WMR", "Car", "Quadcopter", "QuadcopterPerspective", "SmallFixedWing", "SmallFixedWingPerspective",
    "hex_to_rgb", "interpolate_colors",
    # environments
    "Environment", "CityEnvironment", "FieldHeatmapEnvironment",
    # visualisers
    "Visualizer", "FieldTrajectoryVisualizer", "CityVisualizer", "ZoomedCityVisualizer",
]
