"""
larp/env/environments.py
========================
Environment ABC, CityEnvironment, FieldHeatmapEnvironment.

CityEnvironment data layers (bottom-to-top)
--------------------------------------------
water → beaches → green space → roads → buildings

All rendered as PatchCollection / LineCollection for blit-safe background
rasterisation (~0.37 ms/frame blitted vs 84 ms/frame individual patches).
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.transforms as mtransforms
from larp.const import OSM_INSTALLED

if OSM_INSTALLED:
    import geopandas as gpd
    from shapely.geometry import Point
    import osmnx as ox
    import pandas as pd

from larp.environment.twin import hex_to_rgb, interpolate_colors


# ── palette ──────────────────────────────────────────────────────────────────
_WATER_FILL  = "#85c1e9"
_WATER_GLOW  = "#d6eaf8"
_WATER_EDGE  = "#2471a3"
_WATER_WAVE  = "#2e86c1"
_PARK_FILL   = "#c8e6c9"
_PARK_EDGE   = "#81c784"
_SAND_FILL   = "#F3E2C8"
_ROAD_FILL   = "#e0e0e0"
_MAP_BG      = "#f0ede8"
_OBS_FILL_LO = "#575757"
_OBS_FILL_HI = "#3d3127"
_OBS_EDGE    = "#e74c3c"
_BG_FILL_LO  = "#8d8d8d"
_BG_FILL_HI  = "#f5f5f5"

# Default road widths in metres when OSM data has no explicit width tag.
_HIGHWAY_WIDTHS: Dict[str, float] = {
    "motorway": 18.0,      "trunk": 16.0,
    "primary": 14.0,       "secondary": 12.0,   "tertiary": 10.0,
    "motorway_link": 10.0, "trunk_link": 9.0,
    "primary_link": 8.0,   "secondary_link": 7.0, "tertiary_link": 7.0,
    "residential": 7.0,    "unclassified": 6.0,
    "living_street": 6.0,  "service": 4.5,        "road": 6.0,
    "track": 3.0,          "bus_guideway": 6.0,   "escape": 5.0,
    "raceway": 10.0,       "pedestrian": 6.0,
    "path": 2.0,           "footway": 2.0,        "cycleway": 2.5,
    "bridleway": 2.5,      "steps": 1.5,          "corridor": 2.0,
}


class Environment(ABC):
    """
    Abstract base for all larp environments.

    Contract
    --------
    draw(ax)            → draw onto ax, return layer dict
    get_obstacle_rgjs() → list of RGJ dicts for larp.RiskField
    update(**kwargs)    → optional dynamic hook (altitude change, …)
    """
    @abstractmethod
    def draw(self, ax: plt.Axes, patch_store: Optional[List] = None) -> Dict:
        """
        Draw environment background onto *ax*.

        Returns a dict of named layer artists.  If *patch_store* is given,
        all created artists are appended to it for backward compatibility.
        """
        pass

    @abstractmethod
    def get_obstacle_rgjs(self) -> list: ...

    def update(self, **kwargs) -> None:
        pass


class CityEnvironment(Environment):
    """
    OpenStreetMap urban environment with efficient batch rendering.

    Parameters
    ----------
    location          : (lat, lon) tuple or place-name string
    altitude          : float   Drone flight altitude [m]
    dist              : float   OSM tile radius [m]
    safety_margin     : float   Hazard-zone buffer around obstacles [m]
    default_repulsion : list    2x2 repulsion matrix for RGJ export
    load_roads        : bool    Download and render roads (default True)
    load_green        : bool    Download and render parks/green space (default True)
    water_waves       : bool    Overlay wave texture on water bodies (default True)
    """

    # Regions where OSM bay / coastline tags are noisy.
    # Format: (lat, lon, radius_km).
    _RESTRICTED_WATER_ZONES: List[Tuple[float, float, float]] = [
        (22.3193, 114.1694, 50.0),  # Hong Kong
        (35.6892, 139.6917, 30.0),  # Tokyo Bay
        (42.3593, -71.0799, 1200.0) # Boston
    ]

    def __init__(
        self,
        location: tuple | str = (1.305602, 103.836688),
        altitude: float = 50.0,
        dist: float = 400.0,
        safety_margin: float = 0.0,
        default_repulsion: Optional[list] = None,
        load_roads: bool = True,
        load_green: bool = True,
        water_waves: bool = False,
    ):
        if not OSM_INSTALLED:
            raise ImportError("`pip install osmnx` to use CityEnvironment")

        self.location_query    = location
        self.altitude          = altitude
        self.dist              = dist
        self.safety_margin     = safety_margin
        self.default_repulsion = default_repulsion or [[20.0, 0], [0, 20.0]]
        self.load_roads        = load_roads
        self.load_green        = load_green
        self.water_waves       = water_waves

        self.buildings:    Optional[object] = None
        self.water:        Optional[object] = None
        self.green:        Optional[object] = None
        self.beaches:      Optional[object] = None
        self.roads:        Optional[object] = None
        self.map_center_x: float = 0.0
        self.map_center_y: float = 0.0
        self.map_radius:   float = dist*1.4  #needed?
        self._utm_crs      = None
        self._center_point: Optional[Tuple[float, float]] = None
        self.map_clip      = mtransforms.Bbox.from_extents(-self.dist, -self.dist, self.dist, self.dist)


        ox.settings.use_cache   = True
        ox.settings.log_console = False
        print(f"Loading OSM data for {location} ...")
        self._load()
        self._classify(altitude)

    # Restricted-zone guard

    @staticmethod
    def _haversine_km(lat1: float, lon1: float,
                      lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Vectorised great-circle distance (kilometres)."""
        R = 6371.0
        la1, lo1 = np.radians(lat1), np.radians(lon1)
        la2, lo2 = np.radians(np.asarray(lat2)), np.radians(np.asarray(lon2))
        dlat, dlon = la2 - la1, lo2 - lo1
        a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    def _in_restricted_zone(self) -> bool:
        if not self._center_point or not self._RESTRICTED_WATER_ZONES:
            return False
        zones = np.array(self._RESTRICTED_WATER_ZONES)
        dists = self._haversine_km(
            self._center_point[0], self._center_point[1],
            zones[:, 0], zones[:, 1],
        )
        return bool(np.any(dists <= zones[:, 2]))

    # OSM loading

    def _load(self):
        cp = (ox.geocode(self.location_query)
              if isinstance(self.location_query, str)
              else self.location_query)
        self._center_point = cp

        # Buildings
        print("Loading building...")
        btags = {"building": True, "building:levels": True,
                 "building:part": True, "structure": True}
        gdf = ox.features_from_point(cp, tags=btags, dist=self.dist)
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        if "location" in gdf.columns:
            gdf = gdf[gdf["location"] != "underground"]

        utm = gdf.estimate_utm_crs()
        self._utm_crs = utm
        gdf = gdf.to_crs(utm)
        gdf["calc_height"] = gdf.apply(self._extract_height, axis=1)

        cgs = gpd.GeoSeries([Point(cp[1], cp[0])], crs="EPSG:4326").to_crs(utm)
        self.map_center_x = float(cgs[0].x)
        self.map_center_y = float(cgs[0].y)

        gdf["center_dist"] = np.hypot(
            gdf.geometry.centroid.x - self.map_center_x,
            gdf.geometry.centroid.y - self.map_center_y,
        )
        self.buildings = gdf

        # Water
        print("Loading water...")
        natural_water = ["water", "wetland"] # "coastline", "sea"
        if not self._in_restricted_zone():
            natural_water += ["bay"]

        wtags = {
            "natural":  natural_water,
            "place":    [], # "ocean", "sea"
            "waterway": ["riverbank", "dock", "canal"], 
            "landuse":  ["reservoir", "basin", "salt_pond"],
            "leisure":  ["marina"],
        }
        try:
            wgdf = ox.features_from_point(cp, tags=wtags, dist=self.dist)
            wgdf = wgdf[wgdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not wgdf.empty:
                for col, vals in [("tunnel",   ["yes", "culvert"]),
                                   ("covered",  ["yes"]),
                                   ("location", ["underground"])]:
                    if col in wgdf.columns:
                        wgdf = wgdf[~wgdf[col].isin(vals)]
                self.water = wgdf.to_crs(utm) if not wgdf.empty else None
        except Exception as e:
            print(f"  Water: {e}"); self.water = None

        # Nature (green + beaches)
        print("Loading nature...")
        ntags = {
            "natural": ["beach", "sand", "dune", "wood", "scrub",
                        "heath", "grassland"],
            "leisure": ["park", "garden", "nature_reserve", "golf_course",
                        "recreation_ground", "beach_resort"],
            "landuse": ["grass", "forest", "meadow", "village_green"],
        }
        if self.load_green:
            try:
                ngdf = ox.features_from_point(cp, tags=ntags, dist=self.dist)
                if not ngdf.empty:
                    ngdf = (ngdf[ngdf.geometry.type.isin(
                                ["Polygon", "MultiPolygon"])].copy()
                            .to_crs(utm))
                    is_sand = (
                        ngdf.get("natural", pd.Series(dtype=str))
                            .isin(["beach", "sand", "dune"])
                        | (ngdf.get("leisure", pd.Series(dtype=str)) == "beach_resort")
                    )
                    self.beaches = ngdf[is_sand].copy() if is_sand.any() else None
                    self.green   = ngdf[~is_sand].copy()
                    if len(self.green) == 0:
                        self.green = None
            except Exception as e:
                print(f"  Green/Beach: {e}")

        # Roads
        print("Loading roads...")
        if self.load_roads:
            try:
                G = ox.graph_from_point(cp, dist=self.dist,
                                        network_type="all", simplify=True, truncate_by_edge=True)
                self._impute_road_widths(G)
                self.roads = ox.graph_to_gdfs(G, nodes=False).to_crs(utm)
            except Exception as e:
                print(f"  Roads: {e}"); self.roads = None

    @staticmethod
    def _impute_road_widths(G) -> None:
        """Set a ``width`` attribute on every graph edge that lacks one."""
        for _u, _v, _k, data in G.edges(data=True, keys=True):
            if "width" not in data:
                hw = data.get("highway", "")
                hw = hw[0] if isinstance(hw, list) and hw else str(hw)
                data["width"] = _HIGHWAY_WIDTHS.get(hw, 4.0)

    @staticmethod
    def _extract_height(row) -> float:
        """Extract building height from OSM tags; fall back to 12 m."""
        val = row.get("height")
        if pd.notna(val):
            m = re.findall(r"[-+]?\d*\.?\d+", str(val))
            if m: return float(m[0])
        for col in ("building:levels", "levels"):
            lvl = row.get(col)
            if pd.notna(lvl):
                m = re.findall(r"[-+]?\d*\.?\d+", str(lvl))
                if m: return float(m[0]) * 3.5
        return 12.0

    # Building classification

    def _classify(self, altitude: float):
        self.altitude = altitude
        if self.buildings is None or self.buildings.empty:
            return
        df = self.buildings
        df["delta_h"]     = df["calc_height"] - altitude
        df["is_obstacle"] = df["delta_h"] >= 0

        # Obstacle colour: dark brown-grey gradient by how far above altitude
        c_obs_lo = hex_to_rgb(_OBS_FILL_LO)
        c_obs_hi = hex_to_rgb(_OBS_FILL_HI)
        # Background colour: mid-grey fading to near-white as building shrinks below
        c_bg_lo  = hex_to_rgb(_BG_FILL_LO)
        c_bg_hi  = hex_to_rgb(_BG_FILL_HI)

        obs  = df["is_obstacle"].values
        rgba = np.zeros((len(df), 4))

        t_obs = np.clip( df.loc[obs,  "delta_h"] /  150.0, 0, 1).values
        t_bg  = np.clip(-df.loc[~obs, "delta_h"] /  60.0, 0, 1).values
        rgba[obs,  :3] = interpolate_colors(c_obs_lo, c_obs_hi, t_obs[:, None])
        rgba[~obs, :3] = interpolate_colors(c_bg_lo,  c_bg_hi,  t_bg[:, None])

        # Alpha: obstacles slightly more opaque; distant background fades
        # d_fades = np.clip(1.0 - df["center_dist"].values / self.map_radius, 0.2, 1.0)
        # rgba[obs,  3] = (0.55 + 0.35 * t_obs) * np.clip(0.3 + 0.7 * d_fades[obs], 0, 1)
        # rgba[~obs, 3] = 0.30 * d_fades[~obs]

        rgba[obs,  3] = (0.55 + 0.35 * t_obs)
        rgba[~obs, 3] = 0.30

        df["rgba"]       = list(rgba)
        df["edge_color"] = np.where(obs, _OBS_EDGE, "#c8c8c8")
        df["edge_width"] = np.where(obs, 1.5, 0.3)
        df["zorder"]     = df["calc_height"] / 1000.0 + np.where(obs, 100, 1)

    def update_altitude(self, new_altitude: float) -> None:
        """Reclassify buildings against a new altitude; call invalidate() to redraw."""
        self._classify(new_altitude)

    def update(self, altitude: Optional[float] = None, **kwargs) -> None:
        if altitude is not None:
            self.update_altitude(altitude)

    # Draw helpers

    def _iter_poly_pts(self, geom):
        """Yield normalised Nx2 exterior arrays for each polygon in *geom*."""
        if geom.is_empty: return
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                yield np.column_stack((np.array(x) - self.map_center_x, np.array(y) - self.map_center_y))

    def _iter_line_segs(self, geom):
        """Yield normalised segment lists for LineString / MultiLineString."""
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for g in parts:
            yield [(c[0] - self.map_center_x, c[1] - self.map_center_y) for c in np.array(g.coords)]

    def draw(self, ax: plt.Axes, patch_store: Optional[List] = None) -> Dict:
        """Draw all city layers onto *ax* and return a dict of named artists."""
        layers: Dict = {}
        ax.set_facecolor(_MAP_BG)

        clip_box = mtransforms.Bbox.from_extents(-self.dist, -self.dist, self.dist, self.dist)
        self.map_clip = mtransforms.TransformedBbox(clip_box, ax.transData)

        self.clip_patch = mpatches.Rectangle(
            (-self.dist, -self.dist), self.dist * 2, self.dist * 2,
            facecolor=_MAP_BG, edgecolor="#2871f8", zorder=0
        )
        ax.add_patch(self.clip_patch)
        layers["map_bounds"] = self.clip_patch

        layers.update(self._draw_water(ax))
        layers.update(self._draw_beaches(ax))
        layers.update(self._draw_green(ax))
        layers.update(self._draw_roads(ax))
        layers.update(self._draw_buildings(ax))
        
        if patch_store is not None:
            for v in layers.values():
                patch_store.extend(v) if isinstance(v, list) else patch_store.append(v)
        return layers

    def _draw_water(self, ax) -> Dict:
        if self.water is None: return {}
        fill_patches, glow_patches, wave_segs = [], [], []
        
        for geom in self.water.geometry:
            if geom.is_empty: continue
            for pts in self._iter_poly_pts(geom):
                fill_patches.append(mpatches.Polygon(pts, closed=True))
                # Inner glow (85% scale toward centroid)
                ctr = pts.mean(axis=0)
                glow_patches.append(mpatches.Polygon(ctr + (pts - ctr) * 0.85, closed=True))
                
                if self.water_waves:
                    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
                    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
                    for wy in np.arange(ymin + 3, ymax, 7):
                        xs = np.linspace(xmin + 1, xmax - 1, 40)
                        wave_segs.append(list(zip(xs, wy + 1.2 * np.sin(xs * 0.25 + wy * 0.1))))
                            
        layers: Dict = {}
        if fill_patches:
            pc = PatchCollection(fill_patches, facecolors=_WATER_FILL, edgecolors=_WATER_EDGE, linewidths=1.8, alpha=0.85, zorder=1)
            pc.set_clip_path(self.clip_patch)
            pc.set_clip_on(True)
            ax.add_collection(pc); layers["water_fill"] = pc
        if glow_patches:
            gc = PatchCollection(glow_patches, facecolors=_WATER_GLOW, edgecolors="none", alpha=0.45, zorder=1.1)
            ax.add_collection(gc); layers["water_glow"] = gc
        # if wave_segs:
        #     wc = LineCollection(wave_segs, colors=_WATER_WAVE, linewidths=0.35, alpha=0.30, zorder=1.2)
        #     ax.add_collection(wc); layers["water_waves"] = wc
        return layers

    def _draw_beaches(self, ax) -> Dict:
        if self.beaches is None: return {}
        patches = [mpatches.Polygon(pts, closed=True) for geom in self.beaches.geometry for pts in self._iter_poly_pts(geom)]
        if not patches: return {}
        pc = PatchCollection(patches, facecolors=_SAND_FILL, edgecolors="none", alpha=0.75, zorder=1.5)
        pc.set_clip_path(self.clip_patch)
        pc.set_clip_on(True)
        ax.add_collection(pc)
        return {"beaches": pc}

    def _draw_green(self, ax) -> Dict:
        if self.green is None: return {}
        patches = [mpatches.Polygon(pts, closed=True) for geom in self.green.geometry for pts in self._iter_poly_pts(geom)]
        if not patches: return {}
        pc = PatchCollection(patches, facecolors=_PARK_FILL, edgecolors=_PARK_EDGE, linewidths=0.5, alpha=0.55, zorder=2)
        pc.set_clip_path(self.clip_patch)
        pc.set_clip_on(True)
        ax.add_collection(pc)
        return {"green": pc}

    def _draw_roads(self, ax) -> Dict:
        """
        Render roads as physically-accurate buffered PatchCollection polygons.
        Each edge is buffered by half its imputed width so road surfaces are
        geometrically correct rather than uniform-width cosmetic lines.
        """
        if self.roads is None: return {}
        road_patches = []
        
        for _, row in self.roads.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            w_val = row.get("width", 4.0)
            
            try:
                if isinstance(w_val, list): w = float(w_val[0])
                elif isinstance(w_val, str):
                    nums = re.findall(r"[-+]?\d*\.?\d+", w_val)
                    w = float(nums[0]) if nums else 4.0
                else: w = float(w_val)
            except (ValueError, TypeError, IndexError):
                w = 4.0
                
            poly_geom = geom.buffer(w / 2, cap_style=2, join_style=2)
            polys = list(poly_geom.geoms) if poly_geom.geom_type == "MultiPolygon" else [poly_geom]
            
            for p in polys:
                if p.geom_type == "Polygon":
                    xp, yp = p.exterior.xy
                    pts = np.column_stack((np.array(xp) - self.map_center_x, np.array(yp) - self.map_center_y))
                    road_patches.append(mpatches.Polygon(pts, closed=True))
                    
        if not road_patches: return {}
        pc = PatchCollection(road_patches, facecolors=_ROAD_FILL, edgecolors="none", alpha=0.90, zorder=2.5, antialiased=True)
        pc.set_clip_path(self.clip_patch)
        pc.set_clip_on(True)
        ax.add_collection(pc)
        return {"roads": pc}

    def _draw_buildings(self, ax) -> Dict:
        if self.buildings is None: return {}
        obs_patches, obs_colors, obs_ec = [], [], []
        bg_patches, bg_colors, bg_ec = [], [], []
        
        for _, row in self.buildings.iterrows():
            geom = row.geometry
            if geom.is_empty: continue
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            
            for poly in polys:
                x, y = poly.exterior.xy
                pts = np.column_stack((np.array(x) - self.map_center_x, np.array(y) - self.map_center_y))
                p = mpatches.Polygon(pts, closed=True)
                
                if row["is_obstacle"]:
                    obs_patches.append(p); obs_colors.append(row["rgba"]); obs_ec.append(row["edge_color"])
                else:
                    bg_patches.append(p); bg_colors.append(row["rgba"]); bg_ec.append(row["edge_color"])

                # Safety margin border (separate thin collection)
                if row["is_obstacle"] and self.safety_margin > 0:
                    bp = poly.buffer(self.safety_margin, resolution=2)
                    bx, by = bp.exterior.xy
                    bpts = np.column_stack((np.array(bx) - self.map_center_x, np.array(by) - self.map_center_y))
                    obs_patches.append(mpatches.Polygon(bpts, closed=True))
                    obs_colors.append((0, 0, 0, 0)); obs_ec.append(_OBS_EDGE)

        layers: Dict = {}
        if bg_patches:
            pc = PatchCollection(bg_patches, facecolors=bg_colors, edgecolors=bg_ec, linewidths=0.3, zorder=8)
            pc.set_clip_path(self.clip_patch)
            pc.set_clip_on(True)
            ax.add_collection(pc); layers["buildings_bg"] = pc
        if obs_patches:
            pc = PatchCollection(obs_patches, facecolors=obs_colors, edgecolors=obs_ec, linewidths=1.5, zorder=11)
            pc.set_clip_path(self.clip_patch)
            pc.set_clip_on(True)
            ax.add_collection(pc); layers["buildings_obs"] = pc
        return layers

    # RGJ export
    def get_obstacle_rgjs(self) -> list:
        """Export obstacle buildings as RGJ polygon dicts (origin-centred)."""
        if self.buildings is None: return []
        rgjs = []
        for _, row in self.buildings[self.buildings["is_obstacle"]].iterrows():
            geom = row.geometry
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                x, y = poly.convex_hull.exterior.xy
                pts = np.column_stack((np.array(x) - self.map_center_x, np.array(y) - self.map_center_y))
                if not np.allclose(pts[0], pts[-1]): pts = np.vstack([pts, pts[0]])
                rgjs.append({
                    "type": "Polygon",
                    "coordinates": pts.tolist(),
                    "repulsion": self.default_repulsion,
                })
        return rgjs

    @property
    def obstacle_count(self) -> int:
        return int(self.buildings["is_obstacle"].sum()) if self.buildings is not None else 0


class FieldHeatmapEnvironment(Environment):
    """Renders a larp.RiskField as a background heatmap."""

    def __init__(self, field, resolution: int = 900):
        self.field      = field
        self.resolution = resolution

    def draw(self, ax: plt.Axes, patch_store: Optional[List] = None) -> Dict:
        display, extent = self.field.to_image(resolution=self.resolution, return_extent=True)
        img = ax.imshow(display, cmap="jet", extent=extent)
        img.set_clim(0.0, 1.0)
        if patch_store is not None: patch_store.append(img)
        return {"field": img}

    def get_obstacle_rgjs(self) -> list:
        return list(getattr(self.field, "rgjs", []))