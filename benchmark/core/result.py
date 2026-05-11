"""SimulationResult: typed container for a single route-run outcome."""

from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class SimulationResult:
    city:              str
    scenario:          str
    algorithm:         str
    segment:           int
    nominal_speed:     float
    success:           bool  = False
    is_clear:          bool  = False
    crash_reason:      str   = ""
    avg_solve_time:    float = math.nan
    std_solve_time:    float = math.nan
    min_clearance:     float = math.nan
    ref_min_clearance: float = math.nan
    travel_time:       float = math.nan
    path_length:       float = math.nan
    converge_rate:     float = math.nan
    control_effort:    float = math.nan
    steps:             int   = 0
    replay_npz:        str   = ""
    city_meta:         dict  = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "City":              self.city,
            "Scenario":          self.scenario,
            "Algorithm":         self.algorithm,
            "Segment":           self.segment,
            "Nominal Speed":     self.nominal_speed,
            "Success":           self.success,
            "Is Clear":          self.is_clear,
            "Crash Reason":      self.crash_reason,
            "Avg Solve Time":    self.avg_solve_time,
            "Std Solve Time":    self.std_solve_time,
            "Min Clearance":     self.min_clearance,
            "Ref Min Clearance": self.ref_min_clearance,
            "Travel Time":       self.travel_time,
            "Path Length":       self.path_length,
            "Converge Rate":     self.converge_rate,
            "Control Effort":    self.control_effort,
            "Steps":             self.steps,
            "Replay NPZ":        self.replay_npz,
        }

    @classmethod
    def failure(cls, city: str, scenario: str, algorithm: str,
                segment: int, nominal_speed: float, reason: str) -> "SimulationResult":
        return cls(
            city=city, scenario=scenario, algorithm=algorithm,
            segment=segment, nominal_speed=nominal_speed,
            success=False, is_clear=False, crash_reason=reason,
        )