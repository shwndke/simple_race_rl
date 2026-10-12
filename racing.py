from __future__ import annotations

import ctypes as ct
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class VehicleParams:
    mass: float = 1200.0
    yaw_inertia: float = 1500.0
    front_axle: float = 1.2
    rear_axle: float = 1.4
    gravity: float = 9.81
    grip: float = 1.05
    front_stiffness: float = 65000.0
    rear_stiffness: float = 70000.0
    engine_force: float = 8000.0
    brake_force: float = 12000.0
    drag: float = 0.45
    rolling_drag: float = 35.0
    steering_limit: float = 0.55
    steering_rate: float = 3.5
    velocity_min: float = -20.0
    velocity_max: float = 80.0
    lateral_limit: float = 50.0
    yaw_limit: float = 8.0
    minimum_effective_speed: float = 0.5
    low_speed_damping: float = 0.9
    straight_epsilon: float = 1e-06
    speed_normalization: float = 80.0
    lateral_normalization: float = 50.0
    yaw_normalization: float = 8.0
    car_length: float = 5.2
    car_width: float = 2.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Vehicle parameters must be finite")
        for name,value in asdict(self).items():
            if name != "velocity_min" and value <= 0:
                raise ValueError(f"Invalid vehicle parameter {name}")
        if not 0 < self.steering_limit < math.pi/2:
            raise ValueError("Invalid steering limit")


def policy_state(checkpoint, config):
    if tuple(checkpoint.get("lookahead_distances", ())) != tuple(config.lookahead_distances):
        raise ValueError("Checkpoint lookahead distances do not match this environment")
    return checkpoint["agent"]


@dataclass(frozen=True, slots=True)
class RacingConfig:
    control_points: Sequence[tuple[float, float]]
    road_width: float
    lookahead_distances: Sequence[float]
    samples_per_span: int
    step_dt: float
    cell_size: float = 1.0
    substeps: int = 3
    vehicle: VehicleParams = VehicleParams()
    smoothing: Literal["none", "catmull_rom"] = "none"
    max_steps: int = 0

    @classmethod
    def from_track(cls, name: str, **settings):
        if name in ("", ".", "..") or "/" in name or "\\" in name:
            raise ValueError("Invalid track name")
        path = Path(__file__).resolve().parent / "tracks" / f"{name}.json"
        with path.open(encoding="utf-8") as file:
            track = json.load(file)
        points = np.asarray(track["points"], dtype=np.float64)
        width = float(track["road_width"])
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Track must contain at least three finite coordinate pairs")
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Track road width must be positive and finite")
        return cls(control_points=tuple(map(tuple, points)), road_width=width, **settings)


class _CVehicle(ct.Structure):
    _fields_ = [(name, ct.c_float) for name in VehicleParams.__dataclass_fields__]


class _CConfig(ct.Structure):
    _fields_ = [
        ("control_points", ct.POINTER(ct.c_float)),
        ("point_count", ct.c_int32),
        ("road_width", ct.c_float),
        ("lookahead_distances", ct.POINTER(ct.c_float)),
        ("lookahead_count", ct.c_int32),
        ("smoothing", ct.c_int32),
        ("samples_per_span", ct.c_int32),
        ("cell_size", ct.c_float),
        ("step_dt", ct.c_float),
        ("substeps", ct.c_int32),
        ("vehicle", _CVehicle),
        ("max_steps", ct.c_int32),
    ]


def load_library(library_path=None):
    lib = ct.CDLL(str(library_path or Path(__file__).parent / "build" / "racing.dll"))
    if hasattr(lib, "race_step_with_lap_events"):
        raise RuntimeError("Environment DLL uses the old step API; rebuild it before starting a new process")
    lib.race_create.argtypes = [ct.POINTER(_CConfig), ct.c_int32, ct.c_uint64]
    lib.race_create.restype = ct.c_void_p
    lib.race_observation_size.argtypes = [ct.c_void_p]
    lib.race_reset.argtypes = [ct.c_void_p, ct.c_int32, ct.POINTER(ct.c_float)]
    lib.race_step.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_float),
    ]
    lib.race_step.restype = None
    lib.race_destroy.argtypes = [ct.c_void_p]
    lib.race_destroy.restype = None
    return lib


def create_context(lib, config: RacingConfig, num_envs: int, seed: int):
    points = np.ascontiguousarray(config.control_points, dtype=np.float32)
    lookahead = np.ascontiguousarray(config.lookahead_distances, dtype=np.float32)
    assert points.ndim == 2 and points.shape[1] == 2
    assert lookahead.ndim == 1

    c_config = _CConfig(
        points.ctypes.data_as(ct.POINTER(ct.c_float)),
        len(points),
        config.road_width,
        lookahead.ctypes.data_as(ct.POINTER(ct.c_float)),
        len(lookahead),
        int(config.smoothing == "catmull_rom"),
        config.samples_per_span,
        config.cell_size,
        config.step_dt,
        config.substeps,
        _CVehicle(*(getattr(config.vehicle, name) for name, _ in _CVehicle._fields_)),
        config.max_steps,
    )
    env = lib.race_create(ct.byref(c_config), num_envs, seed)
    assert env
    return env


class RacingEnv:
    def __init__(self, config: RacingConfig, num_envs: int, seed: int = 0, library_path=None):
        if config.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.lib = load_library(library_path)
        self.env = create_context(self.lib, config, num_envs, seed)

        obs_size = self.lib.race_observation_size(self.env)
        self.action_buffer = (ct.c_float * (num_envs * 2))()
        self.obs_buffer = (ct.c_float * (num_envs * obs_size))()
        self.final_obs_buffer = (ct.c_float * (num_envs * obs_size))()
        self.reward_buffer = (ct.c_float * num_envs)()
        self.final_progress_buffer = (ct.c_float * num_envs)()
        self.terminated_buffer = (ct.c_uint8 * num_envs)()
        self.truncated_buffer = (ct.c_uint8 * num_envs)()
        self.lap_event_buffer = (ct.c_uint8 * num_envs)()
        self.lap_time_buffer = (ct.c_float * num_envs)()

        self.actions = np.frombuffer(self.action_buffer, np.float32).reshape(num_envs, 2)
        self.obs = np.frombuffer(self.obs_buffer, np.float32).reshape(num_envs, obs_size)
        self.final_obs = np.frombuffer(self.final_obs_buffer, np.float32).reshape(num_envs, obs_size)
        self.rewards = np.frombuffer(self.reward_buffer, np.float32)
        self.final_progress = np.frombuffer(self.final_progress_buffer, np.float32)
        self.terminated = np.frombuffer(self.terminated_buffer, np.uint8)
        self.truncated = np.frombuffer(self.truncated_buffer, np.uint8)
        self.lap_events = np.frombuffer(self.lap_event_buffer, np.uint8)
        self.lap_times = np.frombuffer(self.lap_time_buffer, np.float32)
        self.info = {
            "final_observations": self.final_obs,
            "final_progress": self.final_progress,
            "lap_events": self.lap_events,
            "lap_times": self.lap_times,
        }

    def reset(self):
        self.lib.race_reset(self.env, -1, self.obs_buffer)
        return self.obs

    def step(self):
        self.lib.race_step(
            self.env,
            self.action_buffer,
            self.obs_buffer,
            self.reward_buffer,
            self.terminated_buffer,
            self.truncated_buffer,
            self.final_obs_buffer,
            self.final_progress_buffer,
            self.lap_event_buffer,
            self.lap_time_buffer,
        )
        return self.obs, self.rewards, self.terminated, self.truncated, self.info

    def close(self):
        if self.env:
            self.lib.race_destroy(self.env)
            self.env = None
