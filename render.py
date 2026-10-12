"""Live driving, policy playback, and episode videos."""

import argparse
import ctypes as ct
import math
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import moderngl
import pygame
from pygame._sdl2 import controller

from racing import RacingConfig, RacingEnv, VehicleParams, _CConfig


class CarState(ct.Structure):
    _fields_ = [
        ("x", ct.c_float), ("y", ct.c_float), ("heading", ct.c_float),
        ("forward_speed", ct.c_float), ("lateral_speed", ct.c_float),
        ("yaw_rate", ct.c_float), ("steering_angle", ct.c_float),
        ("last_s", ct.c_float), ("progress", ct.c_float), ("steps", ct.c_int32),
        ("next_lap_progress", ct.c_float), ("last_lap_step", ct.c_int32),
        ("front_slip", ct.c_float), ("rear_slip", ct.c_float),
    ]


class GridCell(ct.Structure):
    _fields_ = [
        ("segment", ct.c_int32), ("row", ct.c_int32),
        ("progress", ct.c_float), ("best_distance2", ct.c_float),
    ]


class CompiledMap(ct.Structure):
    _fields_ = [
        ("segments", ct.POINTER(ct.c_float)), ("normals", ct.POINTER(ct.c_float)),
        ("segment_count", ct.c_int32), ("lap_length", ct.c_float),
        ("cells", ct.POINTER(GridCell)),
        ("grid_width", ct.c_int32), ("grid_height", ct.c_int32),
        ("grid_x", ct.c_float), ("grid_y", ct.c_float), ("cell_size", ct.c_float),
        ("lookahead", ct.POINTER(ct.c_float)), ("on_road_count", ct.c_int32),
    ]


class _CEnv(ct.Structure):
    # Only the prefix through map is needed; keep it aligned with racing.c.
    _fields_ = [
        ("config", _CConfig), ("num_envs", ct.c_int32),
        ("cars", ct.POINTER(CarState)), ("map", CompiledMap),
    ]


class RenderEnv(RacingEnv):
    """Single-car environment with shared views of its C state and map."""

    def __init__(self, config, seed=1):
        super().__init__(config, 1, seed)
        self.config = config
        self.c_env = _CEnv.from_address(self.env)
        assert self.c_env.num_envs == 1
        dtype = np.dtype([
            (name, np.float32 if kind == ct.c_float else np.float64 if kind == ct.c_double else np.int32)
            for name, kind in CarState._fields_
        ])
        assert dtype.itemsize == ct.sizeof(CarState)
        self.states = np.frombuffer(self.c_env.cars.contents, dtype=dtype)
        self.states.flags.writeable = False
        count = self.c_env.map.segment_count
        assert count == len(config.control_points) * (config.samples_per_span if config.smoothing == "catmull_rom" else 1)
        segments = np.ctypeslib.as_array(self.c_env.map.segments, (count * 6,)).reshape(count, 6)
        normals = np.ctypeslib.as_array(self.c_env.map.normals, (count * 2,)).reshape(count, 2)
        # x, y, normal_x, normal_y, distance along lap, segment length.
        self.track = np.column_stack((segments[:, :2], normals, segments[:, 5], segments[:, 4]))
        self.lap_length = self.c_env.map.lap_length
        self.lookahead = np.ctypeslib.as_array(self.c_env.map.lookahead,
            (self.c_env.map.on_road_count * len(config.lookahead_distances) * 4,)).reshape(-1, len(config.lookahead_distances), 2, 2)
        self.lookahead.flags.writeable = False

    def reset(self):
        self.terminated[:] = 0
        self.truncated[:] = 0
        self.actions[:] = 0
        return super().reset()

    def targets(self):
        road = self.c_env.map
        x = (self.states[0]["x"] - road.grid_x) / road.cell_size
        y = (self.states[0]["y"] - road.grid_y) / road.cell_size
        if 0 <= x < road.grid_width and 0 <= y < road.grid_height:
            row = road.cells[int(y) * road.grid_width + int(x)].row
            if row >= 0:
                return self.lookahead[row]
        return np.full((len(self.config.lookahead_distances), 2, 2), np.nan, np.float32)


def interpolate(before, after, fraction):
    state = after.copy()
    for name in state.dtype.names:
        if state.dtype[name].kind == "f":
            state[name] = before[name] + (after[name] - before[name]) * fraction
    angle = (float(after["heading"]) - float(before["heading"]) + math.pi) % (2 * math.pi) - math.pi
    state["heading"] = float(before["heading"]) + angle * fraction
    state["steps"] = before["steps"] if fraction < 1 else after["steps"]
    return state


class Renderer:
    def __init__(self, track, road_width, width=1280, height=720, zoom=6, follow=True, window=True, step_dt=0.05,
                 car_length=VehicleParams().car_length, car_width=VehicleParams().car_width):
        assert width > 0 and height > 0 and zoom > 0 and road_width > 0
        self.width, self.height = width, height
        self.road_width = road_width
        self.zoom, self.follow, self.window = zoom, follow, window
        self.step_dt = step_dt
        self.trail = False
        self.lookahead = False
        self.agent_view = False
        self.history = deque(maxlen=180)
        self.lap_length = float(track[-1, 4] + track[-1, 5])
        self.center = (track[:, :2].min(0) + track[:, :2].max(0)) / 2
        size = np.ptp(track[:, :2], axis=0) + road_width * 2
        self.scale = min(width / size[0], height / size[1]) * 0.92

        pygame.font.init()
        if window:
            pygame.display.init()
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
            pygame.display.set_caption("Racing — WASD/arrows · R reset · P pause · F camera · wheel zoom · T trail · G lookahead · V agent view")
            self.ctx = moderngl.create_context(require=330)
        else:
            self.ctx = moderngl.create_standalone_context(require=330)
        self.framebuffer = self.ctx.simple_framebuffer((width, height), components=3)
        self.framebuffer.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.road_program = self.ctx.program(vertex_shader="""
            #version 330
            in vec2 position;
            in vec3 color;
            in float distance;
            uniform vec4 camera;
            out vec3 rgb;
            out float s;
            void main() {
                gl_Position = vec4(position * camera.xy + camera.zw, 0, 1);
                rgb = color; s = distance;
            }
        """, fragment_shader="""
            #version 330
            in vec3 rgb;
            in float s;
            out vec4 pixel;
            void main() {
                vec3 color = s < 0 ? rgb : (mod(s, 6) < 3 ?
                    vec3(204, 31, 31) / 255 : vec3(245, 245, 238) / 255);
                pixel = vec4(color, 1);
            }
        """)
        meshes = []
        points, normals = track[:, :2], track[:, 2:4]
        half = road_width / 2
        curb = min(road_width * 0.09, 1.25)
        for low, high, kerb in [(-half, half, False), (-half - curb, -half, True), (half, half + curb, True)]:
            a, b = points + normals * low, points + normals * high
            vertices = np.zeros((len(track), 6, 6), np.float32)
            vertices[:, :, :2] = np.stack((a, b, np.roll(a, -1, 0), np.roll(a, -1, 0), b, np.roll(b, -1, 0)), axis=1)
            vertices[:, :, 2:5] = 96 / 255
            start, end = track[:, 4], track[:, 4] + track[:, 5]
            vertices[:, :, 5] = np.stack((start, start, end, end, start, end), axis=1) if kerb else -1
            meshes.append(vertices.reshape(-1, 6))

        tangent = points[1] - points[0]
        tangent /= np.linalg.norm(tangent)
        columns = math.ceil(road_width)
        for row in range(2):
            for col in range(columns):
                a = points[0] + normals[0] * (-half + col * road_width / columns) + tangent * (row - 1)
                b = a + normals[0] * road_width / columns
                c, d = a + tangent, b + tangent
                vertices = np.zeros((6, 6), np.float32)
                vertices[:, :2] = (a, b, c, c, b, d)
                vertices[:, 2:5] = 0.05 if (row + col) % 2 else 0.96
                vertices[:, 5] = -1
                meshes.append(vertices)
        self.road_buffer = self.ctx.buffer(np.concatenate(meshes).tobytes())
        self.road = self.ctx.simple_vertex_array(self.road_program, self.road_buffer, "position", "color", "distance")
        self.line_buffer = self.ctx.buffer(reserve=180 * 6 * 4)
        self.line = self.ctx.simple_vertex_array(self.road_program, self.line_buffer, "position", "color", "distance")
        self.marker_buffer = self.ctx.buffer(reserve=256 * 6 * 4)
        self.markers = self.ctx.simple_vertex_array(self.road_program, self.marker_buffer, "position", "color", "distance")

        self.sprite_program = self.ctx.program(vertex_shader="""
            #version 330
            in vec2 position;
            in vec2 texcoord;
            uniform vec4 camera;
            uniform vec2 center;
            uniform vec2 rotation;
            out vec2 uv;
            void main() {
                vec2 p = center + vec2(rotation.x * position.x - rotation.y * position.y,
                                      rotation.y * position.x + rotation.x * position.y);
                gl_Position = vec4(p * camera.xy + camera.zw, 0, 1); uv = texcoord;
            }
        """, fragment_shader="""
            #version 330
            uniform sampler2D image;
            in vec2 uv;
            out vec4 pixel;
            void main() { pixel = texture(image, uv); }
        """)
        sprite = pygame.image.load(str(Path(__file__).parent / "assets" / "race_car_sprite.png"))
        self.texture = self.ctx.texture(sprite.get_size(), 4, pygame.image.tobytes(sprite, "RGBA"))
        self.texture.build_mipmaps()
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        length = car_length
        self.car_buffer = self.ctx.buffer(np.array([
            (-length / 2, -car_width / 2, 0, 1), (length / 2, -car_width / 2, 1, 1),
            (-length / 2, car_width / 2, 0, 0), (length / 2, car_width / 2, 1, 0),
        ], np.float32).tobytes())
        self.car = self.ctx.simple_vertex_array(self.sprite_program, self.car_buffer, "position", "texcoord")
        self.hud_surface = pygame.Surface((320, 100), pygame.SRCALPHA)
        self.hud_texture = self.ctx.texture((320, 100), 4)
        self.hud_buffer = self.ctx.buffer(np.array([
            (0, 0, 0, 0), (320, 0, 1, 0), (0, 100, 0, 1), (320, 100, 1, 1),
        ], np.float32).tobytes())
        self.hud = self.ctx.simple_vertex_array(self.sprite_program, self.hud_buffer, "position", "texcoord")
        self.font = pygame.font.SysFont("consolas", 18, bold=True)
        self.pixels = bytearray(width * height * 3)

    def draw(self, state, action, status="", targets=None):
        self.framebuffer.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        background = (0.95, 0.96, 0.97) if self.agent_view else (72 / 255, 132 / 255, 73 / 255)
        self.ctx.clear(*background)
        position = float(state["x"]), float(state["y"])
        angle = float(state["heading"])
        center = position if self.follow else self.center
        scale = self.scale * (self.zoom if self.follow else 1)
        if self.agent_view:
            if targets is not None:
                relative = np.asarray(targets) - position
                c, s = math.cos(angle), math.sin(angle)
                targets = np.stack((s * relative[..., 0] - c * relative[..., 1],
                                    c * relative[..., 0] + s * relative[..., 1]), axis=-1)
            position, center, angle = (0, 0), (0, 0), math.pi / 2
            if targets is not None:
                points = targets.reshape(-1, 2)
                points = points[np.isfinite(points).all(1)]
                if len(points):
                    low, high = np.minimum(points.min(0), 0), np.maximum(points.max(0), 0)
                    center = (low + high) / 2
                    size = high - low + self.road_width * 2
                    scale = min(self.width / size[0], self.height / size[1]) * 0.92 * self.zoom / 6
        camera = (2 * scale / self.width, 2 * scale / self.height,
                  -2 * scale * center[0] / self.width, -2 * scale * center[1] / self.height)
        self.road_program["camera"].value = camera
        if not self.agent_view:
            self.road.render()
        if self.trail and not self.agent_view:
            if not self.history or position != self.history[-1]:
                self.history.append(position)
            vertices = np.zeros((len(self.history), 6), np.float32)
            vertices[:, :2] = self.history
            vertices[:, 2:5] = (56 / 255, 161 / 255, 105 / 255)
            vertices[:, 5] = -1
            self.line_buffer.write(vertices.tobytes())
            self.line.render(moderngl.LINE_STRIP, vertices=len(vertices))
        if (self.lookahead or self.agent_view) and targets is not None:
            if self.agent_view:
                groups = [(targets[:, 0], (0.1, 0.55, 1)), (targets[:, 1], (1, 0.45, 0.15))]
            else:
                groups = [(np.asarray(targets).reshape(-1, 2), (1, 231 / 255, 74 / 255))]
            for points, color in groups:
                points = points[np.isfinite(points).all(1)]
                assert len(points) <= 256
                if len(points):
                    vertices = np.zeros((len(points), 6), np.float32)
                    vertices[:, :2] = points
                    vertices[:, 2:5] = color
                    vertices[:, 5] = -1
                    self.marker_buffer.write(vertices.tobytes())
                    if self.agent_view:
                        self.markers.render(moderngl.LINE_STRIP, vertices=len(points))
                    self.ctx.point_size = 7
                    self.markers.render(moderngl.POINTS, vertices=len(points))

        self.sprite_program["camera"].value = camera
        self.sprite_program["center"].value = position
        self.sprite_program["rotation"].value = math.cos(angle), math.sin(angle)
        self.texture.use(0)
        self.car.render(moderngl.TRIANGLE_STRIP)

        self.hud_surface.fill((0, 0, 0, 0))
        speed = round(math.hypot(float(state["forward_speed"]), float(state["lateral_speed"])) * 3.6)
        # Same 3x5 pixel speed font as the car_rl video renderer.
        glyphs = {
            "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
            "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
            "6": "111100111101111", "7": "111001010010010", "8": "111101111101111",
            "9": "111101111001111", "K": "101101110101101", "M": "101111111101101",
            "H": "101101111101101", "/": "001001010100100",
        }
        text = f"{speed:3d} KM/H"
        for i, char in enumerate(text):
            for j, pixel in enumerate(glyphs.get(char, "0" * 15)):
                if pixel == "1":
                    pygame.draw.rect(self.hud_surface, (18, 33, 41), (170 + i * 16 + j % 3 * 4, j // 3 * 4, 4, 4))
        if self.agent_view:
            labels = [(30, f"vy {float(state['lateral_speed']):.1f}  yaw {float(state['yaw_rate']):.2f}"),
                      (54, f"steer {math.degrees(float(state['steering_angle'])):.1f}°  {status}")]
        else:
            elapsed = float(state["steps"]) * self.step_dt
            progress = float(state["progress"]) / self.lap_length
            labels = [(30, f"{elapsed:.2f}s   {progress:.1%}"), (54, status)]
        for y, text in labels:
            label = self.font.render(text, True, (18, 33, 41))
            self.hud_surface.blit(label, (310 - label.get_width(), y))
        self.hud_texture.write(pygame.image.tobytes(self.hud_surface, "RGBA"))
        pixel_camera = (2 / self.width, -2 / self.height, -1, 1)
        self.sprite_program["camera"].value = pixel_camera
        self.sprite_program["center"].value = self.width - 330, 10
        self.sprite_program["rotation"].value = 1, 0
        self.hud_texture.use(0)
        self.hud.render(moderngl.TRIANGLE_STRIP)

        bars = np.zeros((12, 6), np.float32)
        pedal = float(np.clip(action[1], -1, 1))
        for i, (value, color) in enumerate([(max(pedal, 0), (0, 1, 0)), (max(-pedal, 0), (1, 0, 0))]):
            x, y = i * 7, self.height * (1 - value)
            bars[i * 6:i * 6 + 6, :2] = ((x, y), (x + 6, y), (x, self.height), (x, self.height), (x + 6, y), (x + 6, self.height))
            bars[i * 6:i * 6 + 6, 2:5] = color
        bars[:, 5] = -1
        self.road_program["camera"].value = pixel_camera
        self.marker_buffer.write(bars.tobytes())
        self.markers.render(vertices=12)
        if self.window:
            self.ctx.copy_framebuffer(self.ctx.screen, self.framebuffer)
            pygame.display.flip()

    def capture(self):
        self.framebuffer.read_into(self.pixels, components=3, alignment=1)
        return self.pixels

    def close(self):
        self.ctx.release()
        if self.window:
            pygame.display.quit()
        pygame.font.quit()


def save_episode(path, env, states, actions, rewards, terminated, truncated, targets, seed, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(file, track=env.track, road_width=env.config.road_width,
            step_dt=env.config.step_dt, states=np.array(states), actions=np.array(actions, np.float32).reshape(-1, 2),
            rewards=np.array(rewards, np.float32), terminated=np.array(terminated, np.uint8),
            truncated=np.array(truncated, np.uint8), targets=np.array(targets, np.float32),
            seed=seed, checkpoint=str(checkpoint or ""),
            car_length=env.config.vehicle.car_length, car_width=env.config.vehicle.car_width)
    print(f"Saved {path}: {len(actions)} steps, {len(actions) * env.config.step_dt:.2f}s")


def episode_status(terminated, truncated):
    if terminated:
        return "OFF ROAD"
    return "TIME LIMIT" if truncated else ""


def replay(data, args):
    states, actions = data["states"], data["actions"]
    dt = float(data["step_dt"])
    assert len(states) == len(actions) + 1 and len(actions) > 0
    assert dt > 0
    renderer = Renderer(data["track"], float(data["road_width"]), args.width, args.height,
                        args.zoom, not args.overview, window=not args.output,
                        car_length=float(data.get("car_length", 5.2)),
                        car_width=float(data.get("car_width", 2.0)))
    renderer.step_dt = dt
    renderer.trail, renderer.lookahead = args.trail, args.lookahead
    renderer.agent_view = args.agent_view
    duration = len(actions) * dt
    status = episode_status(data["terminated"][-1], data["truncated"][-1])
    process = None
    try:
        if args.output:
            assert args.width % 2 == 0 and args.height % 2 == 0, "H.264 output dimensions must be even"
            ffmpeg = shutil.which("ffmpeg")
            assert ffmpeg, "Install FFmpeg for MP4 export"
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-", "-vf", "vflip",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output),
            ], stdin=subprocess.PIPE)
            frames = math.ceil(duration * args.fps) + 1
            for frame in range(frames):
                t = min(frame / args.fps, duration)
                index = min(int(t / dt), len(actions) - 1)
                fraction = min((t - index * dt) / dt, 1)
                state = interpolate(states[index], states[index + 1], fraction)
                renderer.draw(state, actions[index], status if t == duration else "", data["targets"][index + (fraction >= 0.5)])
                process.stdin.write(renderer.capture())
                if frame and frame % (args.fps * 10) == 0:
                    print(f"Rendering {t:.0f}/{duration:.1f}s", flush=True)
            process.stdin.close()
            assert process.wait() == 0, "FFmpeg failed"
            print(f"Saved {output}")
        else:
            clock = pygame.time.Clock()
            t, paused, running = 0.0, False, True
            while running:
                elapsed = clock.tick(args.fps) / 1000
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            paused = not paused
                        elif event.key == pygame.K_r:
                            t = 0
                            renderer.history.clear()
                        elif event.key == pygame.K_LEFT:
                            t = max(0, t - 5)
                            renderer.history.clear()
                        elif event.key == pygame.K_RIGHT:
                            t = min(duration, t + 5)
                            renderer.history.clear()
                        camera_controls(renderer, event)
                    elif event.type == pygame.MOUSEWHEEL:
                        camera_controls(renderer, event)
                if not paused:
                    t = min(t + elapsed, duration)
                index = min(int(t / dt), len(actions) - 1)
                fraction = min((t - index * dt) / dt, 1)
                state = interpolate(states[index], states[index + 1], fraction)
                renderer.draw(state, actions[index], status if t == duration else "PAUSED" if paused else "", data["targets"][index + (fraction >= 0.5)])
    finally:
        if process and process.poll() is None:
            process.stdin.close()
            process.wait()
        renderer.close()


def camera_controls(renderer, event):
    if event.type == pygame.MOUSEWHEEL:
        renderer.zoom = float(np.clip(renderer.zoom * 1.15 ** event.y, 1, 100))
    elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_f:
            renderer.follow = not renderer.follow
        elif event.key == pygame.K_t:
            renderer.trail = not renderer.trail
            renderer.history.clear()
        elif event.key == pygame.K_g:
            renderer.lookahead = not renderer.lookahead
        elif event.key == pygame.K_v:
            renderer.agent_view = not renderer.agent_view


def run(args):
    config = RacingConfig.from_track(args.circuit, lookahead_distances=args.lookahead_distances,
                                    samples_per_span=1, step_dt=0.05, max_steps=args.max_steps)
    env = RenderEnv(config, args.seed)
    renderer = None
    try:
        policy = None
        if args.checkpoint:
            import torch
            from ppo import Agent

            torch.set_num_threads(1)
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            policy = Agent(env.obs.shape[1])
            policy.load_checkpoint(checkpoint, config)
            policy.eval()

        env.reset()
        states, actions, rewards, terminated, truncated, targets = [env.states[0].copy()], [], [], [], [], [env.targets().copy()]
        if not args.output:
            renderer = Renderer(env.track, config.road_width, args.width, args.height, args.zoom, not args.overview,
                                car_length=config.vehicle.car_length, car_width=config.vehicle.car_width)
            renderer.step_dt = config.step_dt
            renderer.trail, renderer.lookahead = args.trail, args.lookahead
            renderer.agent_view = args.agent_view
            pygame.joystick.init()
            controller.init()
        gamepad = None
        clock = pygame.time.Clock()
        accumulator, paused, running, done, saved = 0.0, False, True, False, False
        while running:
            elapsed = clock.tick(args.fps) / 1000 if renderer else config.step_dt
            reset = False
            if renderer:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            reset = True
                        elif event.key == pygame.K_p:
                            paused = not paused
                        camera_controls(renderer, event)
                    elif event.type == pygame.MOUSEWHEEL:
                        camera_controls(renderer, event)
                    elif event.type == pygame.JOYDEVICEADDED:
                        if controller.is_controller(event.device_index):
                            gamepad = controller.Controller(event.device_index)
                    elif event.type == pygame.JOYDEVICEREMOVED:
                        if gamepad and gamepad.as_joystick().get_instance_id() == event.instance_id:
                            gamepad.quit()
                            gamepad = None
                if reset:
                    env.reset()
                    states, actions, rewards, terminated, truncated, targets = [env.states[0].copy()], [], [], [], [], [env.targets().copy()]
                    accumulator, done, saved = 0.0, False, False
                    renderer.history.clear()
            if not running:
                break
            if not paused and not done:
                accumulator += elapsed
                while accumulator >= config.step_dt and not done:
                    if policy:
                        with torch.inference_mode():
                            env.actions[:] = policy.actor_mean(torch.from_numpy(env.obs)).numpy()
                    else:
                        keys = pygame.key.get_pressed()
                        steering = int(keys[pygame.K_a] or keys[pygame.K_LEFT]) - int(keys[pygame.K_d] or keys[pygame.K_RIGHT])
                        pedal = int(keys[pygame.K_w] or keys[pygame.K_UP]) - int(keys[pygame.K_s] or keys[pygame.K_DOWN] or keys[pygame.K_SPACE])
                        if gamepad and gamepad.attached():
                            stick = gamepad.get_axis(pygame.CONTROLLER_AXIS_LEFTX) / 32768
                            steering = -stick if abs(stick) > 0.08 else steering
                            triggers = (gamepad.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)
                                        - gamepad.get_axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT)) / 32767
                            pedal = triggers if abs(triggers) > 0.02 else pedal
                        env.actions[0] = steering, pedal
                    action = env.actions[0].copy()
                    _, _, _, _, info = env.step()
                    done = bool(env.terminated[0] or env.truncated[0])
                    state = states[-1].copy() if done else env.states[0].copy()
                    if done:
                        # The C car already reset. Hold the last pose, with final episode statistics.
                        state["steps"] += 1
                        state["progress"] = info["final_progress"][0] * env.lap_length
                    states.append(state)
                    targets.append(targets[-1].copy() if done else env.targets().copy())
                    actions.append(action)
                    rewards.append(float(env.rewards[0]))
                    terminated.append(int(env.terminated[0]))
                    truncated.append(int(env.truncated[0]))
                    accumulator -= config.step_dt
                if done:
                    status = episode_status(terminated[-1], truncated[-1])
                    print(f"{status}: {len(actions) * config.step_dt:.2f}s, progress {states[-1]['progress'] / env.lap_length:.1%}, return {sum(rewards):.3f}")
                    if args.record:
                        save_episode(args.record, env, states, actions, rewards, terminated, truncated, targets, args.seed, args.checkpoint)
                        saved = True
                    if args.output:
                        running = False
            if renderer:
                state = states[-1] if done or paused or len(states) == 1 else interpolate(states[-2], states[-1], accumulator / config.step_dt)
                status = episode_status(terminated[-1], truncated[-1]) if done else "PAUSED" if paused else ""
                renderer.draw(state, actions[-1] if actions else (0, 0), status, targets[-1])
        if args.record and actions and not saved:
            save_episode(args.record, env, states, actions, rewards, terminated, truncated, targets, args.seed, args.checkpoint)
        if args.output:
            data = dict(track=env.track, road_width=config.road_width, step_dt=config.step_dt,
                        states=np.array(states), actions=np.array(actions), targets=np.array(targets),
                        terminated=np.array(terminated), truncated=np.array(truncated))
            data['car_length'] = config.vehicle.car_length
            data['car_width'] = config.vehicle.car_width
            replay(data, args)
    finally:
        if renderer:
            renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", default="simple")
    parser.add_argument("--lookahead-distances", type=float, nargs="+",
                        default=(0, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45,
                                 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150))
    parser.add_argument("--play", action="store_true", help="Drive with keyboard or gamepad")
    parser.add_argument("--checkpoint", type=Path, help="Watch or export a deterministic policy episode")
    parser.add_argument("--episode", type=Path, help="Replay a recorded episode")
    parser.add_argument("--record", type=Path, help="Save the current episode as NPZ")
    parser.add_argument("--output", type=Path, help="Export MP4 instead of opening a window")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--zoom", type=float, default=6)
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--trail", action="store_true")
    parser.add_argument("--lookahead", action="store_true")
    parser.add_argument("--agent-view", action="store_true", help="Show only car-relative lookahead edges and car state")
    args = parser.parse_args()
    assert sum((args.play, bool(args.checkpoint), bool(args.episode))) == 1, "Choose --play, --checkpoint, or --episode"
    assert args.width > 0 and args.height > 0 and args.fps > 0 and args.zoom > 0
    assert args.max_steps > 0
    assert not args.play or not args.output, "Record live play first, then export the episode"
    assert not args.episode or not args.record, "--record is for live play or checkpoint playback"
    if args.episode:
        with np.load(args.episode, allow_pickle=False) as data:
            replay(data, args)
    else:
        run(args)


if __name__ == "__main__":
    main()
