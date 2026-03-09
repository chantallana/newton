# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example SDF RJ45 Plug-Socket Insertion
#
# Use the translation gizmo to move the plug toward the socket.
# Click an axis arrow to slide along one axis, or a plane square
# for two-axis motion. The latch deflects on entry and locks
# the plug once fully inserted.
#
# Commands:
#     uv sync --extra examples
#     uv run -m newton.examples contacts_rj45_plug
#
###########################################################################

import asyncio
import threading

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.solvers import SolverXPBD

try:
    import websockets
    import orjson

    _HAS_HAPLY_DEPS = True
except ImportError:
    _HAS_HAPLY_DEPS = False

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=0.1,
    ke=1e6,
    kd=1e3,
    gap=0.0002,
    density=1e6,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=False,
)

MESH_SDF_MAX_RESOLUTION = 256
MESH_SDF_NARROW_BAND_RANGE = (-0.0005, 0.0005)

PLUG_Y_OFFSET = -0.05

# Latch hinge pivot in Y-up USD coordinates (measured from the asset).
HINGE_PIVOT_Y_UP = (0.0, -0.00360328, 0.0380272)

# Latch revolute-joint tuning.
LATCH_REST_ANGLE = 0.10   # resting angle [rad] \u2014 latch sits slightly open
LATCH_LIMIT_LOWER = -0.2  # max inward deflection [rad]
LATCH_LIMIT_UPPER = 0.3   # max outward deflection [rad]
LATCH_SPRING_KE = 0.15    # angular return-spring stiffness [N*m/rad]
LATCH_SPRING_KD = 0.01    # angular return-spring damping [N*m*s/rad]

# Viewer pick stiffness override (default 50 is too weak to disconnect).
# Damping is left at the default (5) to stay within the explicit-integration
# stability limit for the light latch body.
PICK_STIFFNESS = 2000.0

# Haply Inverse3 haptic-device settings.
HAPLY_URI = "ws://localhost:10001"  # WebSocket endpoint for Inverse Service 3.1
HAPLY_FORCE_SCALE = 0.8             # tune to map simulation Newtons to device Newtons
HAPLY_POSITION_SCALE = 1.0          # tune to map device metres to simulation metres
HAPLY_FORCE_DAMPING = 0.5           # velocity damping added to haptic feedback [N*s/m]

# Click-impulse feedback: a short force pulse when the latch snaps past the
# retention ledge.  The impulse is triggered when the latch angle jumps from
# below to above the rest position (latch springing outward after clearing
# the retention ledge) and decays exponentially.
CLICK_IMPULSE_MAGNITUDE = 1.0       # peak force [N] of the click pulse (along insertion axis)
CLICK_DECAY_TIME = 0.03             # exponential decay time constant [s]
CLICK_ANGLE_THRESHOLD = 0.052       # latch angle [rad] crossing triggers click (~3°)

# Button B: override latch joint target to push it inward.
LATCH_PUSH_ANGLE = -0.2             # target angle [rad] when button B held (= limit_lower)
LATCH_PUSH_FORCE = 0.5              # upward haptic force [N] while pushing latch (button B)


@wp.kernel
def _apply_gizmo_force(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    target_arr: wp.array(dtype=wp.vec3),
    stiffness: float,
    damping: float,
    picked_body_arr: wp.array(dtype=int),
    plug_idx: int,
    latch_idx: int,
):
    target = target_arr[0]
    picked_body = picked_body_arr[0]

    if picked_body >= 0:
        # During picking, apply only velocity damping to non-picked bodies
        # to prevent undamped oscillation transmitted through the joint.
        # The spring term is omitted so it cannot fight the pick force.
        if picked_body != plug_idx:
            vel0 = wp.spatial_top(body_qd[plug_idx])
            mass0 = body_mass[plug_idx]
            f0 = -(10.0 + mass0) * damping * vel0
            wp.atomic_add(body_f, plug_idx, wp.spatial_vector(f0, wp.vec3(0.0)))
        if picked_body != latch_idx:
            vel1 = wp.spatial_top(body_qd[latch_idx])
            mass1 = body_mass[latch_idx]
            f1 = -(10.0 + mass1) * damping * vel1
            wp.atomic_add(body_f, latch_idx, wp.spatial_vector(f1, wp.vec3(0.0)))
        return

    pos0 = wp.transform_get_translation(body_q[plug_idx])
    vel0 = wp.spatial_top(body_qd[plug_idx])
    mass0 = body_mass[plug_idx]
    mult0 = 10.0 + mass0
    f0 = mult0 * (stiffness * (target - pos0) - damping * vel0)
    wp.atomic_add(body_f, plug_idx, wp.spatial_vector(f0, wp.vec3(0.0)))

    # Give the latch the same translational acceleration as the plug so both
    # bodies predict the same displacement; the revolute joint only has to
    # correct the small relative error instead of bridging the full gap.
    vel1 = wp.spatial_top(body_qd[latch_idx])
    mass1 = body_mass[latch_idx]
    spring_accel = (target - pos0) * (mult0 * stiffness / mass0)
    f1 = spring_accel * mass1 - vel1 * ((10.0 + mass1) * damping)
    wp.atomic_add(body_f, latch_idx, wp.spatial_vector(f1, wp.vec3(0.0)))


def _convert_points_y_up_to_z_up(vertices: np.ndarray) -> np.ndarray:
    """Rotate vertices from Y-up to Z-up so the latch faces +Z: (x, y, z) -> (-x, -z, -y)."""
    out = np.empty_like(vertices)
    out[:, 0] = -vertices[:, 0]
    out[:, 1] = -vertices[:, 2]
    out[:, 2] = -vertices[:, 1]
    return out


def _convert_point_y_up_to_z_up(pt: tuple[float, float, float]) -> np.ndarray:
    """Convert a single point from Y-up to Z-up so the latch faces +Z."""
    return np.array([-pt[0], -pt[2], -pt[1]], dtype=np.float64)


def _load_mesh(stage, prim_path: str) -> tuple[newton.Mesh, np.ndarray, np.ndarray]:
    """Load a mesh from USD, apply prim world transform, convert to Z-up, and center.

    Returns:
        Tuple of (mesh, center, half_extents) where half_extents are measured
        from the centered origin along each axis.
    """
    prim = stage.GetPrimAtPath(prim_path)
    usd_mesh = newton.usd.get_mesh(prim)
    vertices_yup = np.array(usd_mesh.vertices, dtype=np.float64)

    tf = newton.usd.get_transform(prim, local=False)
    prim_pos = np.array([float(tf[i]) for i in range(3)], dtype=np.float64)
    vertices_yup += prim_pos

    vertices = _convert_points_y_up_to_z_up(vertices_yup.astype(np.float32))
    indices = np.array(usd_mesh.indices, dtype=np.int32)

    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices -= center
    half_extents = vertices.max(axis=0)

    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=SHAPE_CFG.gap if SHAPE_CFG.gap is not None else 0.05,
    )
    return mesh, center, half_extents


class Example:
    def __init__(self, viewer, use_haply_device=False):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.pick_stiffness = 2000.0  # Higher stiffness for better control
        self.pick_damping = 50.0     # Higher damping for stability

        usd_path = newton.examples.get_asset("rj45_plug.usd")
        stage = Usd.Stage.Open(usd_path)

        socket_mesh, socket_center, _ = _load_mesh(stage, "/World/Socket")
        plug_mesh, plug_center, plug_half_ext = _load_mesh(stage, "/World/Plug")
        latch_mesh, latch_center, _ = _load_mesh(stage, "/World/Latch")

        builder = newton.ModelBuilder(gravity=0.0)

        sc = socket_center.astype(np.float64)
        pc = plug_center.astype(np.float64)
        lc = latch_center.astype(np.float64)

        # Socket (static body)
        builder.add_shape_mesh(
            -1,
            mesh=socket_mesh,
            xform=wp.transform(wp.vec3(*sc), wp.quat_identity()),
            cfg=SHAPE_CFG,
            label="socket",
        )

        # Plug (dynamic body, offset along -Y insertion axis)
        plug_pos = pc.copy()
        plug_pos[1] += PLUG_Y_OFFSET
        self._plug_body = builder.add_link(
            xform=wp.transform(wp.vec3(*plug_pos), wp.quat_identity()),
            label="plug",
        )
        builder.add_shape_mesh(
            self._plug_body,
            mesh=plug_mesh,
            cfg=SHAPE_CFG,
        )

        # Latch (dynamic body, same Y offset as plug)
        latch_pos = lc.copy()
        latch_pos[1] += PLUG_Y_OFFSET
        self._latch_body = builder.add_link(
            xform=wp.transform(wp.vec3(*latch_pos), wp.quat_identity()),
            label="latch",
        )
        builder.add_shape_mesh(
            self._latch_body,
            mesh=latch_mesh,
            cfg=SHAPE_CFG,
        )

        # D6 joint: world -> plug (free translation, locked rotation)
        plug_world_pos = wp.vec3(*plug_pos)
        JointDof = newton.ModelBuilder.JointDofConfig
        d6_joint = builder.add_joint_d6(
            parent=-1,
            child=self._plug_body,
            linear_axes=(
                JointDof(axis=(1.0, 0.0, 0.0)),
                JointDof(axis=(0.0, 1.0, 0.0)),
                JointDof(axis=(0.0, 0.0, 1.0)),
            ),
            angular_axes=None,
            parent_xform=wp.transform(plug_world_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )

        # Revolute joint: plug -> latch (hinge along -X axis)
        pivot_zup = _convert_point_y_up_to_z_up(HINGE_PIVOT_Y_UP)
        hinge_in_plug = pivot_zup - pc
        hinge_in_latch = pivot_zup - lc
        rev_joint = builder.add_joint_revolute(
            parent=self._plug_body,
            child=self._latch_body,
            axis=(-1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(*hinge_in_plug), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*hinge_in_latch), wp.quat_identity()),
            target_ke=LATCH_SPRING_KE,
            target_kd=LATCH_SPRING_KD,
            target_pos=LATCH_REST_ANGLE,
            limit_lower=LATCH_LIMIT_LOWER,
            limit_upper=LATCH_LIMIT_UPPER,
            collision_filter_parent=True,
        )

        builder.add_articulation([d6_joint, rev_joint])

        # Store the revolute joint index so we can read its DOF later.
        self._latch_joint = rev_joint

        self.model = builder.finalize()

        # Resolve the latch revolute joint's position/velocity DOF offsets.
        self._latch_q_idx = int(self.model.joint_q_start.numpy()[self._latch_joint])
        self._latch_qd_idx = int(self.model.joint_qd_start.numpy()[self._latch_joint])

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        if hasattr(self.viewer, "picking"):
            self.viewer.picking.pick_stiffness = PICK_STIFFNESS
            pick_state_np = self.viewer.picking.pick_state.numpy()
            pick_state_np[0]["pick_stiffness"] = PICK_STIFFNESS
            self.viewer.picking.pick_state.assign(pick_state_np)

        mid_y = (float(sc[1]) + float(plug_pos[1])) / 2.0
        self.viewer.set_camera(
            pos=wp.vec3(0.06, mid_y, 0.015),
            pitch=-10.0,
            yaw=180.0,
        )
        if hasattr(self.viewer, "_cam_speed"):
            self.viewer._cam_speed = 0.2

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._initial_body_q = self.state_0.body_q.numpy().copy()

        self.solver = SolverXPBD(self.model, iterations=16, rigid_contact_relaxation=0.8, angular_damping=0.5)

        self._gizmo_offset_y = float(plug_half_ext[1])
        gizmo_pos = wp.vec3(plug_world_pos[0], plug_world_pos[1] + self._gizmo_offset_y, plug_world_pos[2])
        self.gizmo_tf = wp.transform(gizmo_pos, wp.quat_identity())
        self._gizmo_target = plug_world_pos

        self._pick_body_arr = wp.array([-1], dtype=int, device=self.model.device)
        self._gizmo_target_arr = wp.zeros(1, dtype=wp.vec3, device=self.model.device)

        # Click detection state: tracks latch angle for edge detection
        self._prev_latch_angle = None  # previous-frame latch angle for edge detection

        # Haptic feedback: spring reaction force sent to the Haply device.
        self._feedback_force = np.zeros(3, dtype=np.float64)
        # Haptic input: cursor position received from the Haply device.
        # None means no position has been received yet (device not connected).
        self._haply_position = None  # type: np.ndarray | None
        self._haply_origin = None    # type: np.ndarray | None  - device origin, set on first message
        self._gizmo_origin = np.array(
            [float(plug_world_pos[0]), float(plug_world_pos[1]), float(plug_world_pos[2])],
            dtype=np.float64,
        )
        self._clutch_active = False  # True while Verse Grip button A is held
        self._button_b_active = False  # True while Verse Grip button B is held (latch push)
        self._reset_requested = False  # True on Verse Grip button C press (reset positions)
        self._feedback_lock = threading.Lock()
        self._haply_running = False
        self._use_haply_device = use_haply_device
        if use_haply_device and _HAS_HAPLY_DEPS:
            self._haply_running = True
            self._haply_thread = threading.Thread(target=self._run_haply_loop, daemon=True)
            self._haply_thread.start()
        elif use_haply_device and not _HAS_HAPLY_DEPS:
            print("[Haply] --use-haply-device requires 'websockets' and 'orjson' packages. Falling back to mouse.")

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                kernel=_apply_gizmo_force,
                dim=1,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.model.body_mass,
                    self._gizmo_target_arr,
                    self.pick_stiffness,
                    self.pick_damping,
                    self._pick_body_arr,
                    self._plug_body,
                    self._latch_body,
                ],
                device=self.model.device,
            )
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # If the Haply device has sent a position and the clutch is held, drive the gizmo.
        if self._haply_running:
            with self._feedback_lock:
                haply_pos = self._haply_position
                clutch = self._clutch_active
            if haply_pos is not None and clutch:
                sim_pos = self._gizmo_origin + haply_pos * HAPLY_POSITION_SCALE
                self.gizmo_tf = wp.transform(
                    wp.vec3(
                        float(sim_pos[0]),
                        float(sim_pos[1]) + self._gizmo_offset_y,
                        float(sim_pos[2]),
                    ),
                    wp.quat_identity(),
                )

        gp = wp.transform_get_translation(self.gizmo_tf)
        self._gizmo_target = wp.vec3(float(gp[0]), float(gp[1]) - self._gizmo_offset_y, float(gp[2]))

        picking = getattr(self.viewer, "picking", None)
        picked_body = int(picking.pick_body.numpy()[0]) if picking is not None else -1

        self._pick_body_arr.assign([picked_body])
        self._gizmo_target_arr.assign([self._gizmo_target])

        # Button B: push the latch inward by changing the joint target.
        if self._haply_running:
            with self._feedback_lock:
                push_latch = self._button_b_active
            jtp = self.control.joint_target_pos.numpy()
            jtk = self.model.joint_target_ke.numpy()
            if push_latch:
                jtp[self._latch_q_idx] = LATCH_PUSH_ANGLE
                jtk[self._latch_q_idx] = 2.0  # stronger spring to overcome contacts
            else:
                jtp[self._latch_q_idx] = LATCH_REST_ANGLE
                jtk[self._latch_q_idx] = 1.0  # strong enough to recover from pushed-down position
            self.control.joint_target_pos.assign(jtp)
            self.model.joint_target_ke.assign(jtk)
        
        # Button C: reset plug and latch to initial positions.
        if self._haply_running:
            with self._feedback_lock:
                do_reset = self._reset_requested
                self._reset_requested = False
            if do_reset:
                self.state_0.body_q.assign(self._initial_body_q)
                self.state_0.body_qd.zero_()
                # Reset gizmo to initial position.
                plug_q = self._initial_body_q[self._plug_body]
                self.gizmo_tf = wp.transform(
                    wp.vec3(float(plug_q[0]), float(plug_q[1]) + self._gizmo_offset_y, float(plug_q[2])),
                    wp.quat_identity(),
                )
                gp = wp.transform_get_translation(self.gizmo_tf)
                self._gizmo_target = wp.vec3(float(gp[0]), float(gp[1]) - self._gizmo_offset_y, float(gp[2]))
                self._gizmo_origin = np.array(
                    [float(plug_q[0]), float(plug_q[1]), float(plug_q[2])], dtype=np.float64
                )
                self._prev_latch_angle = None
                print("[Haply] Reset to initial positions.")

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Snap the gizmo to the plug while picking so it stays anchored at
        # the new position once the pick is released.
        is_picking = picked_body >= 0

        if is_picking or self._haply_running:
            body_q_np = self.state_0.body_q.numpy()

            if is_picking:
                plug_tf = np.asarray(body_q_np[self._plug_body], dtype=np.float64)
                self.gizmo_tf = wp.transform(
                    wp.vec3(float(plug_tf[0]), float(plug_tf[1]) + self._gizmo_offset_y, float(plug_tf[2])),
                    wp.quat_identity(),
                )

            if self._haply_running:
                # Only compute and send force when the clutch is held.
                with self._feedback_lock:
                    clutch = self._clutch_active
                if clutch:
                    # Compute the spring reaction force (what the user "feels").
                    # reaction = stiffness * (plug_pos - target) - damping * velocity
                    body_qd_np = self.state_0.body_qd.numpy()
                    plug_pos = np.array(body_q_np[self._plug_body][:3], dtype=np.float64)
                    plug_vel = np.array(body_qd_np[self._plug_body][:3], dtype=np.float64)
                    target = np.array(
                        [self._gizmo_target[0], self._gizmo_target[1], self._gizmo_target[2]],
                        dtype=np.float64,
                    )
                    spring_reaction = (
                        self.pick_stiffness * (plug_pos - target) - HAPLY_FORCE_DAMPING * plug_vel
                    )

                    # --- Click impulse from latch snap ---
                    # XPBD works in maximal coordinates so joint_q is NOT
                    # updated after solving.  Compute the latch hinge angle
                    # from the relative orientation of the two bodies instead.
                    plug_quat = body_q_np[self._plug_body][3:7]   # (x, y, z, w)
                    latch_quat = body_q_np[self._latch_body][3:7]
                    # Relative quaternion: q_rel = conj(plug) * latch
                    pw, px, py, pz = float(plug_quat[3]), float(plug_quat[0]), float(plug_quat[1]), float(plug_quat[2])
                    lw, lx, ly, lz = float(latch_quat[3]), float(latch_quat[0]), float(latch_quat[1]), float(latch_quat[2])
                    # conj(plug) = (pw, -px, -py, -pz)
                    rw = pw * lw + px * lx + py * ly + pz * lz
                    rx = pw * lx - px * lw - py * lz + pz * ly
                    # Hinge axis is -X, so the angle around X is:
                    latch_angle = 2.0 * np.arctan2(-rx, rw)  # negate for -X axis
                    print(f"[Latch] {np.degrees(latch_angle):.1f}°")

                    # Trigger: latch angle drops below the click threshold —
                    # the latch is snapping inward past the retention ledge.
                    click_force = np.zeros(3, dtype=np.float64)
                    if (
                        self._prev_latch_angle is not None
                        and self._prev_latch_angle >= CLICK_ANGLE_THRESHOLD
                        and latch_angle < CLICK_ANGLE_THRESHOLD
                    ):
                        # Apply instant click force on the snap event
                        click_force[1] = CLICK_IMPULSE_MAGNITUDE
                        print(f"[Haply] CLICK! latch {np.degrees(self._prev_latch_angle):.1f}° -> {np.degrees(latch_angle):.1f}°")

                    self._prev_latch_angle = latch_angle

                    with self._feedback_lock:
                        if click_force[1] > 0:
                            # During click event, send only the click force
                            self._feedback_force = click_force
                        else:
                            # Normal operation, send zero force (no spring resistance felt)
                            self._feedback_force = np.zeros(3, dtype=np.float64)
                else:
                    self._prev_latch_angle = None
                    with self._feedback_lock:
                        self._feedback_force = np.zeros(3, dtype=np.float64)

                # Button B: add an upward force even when not clutching so
                # the user always feels the latch push.
                if push_latch:
                    with self._feedback_lock:
                        self._feedback_force = self._feedback_force.copy()
                        self._feedback_force[2] += LATCH_PUSH_FORCE

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_gizmo("plug", self.gizmo_tf)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Haply Inverse3 haptic-device integration (background thread)
    # ------------------------------------------------------------------

    def _run_haply_loop(self):
        """Entry point for the daemon thread running the Haply haptic loop."""
        try:
            asyncio.run(self._haply_async_loop())
        except Exception as exc:  # noqa: BLE001
            print(f"[Haply] connection closed: {exc}")
        finally:
            self._haply_running = False

    async def _haply_async_loop(self):
        """Async loop: receive device state, send back the simulation spring force."""
        first_message = True
        inverse3_device_id = None
        prev_clutch = False
        buttons_printed = False

        async with websockets.connect(HAPLY_URI) as ws:
            while True:
                response = await ws.recv()
                data = orjson.loads(response)

                inverse3_devices = data.get("inverse3", [])
                inverse3_data = inverse3_devices[0] if inverse3_devices else {}
                verse_grip_devices = data.get("wireless_verse_grip", [])
                verse_grip_data = verse_grip_devices[0] if verse_grip_devices else {}

                if first_message:
                    first_message = False
                    if not inverse3_data:
                        print("[Haply] No Inverse3 device found.")
                        break
                    inverse3_device_id = inverse3_data.get("device_id")
                    print(f"[Haply] Inverse3 device ID: {inverse3_device_id}")
                    if verse_grip_data:
                        print(f"[Haply] Verse Grip device ID: {verse_grip_data.get('device_id')}")

                # Read Verse Grip buttons.
                buttons = verse_grip_data.get("state", {}).get("buttons", {})
                if buttons and not buttons_printed:
                    print(f"[Haply] Raw buttons dict: {buttons}")
                    buttons_printed = True
                clutch = bool(
                    buttons.get("1", False)
                    or buttons.get("b1", False)
                    or buttons.get("a", False)
                    or buttons.get("primary", False)
                )
                button_b = bool(
                    buttons.get("2", False)
                    or buttons.get("b2", False)
                    or buttons.get("b", False)
                )
                button_c = bool(
                    buttons.get("3", False)
                    or buttons.get("b3", False)
                    or buttons.get("c", False)
                )
                if button_b:
                    print("[Haply] Button B held")
                if button_c:
                    print("[Haply] Button C pressed")

                # Extract cursor position from device state and store it.
                cursor_pos = inverse3_data.get("state", {}).get("cursor_position", {})
                if cursor_pos:
                    device_pos = np.array(
                        [cursor_pos.get("x", 0.0), cursor_pos.get("y", 0.0), cursor_pos.get("z", 0.0)],
                        dtype=np.float64,
                    )
                    with self._feedback_lock:
                        # Rising edge of clutch: recalibrate origins so the
                        # user can release, reposition, and re-engage without
                        # the gizmo jumping.
                        if clutch and not prev_clutch:
                            # Anchor device origin at current device position.
                            self._haply_origin = device_pos.copy()
                            # Anchor gizmo origin at current gizmo target so the
                            # mapping resumes from where the plug currently is.
                            gp = self._gizmo_target
                            self._gizmo_origin = np.array(
                                [float(gp[0]), float(gp[1]), float(gp[2])],
                                dtype=np.float64,
                            )
                        elif self._haply_origin is None:
                            # First position ever: set initial origin.
                            self._haply_origin = device_pos.copy()
                        self._haply_position = device_pos - self._haply_origin
                        self._clutch_active = clutch
                        self._button_b_active = button_b
                        # Button C: set flag on rising edge only (one-shot reset).
                        if button_c and not getattr(self, '_prev_button_c', False):
                            self._reset_requested = True
                        self._prev_button_c = button_c

                prev_clutch = clutch

                # Read the latest feedback force (thread-safe).
                with self._feedback_lock:
                    fx, fy, fz = self._feedback_force

                request_msg = {
                    "inverse3": [
                        {
                            "device_id": inverse3_device_id,
                            "commands": {
                                "set_cursor_force": {
                                    "values": {
                                        "x": float(fx),
                                        "y": float(fy),
                                        "z": float(fz),
                                    }
                                }
                            },
                        }
                    ]
                }
                await ws.send(orjson.dumps(request_msg))

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        initial_q = self._initial_body_q
        for i in range(len(body_q)):
            assert np.all(np.isfinite(body_q[i])), f"Body {i} has non-finite transform"
            drift = float(np.linalg.norm(
                np.asarray(body_q[i], dtype=np.float64)
                - np.asarray(initial_q[i], dtype=np.float64)
            ))
            assert drift < 1.0, f"Body {i} drifted {drift:.4f} from initial transform"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--use-haply-device",
        action="store_true",
        default=False,
        help="Use a Haply Inverse3 haptic device instead of the mouse gizmo.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, use_haply_device=args.use_haply_device)
    newton.examples.run(example, args)
