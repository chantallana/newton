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

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.solvers import SolverXPBD

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
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.pick_stiffness = 50.0
        self.pick_damping = 5.0

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

        self.model = builder.finalize()

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
        gp = wp.transform_get_translation(self.gizmo_tf)
        self._gizmo_target = wp.vec3(float(gp[0]), float(gp[1]) - self._gizmo_offset_y, float(gp[2]))

        picking = getattr(self.viewer, "picking", None)
        picked_body = int(picking.pick_body.numpy()[0]) if picking is not None else -1

        self._pick_body_arr.assign([picked_body])
        self._gizmo_target_arr.assign([self._gizmo_target])

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Snap the gizmo to the plug while picking so it stays anchored at
        # the new position once the pick is released.
        if picked_body >= 0:
            plug_tf = np.asarray(
                self.state_0.body_q.numpy()[self._plug_body], dtype=np.float64
            )
            self.gizmo_tf = wp.transform(
                wp.vec3(float(plug_tf[0]), float(plug_tf[1]) + self._gizmo_offset_y, float(plug_tf[2])),
                wp.quat_identity(),
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_gizmo("plug", self.gizmo_tf)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

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
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)