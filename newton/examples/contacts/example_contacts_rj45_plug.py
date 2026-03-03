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
    mu=0.1,
    ke=1e6,
    kd=3e3,
    gap=0.002,
    density=1e6,
    mu_torsional=0.0,
    mu_rolling=0.0,
)

MESH_SDF_MAX_RESOLUTION = 128
MESH_SDF_NARROW_BAND_RANGE = (-2.0 * SHAPE_CFG.gap, 2.0 * SHAPE_CFG.gap)

PLUG_Y_OFFSET = -0.025

# Latch revolute-joint tuning.
LATCH_LIMIT_LOWER = -0.2  # max inward deflection [rad]
LATCH_LIMIT_UPPER = 0.3   # max outward deflection [rad]
LATCH_SPRING_KE = 0.15    # angular return-spring stiffness [N*m/rad]
LATCH_SPRING_KD = 0.03    # angular return-spring damping [N*m*s/rad]


@wp.kernel
def _apply_gizmo_force(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    pick_target: wp.array(dtype=wp.vec3),
    stiffness: float,
    damping: float,
    pick_body: wp.array(dtype=int),
    plug_idx: int,
    latch_idx: int,
):
    """Apply forces based on interaction mode.

    ``pick_body[0]`` encodes the mode:
      * ``>= 0`` -- viewer is picking that body index (damping only on others)
      * ``< 0``  -- spring toward ``pick_target[0]``
    """
    target = pick_target[0]
    picked_body = pick_body[0]

    if picked_body >= 0:
        # During picking, apply only velocity damping to non-picked bodies
        # to prevent undamped oscillation transmitted through the joint.
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


def _load_mesh(stage, prim_path: str) -> tuple[newton.Mesh, wp.vec3]:
    """Load a mesh from USD, center at prim origin, and build SDF.

    Returns:
        Tuple of (mesh, prim_pos) where prim_pos is the prim world-space
        translation [m], usable as the body/shape position.
    """
    prim = stage.GetPrimAtPath(prim_path)
    usd_mesh = newton.usd.get_mesh(prim, load_normals=True)

    tf = newton.usd.get_transform(prim, local=False)
    prim_pos = wp.transform_get_translation(tf)

    vertices = np.array(usd_mesh.vertices, dtype=np.float32)
    indices = np.array(usd_mesh.indices, dtype=np.int32)
    normals = np.array(usd_mesh.normals, dtype=np.float32) if usd_mesh.normals is not None else None

    mesh = newton.Mesh(vertices, indices, normals=normals)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=SHAPE_CFG.gap,
    )
    return mesh, prim_pos


class Example:
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.pick_stiffness = 50.0
        self.pick_damping = 10.0

        usd_path = newton.examples.get_asset("rj45_plug.usd")
        stage = Usd.Stage.Open(usd_path)

        socket_mesh, sc = _load_mesh(stage, "/World/Socket")
        plug_mesh, pc = _load_mesh(stage, "/World/Plug")
        latch_mesh, lc = _load_mesh(stage, "/World/Latch")

        builder = newton.ModelBuilder(gravity=0.0)

        # Socket (static body)
        builder.add_shape_mesh(
            -1,
            mesh=socket_mesh,
            xform=wp.transform(sc, wp.quat_identity()),
            cfg=SHAPE_CFG,
            label="socket",
        )

        # Plug (dynamic body, offset along -Y insertion axis)
        plug_pos = wp.vec3(pc[0], pc[1] + PLUG_Y_OFFSET, pc[2])
        self._plug_body = builder.add_link(
            xform=wp.transform(plug_pos, wp.quat_identity()),
            label="plug",
        )
        builder.add_shape_mesh(
            self._plug_body,
            mesh=plug_mesh,
            cfg=SHAPE_CFG,
        )

        # Latch (dynamic body, same Y offset as plug)
        latch_pos = wp.vec3(lc[0], lc[1] + PLUG_Y_OFFSET, lc[2])
        self._latch_body = builder.add_link(
            xform=wp.transform(latch_pos, wp.quat_identity()),
            label="latch",
        )
        builder.add_shape_mesh(
            self._latch_body,
            mesh=latch_mesh,
            cfg=SHAPE_CFG,
        )

        # D6 joint: world -> plug (free translation, locked rotation)
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
            parent_xform=wp.transform(plug_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )

        # Revolute joint: plug -> latch (hinge along -X axis)
        rev_joint = builder.add_joint_revolute(
            parent=self._plug_body,
            child=self._latch_body,
            axis=(-1.0, 0.0, 0.0),
            parent_xform=wp.transform(lc - pc, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_ke=LATCH_SPRING_KE,
            target_kd=LATCH_SPRING_KD,
            limit_lower=LATCH_LIMIT_LOWER,
            limit_upper=LATCH_LIMIT_UPPER,
            collision_filter_parent=True,
        )

        builder.add_articulation([d6_joint, rev_joint])

        self.model = builder.finalize()

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True

        mid_y = 0.3 * sc[1] + 0.7 * plug_pos[1]
        self.viewer.set_camera(
            pos=wp.vec3(0.08, mid_y, 0.02),
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

        self.solver = SolverXPBD(self.model, iterations=16, rigid_contact_relaxation=0.5, angular_damping=1.0)

        self._rest_pos = plug_pos
        self.gizmo_tf = wp.transform(plug_pos, wp.quat_identity())

        self._pick_body = wp.array([-1], dtype=int, device=self.model.device)
        self._pick_target = wp.zeros(1, dtype=wp.vec3, device=self.model.device)

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
                    self._pick_target,
                    self.pick_stiffness,
                    self.pick_damping,
                    self._pick_body,
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

        picked_body = int(self.viewer.picking.pick_body.numpy()[0])

        self._pick_body.assign([picked_body if picked_body >= 0 else -1])
        self._pick_target.assign([gp])

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Snap gizmo to the plug when the user isn't dragging it.
        gizmo_active = self.viewer.gizmo_is_using
        if not gizmo_active:
            plug_tf = self.state_0.body_q.numpy()[self._plug_body]
            if picked_body >= 0:
                snap = wp.vec3(*plug_tf[:3])
            else:
                snap = wp.vec3(self._rest_pos[0], plug_tf[1], self._rest_pos[2])
            self.gizmo_tf[:] = wp.transform(snap, wp.quat_identity())

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
            drift = np.linalg.norm(body_q[i] - initial_q[i])
            assert drift < 1.0, f"Body {i} drifted {drift:.4f} from initial transform"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
