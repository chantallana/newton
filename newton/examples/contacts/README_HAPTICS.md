# RJ45 Plug-Socket Haptic Feedback

Haptic feedback for the RJ45 plug-socket insertion example using a
[Haply Inverse3](https://www.haply.co/inverse-3) device and optional
Verse Grip controller.

## Quick Start

```bash
uv sync --extra examples
uv run -m newton.examples contacts_rj45_plug_haptics --use-haply-device
```

> **Prerequisites:** The Haply Inverse Service 3.1 must be running and
> the `websockets` and `orjson` Python packages must be installed.
> Without them the example falls back to mouse-gizmo control.

---

## Architecture Overview

```
┌─────────────────────┐        WebSocket         ┌──────────────────┐
│  Haply Inverse3     │◄──────────────────────────│  Haptic Thread   │
│  + Verse Grip       │  cursor position (in)     │  (daemon, async) │
│                     │  set_cursor_force (out)    │  ~2500 Hz        │
└─────────────────────┘                           └────────┬─────────┘
                                                           │
                                              pinned CPU   │ lock-free
                                              buffers      │ reads
                                                           │
                                                  ┌────────┴─────────┐
                                                  │  CUDA Graph      │
                                                  │  (simulation)    │
                                                  │  ~960 Hz substep │
                                                  └────────┬─────────┘
                                                           │
                                                  ┌────────┴─────────┐
                                                  │  Main Thread     │
                                                  │  step() @ 60 Hz  │
                                                  │  click detection  │
                                                  │  button handling  │
                                                  └──────────────────┘
```

### Data Flow

1. **Device → Simulation (position):**
   The Inverse3 sends its cursor position over WebSocket.  The haptic
   thread stores it (relative to a clutch-calibrated origin).  The main
   thread's `step()` reads it and drives the plug gizmo target when the
   clutch button is held.

2. **Simulation → Device (force):**
   A GPU kernel (`_compute_haptic_force_kernel`) computes a spring
   reaction force at every substep (~960 Hz).  The result is
   asynchronously copied to **pinned CPU memory** inside the CUDA graph,
   so the haptic thread can read it at ~2500 Hz without Python-side
   synchronisation or graph-breaking readbacks.

3. **Post-processing (haptic thread):**
   The raw GPU force is smoothed (EMA low-pass filter), slew-rate
   limited, magnitude-clamped, and blended with click-impulse and
   latch-push overlays before being sent to the device.

---

## Verse Grip Button Mapping

| Button | Action | Description |
|--------|--------|-------------|
| **A** (button 1 / primary) | **Clutch** | Hold to engage — device position drives the plug. Release to disengage (device moves freely, plug stays put). Re-engaging recalibrates the origin so the gizmo doesn't jump. |
| **B** (button 2) | **Latch push** | Hold to push the latch inward (overrides joint target to `LATCH_PUSH_ANGLE`). A small downward haptic force (`LATCH_PUSH_FORCE`) is added so you feel the action. |
| **C** (button 3) | **Reset** | Press to reset the plug and latch to their initial positions (one-shot, rising-edge only). |

> **Note:** The clutch currently activates on *either* button A or
> button B to accommodate different Verse Grip firmware versions that
> report buttons under different key names.

---

## Tuning Constants

All constants are defined at the top of
`example_contacts_rj45_plug_haptics.py`.

### Haply Device Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `HAPLY_URI` | `"ws://localhost:10001"` | WebSocket endpoint for Inverse Service 3.1. |
| `HAPLY_FORCE_SCALE` | `0.2` | Scales simulation Newtons → device Newtons. The simulation forces are much larger than what the device should render. |
| `HAPLY_POSITION_SCALE` | `1.0` | Scales device metres → simulation metres. Currently 1:1. Increase to amplify small hand movements. |
| `HAPLY_FORCE_DAMPING` | `0.0` | Velocity-proportional damping added to the haptic force in the GPU kernel. Currently unused (zero). Available as a tuning knob. |
| `HAPLY_FORCE_SMOOTHING` | `0.05` | EMA (exponential moving average) time constant in seconds. Smooths the ~60 Hz GPU-computed force being sent at ~2500 Hz to remove staircase artifacts. Lower = more responsive but noisier. |
| `HAPLY_MAX_FORCE` | `3.0` | Hard clamp on force magnitude (N) sent to the device. Safety limit. |
| `HAPLY_SLEW_RATE` | `40.0` | Maximum force change rate (N/s) per axis. Prevents contact-bounce spikes from reaching the device. |

### Click-Impulse Feedback

A short tactile pulse when the latch snaps past the retention ledge
(latch angle crosses `CLICK_ANGLE_THRESHOLD` from above to below).

| Constant | Default | Description |
|----------|---------|-------------|
| `CLICK_IMPULSE_MAGNITUDE` | `1.0` | Peak force (N) of the click pulse along the insertion axis. |
| `CLICK_DECAY_TIME` | `0.15` | Time (s) to fade the spring force back in after a click. Prevents the stale GPU spring from causing an abrupt snap-back that feels like a second click. |
| `CLICK_ANGLE_THRESHOLD` | `0.052` | Latch angle (rad, ~3°) crossing that triggers the click. |
| `CLICK_COOLDOWN` | `0.8` | Minimum time (s) between consecutive click pulses. Prevents double-clicks. |

### Latch Push (Button B)

| Constant | Default | Description |
|----------|---------|-------------|
| `LATCH_PUSH_ANGLE` | `-0.2` | Joint target angle (rad) when button B is held — pushes latch to its lower limit. |
| `LATCH_PUSH_FORCE` | `0.15` | Downward haptic force (N) added while pushing, so the user feels the action. |

### Gizmo Spring Coupling

These control how tightly the plug tracks the target position (whether
from the Haply device or the mouse gizmo).  They also determine the
magnitude of the spring reaction force sent to the haptic device.

| Constant / Attribute | Default | Description |
|----------------------|---------|-------------|
| `self.gizmo_stiffness` | `1000.0` | Spring stiffness (N/m) coupling the plug body to the gizmo target. Higher = stiffer tracking, stronger haptic forces. |
| `self.gizmo_damping` | `10.0` | Velocity damping (N·s/m) for the gizmo spring. Prevents oscillation. |

### Viewer Mouse Picking

| Constant | Default | Description |
|----------|---------|-------------|
| `VIEWER_PICK_STIFFNESS` | `1000.0` | Overrides the viewer's default mouse-pick spring stiffness (50). Needs to be high enough to pull the plug out of the socket when clicking. Independent of the gizmo spring. |

### Latch Joint Tuning

| Constant | Default | Description |
|----------|---------|-------------|
| `LATCH_REST_ANGLE` | `0.10` | Resting angle (rad) — latch sits slightly open. |
| `LATCH_LIMIT_LOWER` | `-0.2` | Maximum inward deflection (rad). |
| `LATCH_LIMIT_UPPER` | `0.3` | Maximum outward deflection (rad). |
| `LATCH_SPRING_KE` | `0.15` | Angular return-spring stiffness (N·m/rad). |
| `LATCH_SPRING_KD` | `0.01` | Angular return-spring damping (N·m·s/rad). |

---

## Force Processing Pipeline

The force sent to the device goes through the following stages in the
haptic thread:

```
GPU spring force (pinned buffer, ~960 Hz updates)
        │
        ▼
   Zero if clutch not held
        │
        ▼
   Click-decay blend (fade spring back in after click event)
        │
        ▼
   EMA low-pass filter (HAPLY_FORCE_SMOOTHING)
        │
        ▼
   Slew-rate limiter (HAPLY_SLEW_RATE)
        │
        ▼
   Magnitude clamp (HAPLY_MAX_FORCE)
        │
        ▼
   Click impulse overlay (replaces force for one frame)
        │
        ▼
   Latch push overlay (subtracts LATCH_PUSH_FORCE on Z)
        │
        ▼
   Send to device via WebSocket
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `[Haply] --use-haply-device requires 'websockets' and 'orjson'` | Missing packages | `uv pip install websockets orjson` |
| `[Haply] No Inverse3 device found.` | Inverse Service not running or device not connected | Start Inverse Service 3.1, check USB connection |
| Forces feel too weak | `HAPLY_FORCE_SCALE` too low, or `HAPLY_MAX_FORCE` clamping | Increase `HAPLY_FORCE_SCALE` or `HAPLY_MAX_FORCE` |
| Forces feel jittery / buzzy | Smoothing too low or slew rate too high | Increase `HAPLY_FORCE_SMOOTHING` or decrease `HAPLY_SLEW_RATE` |
| No click felt on insertion | Angle threshold not being crossed | Check `CLICK_ANGLE_THRESHOLD` matches the latch geometry; increase `CLICK_IMPULSE_MAGNITUDE` |
| Double-click on insertion | Cooldown too short | Increase `CLICK_COOLDOWN` |
| Gizmo jumps when re-engaging clutch | Origin recalibration issue | This should not happen — origins are recalibrated on clutch rising edge. File a bug. |
| Plug doesn't track hand well | Gizmo spring too soft | Increase `self.gizmo_stiffness` |
