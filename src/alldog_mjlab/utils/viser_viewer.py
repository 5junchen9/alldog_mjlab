"""Project play diagnostics on the pinned MjLab Viser viewer."""

from __future__ import annotations

import torch
from mjlab.viewer import ViserPlayViewer

from alldog_mjlab.algorithms.him.policy import HIMPolicy


class HimViserPlayViewer(ViserPlayViewer):
    """Compare HIM and actual body velocity in the selected environment's Info card.

    A play-only forward hook reads the encoder output used by the actor. It
    neither runs a second inference nor changes the policy or its checkpoint.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observed_policy: HIMPolicy | None = None
        self._velocity_hook = None
        self._estimated_velocity: torch.Tensor | None = None
        self._velocity_scale = 1.0
        self._bind_velocity_output()

    def _bind_velocity_output(self) -> None:
        if self.policy is self._observed_policy:
            return
        if self._velocity_hook is not None:
            self._velocity_hook.remove()
            self._velocity_hook = None
        self._observed_policy = None
        self._estimated_velocity = None
        if isinstance(self.policy, HIMPolicy):
            self._observed_policy = self.policy
            term = self.env.unwrapped.observation_manager.get_term_cfg(
                "estimator_velocity", "base_lin_vel"
            )
            self._velocity_scale = term.scale if term.scale is not None else 1.0
            self._velocity_hook = self.policy.estimator.encoder.register_forward_hook(
                self._capture_velocity
            )

    def _capture_velocity(self, module, args, output: torch.Tensor) -> None:
        del module, args
        self._estimated_velocity = output[..., :3].detach()

    def _execute_step(self) -> bool:
        self._bind_velocity_output()
        return super()._execute_step()

    def _update_status_display(self) -> None:
        self._bind_velocity_output()
        super()._update_status_display()
        if self._observed_policy is None:
            return
        if self._estimated_velocity is None:
            values = ("—", "—", "—")
        else:
            velocity = self._estimated_velocity[self._scene.env_idx]
            scale = torch.as_tensor(
                self._velocity_scale, device=velocity.device, dtype=velocity.dtype
            )
            values = tuple(f"{v:+.3f}" for v in (velocity / scale).cpu().tolist())
        actual = self.env.unwrapped.scene["robot"].data.root_link_lin_vel_b[
            self._scene.env_idx
        ]
        readings = (
            ("HIM velocity", values),
            ("Actual velocity", tuple(f"{v:+.3f}" for v in actual.cpu().tolist())),
        )
        for label, row_values in readings:
            cells = "".join(
                '<div style="min-width:0;">'
                f'<div style="opacity:0.65;font-size:0.9em;">{axis}</div>'
                '<div style="font-variant-numeric:tabular-nums;'
                f'font-family:monospace;font-size:1.15em;">{value}</div></div>'
                for axis, value in zip(("vx", "vy", "vz"), row_values, strict=True)
            )
            self._status_html.content += (
                '<div style="font-size:0.85em;line-height:1.4;'
                'padding:0.4em 1em 0.7em 1em;">'
                '<div style="display:flex;justify-content:space-between;'
                'align-items:baseline;margin-bottom:0.35em;">'
                f'<strong>{label}</strong>'
                '<span style="opacity:0.65;font-size:0.9em;">Body frame · m/s</span>'
                '</div><div style="display:grid;'
                f'grid-template-columns:repeat(3,minmax(0,1fr));gap:0.5em;">{cells}</div></div>'
            )

    def reset_environment(self) -> None:
        self._estimated_velocity = None
        super().reset_environment()

    def _handle_gui_reset(self, all_envs: bool) -> None:
        self._estimated_velocity = None
        super()._handle_gui_reset(all_envs)

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._velocity_hook is not None:
                self._velocity_hook.remove()
                self._velocity_hook = None
