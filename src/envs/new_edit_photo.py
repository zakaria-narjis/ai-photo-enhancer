"""Fast, Lightroom-style global photo adjustments for the RL environment.

The editor accepts display-encoded sRGB images in ``(B, H, W, 3)`` layout and
normalized actions in ``[-1, 1]``. Processing is performed in scene-linear
Rec.2020, with an edge-aware log-luminance pass for Highlights/Shadows and
perceptual colour controls in OkLab.

The implementation is intentionally limited to the ten controls that account
for most global photographic correction. Expensive creative/spatial effects
such as Clarity and Dehaze are outside this training environment.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# This order is part of the environment/checkpoint interface. It matches the
# order used by the previous implementation and by demo.py's SLIDERS_ORD.
SLIDER_ORDER: tuple[str, ...] = (
    "contrast",
    "exposure",
    "temp",
    "tint",
    "whites",
    "blacks",
    "highlights",
    "shadows",
    "vibrance",
    "saturation",
)

_GREY = 0.18
_REC2020_LUMA = (0.2627, 0.6780, 0.0593)
_SRGB_LUMA = (0.2126, 0.7152, 0.0722)

# D65 matrices. Keeping the whole editing path in a linear wide-gamut space
# avoids the clipping and hue shifts caused by editing gamma-encoded HSV.
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_REC2020_TO_XYZ = (
    (0.6369580483, 0.1446169036, 0.1688809752),
    (0.2627002120, 0.6779980715, 0.0593017165),
    (0.0, 0.0280726930, 1.0609850577),
)
_BRADFORD = (
    (0.8951, 0.2664, -0.1614),
    (-0.7502, 1.7135, 0.0367),
    (0.0389, -0.0685, 1.0296),
)
_XYZ_TO_LMS = (
    (0.8189330101, 0.3618667424, -0.1288597137),
    (0.0329845436, 0.9293118715, 0.0361456387),
    (0.0482003018, 0.2643662691, 0.6338517070),
)
_LMS_TO_OKLAB = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)


def _matrix(values: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _apply_matrix(image: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply row-major matrices shaped ``(3, 3)`` or ``(B, 3, 3)``."""
    if matrix.ndim == 2:
        return torch.matmul(image, matrix.transpose(0, 1))
    return torch.einsum("bij,bhwj->bhwi", matrix, image)


def _srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )


def _linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * image.pow(1.0 / 2.4) - 0.055,
    )


def _smoothstep(edge0: float, edge1: float, value: torch.Tensor) -> torch.Tensor:
    t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance(image: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (image * weights).sum(dim=-1)


def _log_luminance(image: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.log2(_luminance(image, weights).clamp_min(1e-7) / _GREY)


def _box_mean(plane: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return plane
    kernel = radius * 2 + 1
    padded = F.pad(plane, (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1)


def _guided_base(
    log_luminance: torch.Tensor, radius: int, eps: float = 0.35
) -> torch.Tensor:
    """Fast self-guided base layer, retaining edges while smoothing texture."""
    source = log_luminance.unsqueeze(1)

    # Large full-resolution windows are wasteful. Estimate the guided-filter
    # coefficients on a proxy and upsample them, as in a fast guided filter.
    step = max(1, min(16, radius // 3))
    if step > 1:
        height, width = source.shape[-2:]
        low = F.interpolate(
            source,
            size=((height + step - 1) // step, (width + step - 1) // step),
            mode="area",
        )
        low_radius = max(1, round(radius / step))
    else:
        low = source
        low_radius = radius

    mean = _box_mean(low, low_radius)
    correlation = _box_mean(low * low, low_radius)
    variance = (correlation - mean * mean).clamp_min(0.0)
    a = variance / (variance + eps)
    b = mean - a * mean
    mean_a = _box_mean(a, low_radius)
    mean_b = _box_mean(b, low_radius)

    if step > 1:
        mean_a = F.interpolate(
            mean_a, size=source.shape[-2:], mode="bilinear", align_corners=False
        )
        mean_b = F.interpolate(
            mean_b, size=source.shape[-2:], mode="bilinear", align_corners=False
        )
    return (mean_a * source + mean_b).squeeze(1)


def _kim_xy(temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = temperature.clamp(1667.0, 25000.0)
    t2 = temperature * temperature
    t3 = t2 * temperature
    x_low = -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / temperature + 0.179910
    x_high = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / temperature + 0.240390
    x = torch.where(temperature <= 4000.0, x_low, x_high)
    x2 = x * x
    x3 = x2 * x
    y = torch.where(
        temperature <= 2222.0,
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683,
        torch.where(
            temperature <= 4000.0,
            -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867,
            3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483,
        ),
    )
    return x, y


def _xy_to_uv(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = -2.0 * x + 12.0 * y + 3.0
    return 4.0 * x / denominator, 6.0 * y / denominator


def _uv_to_xy(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = 2.0 * u - 8.0 * v + 4.0
    return 3.0 * u / denominator, 2.0 * v / denominator


def _locus_uv(temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The RL-facing relative temperature range maps to about 4276..13542 K,
    # entirely inside the Kim approximation's calibrated interval.
    return _xy_to_uv(*_kim_xy(temperature))


def _temp_tint_xy(
    temperature: torch.Tensor, tint: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    u, v = _locus_uv(temperature)
    mired = 1e6 / temperature
    u0, v0 = _locus_uv(1e6 / (mired + 1.0))
    u1, v1 = _locus_uv(1e6 / (mired - 1.0).clamp_min(1.0))
    du = u1 - u0
    dv = v1 - v0
    length = torch.sqrt(du * du + dv * dv).clamp_min(1e-12)
    nx = -dv / length
    ny = du / length
    flip = ny < 0.0
    nx = torch.where(flip, -nx, nx)
    ny = torch.where(flip, -ny, ny)
    duv = tint / 3000.0
    return _uv_to_xy(u + nx * duv, v + ny * duv)


class PhotoEditor(nn.Module):
    """Apply a fixed-order, ten-control global development pipeline.

    Parameters are supplied only for selected sliders, but their tensor order is
    always the canonical :data:`SLIDER_ORDER` filtered to those selections. The
    caller's list order does not alter the action contract, matching the legacy
    environment behavior.
    """

    slider_order = SLIDER_ORDER

    def __init__(self, sliders: str | Sequence[str] = "all") -> None:
        super().__init__()
        if sliders == "all":
            active = SLIDER_ORDER
        else:
            requested = tuple(sliders)
            unknown = sorted(set(requested) - set(SLIDER_ORDER))
            if unknown:
                raise ValueError(
                    f"Unsupported photo-editing sliders: {unknown}. "
                    f"Available sliders: {list(SLIDER_ORDER)}"
                )
            if len(set(requested)) != len(requested):
                raise ValueError("Photo-editing slider names must be unique")
            active = tuple(name for name in SLIDER_ORDER if name in requested)

        self.sliders = active
        self.num_parameters = len(active)
        self._active_positions = tuple(SLIDER_ORDER.index(name) for name in active)
        self._has_wb = any(name in active for name in ("temp", "tint"))
        self._has_local_tone = any(name in active for name in ("highlights", "shadows"))
        self._has_tone_map = any(
            name in active for name in ("contrast", "whites", "blacks")
        )
        self._has_colour = any(name in active for name in ("vibrance", "saturation"))

        srgb_to_xyz = _matrix(_SRGB_TO_XYZ)
        working_to_xyz = _matrix(_REC2020_TO_XYZ)
        xyz_to_working = torch.linalg.inv(working_to_xyz)
        srgb_to_working = xyz_to_working @ srgb_to_xyz
        working_to_srgb = torch.linalg.inv(srgb_to_working)
        bradford = _matrix(_BRADFORD)

        self.register_buffer("_srgb_to_working", srgb_to_working)
        self.register_buffer("_working_to_srgb", working_to_srgb)
        self.register_buffer("_working_to_xyz", working_to_xyz)
        self.register_buffer("_xyz_to_working", xyz_to_working)
        self.register_buffer("_bradford", bradford)
        self.register_buffer("_bradford_inverse", torch.linalg.inv(bradford))
        self.register_buffer("_working_luma", torch.tensor(_REC2020_LUMA))
        self.register_buffer("_srgb_luma", torch.tensor(_SRGB_LUMA))

        to_lms = _matrix(_XYZ_TO_LMS) @ working_to_xyz
        to_lab = _matrix(_LMS_TO_OKLAB)
        self.register_buffer("_ok_to_lms", to_lms)
        self.register_buffer("_ok_from_lms", torch.linalg.inv(to_lms))
        self.register_buffer("_ok_to_lab", to_lab)
        self.register_buffer("_ok_from_lab", torch.linalg.inv(to_lab))

        # LightCraft/Lightroom-style vibrance protects orange skin hues.
        skin_srgb = torch.tensor([1.0, 25.0 / 60.0, 0.0])
        skin_linear = _srgb_to_linear(skin_srgb)
        skin_working = _apply_matrix(skin_linear, srgb_to_working)
        skin_lab = self._working_to_oklab(skin_working)
        self.register_buffer("_skin_hue", torch.atan2(skin_lab[2], skin_lab[1]))

        shot_temperature = torch.tensor([6500.0])
        shot_tint = torch.tensor([0.0])
        shot = self._absolute_wb_matrix(shot_temperature, shot_tint)[0]
        self.register_buffer("_shot_wb_inverse", torch.linalg.inv(shot))

    def _absolute_wb_matrix(
        self, temperature: torch.Tensor, tint: torch.Tensor
    ) -> torch.Tensor:
        x, y = _temp_tint_xy(temperature, tint)
        source_xyz = torch.stack((x / y, torch.ones_like(x), (1.0 - x - y) / y), dim=-1)
        d65_xy = source_xyz.new_tensor((0.3127, 0.3290))
        destination_xyz = torch.stack(
            (
                d65_xy[0] / d65_xy[1],
                d65_xy.new_tensor(1.0),
                (1.0 - d65_xy.sum()) / d65_xy[1],
            )
        )
        source_cone = torch.matmul(source_xyz, self._bradford.transpose(0, 1))
        destination_cone = torch.matmul(self._bradford, destination_xyz)
        diagonal = torch.diag_embed(destination_cone / source_cone)
        adaptation = self._bradford_inverse @ diagonal @ self._bradford
        return self._xyz_to_working @ adaptation @ self._working_to_xyz

    def _relative_wb_matrix(
        self, temperature_action: torch.Tensor, tint_action: torch.Tensor
    ) -> torch.Tensor:
        relative_temperature = temperature_action * 100.0
        kelvin = 1e6 / (1e6 / 6500.0 - relative_temperature * 0.8)
        matrix = self._absolute_wb_matrix(kelvin, tint_action * 100.0)
        matrix = matrix @ self._shot_wb_inverse

        # White balance should not change overall exposure.
        neutral = matrix.sum(dim=-1)
        luminance = (neutral * self._working_luma).sum(dim=-1).clamp_min(1e-8)
        return matrix / luminance[:, None, None]

    def _working_to_oklab(self, image: torch.Tensor) -> torch.Tensor:
        lms = _apply_matrix(image, self._ok_to_lms)
        lms_root = torch.sign(lms) * lms.abs().pow(1.0 / 3.0)
        return _apply_matrix(lms_root, self._ok_to_lab)

    def _oklab_to_working(self, image: torch.Tensor) -> torch.Tensor:
        lms_root = _apply_matrix(image, self._ok_from_lab)
        return _apply_matrix(lms_root * lms_root * lms_root, self._ok_from_lms)

    def _unpack_actions(self, parameters: torch.Tensor) -> torch.Tensor:
        batch = parameters.shape[0]
        actions = parameters.new_zeros((batch, len(SLIDER_ORDER)))
        if self._active_positions:
            actions[:, self._active_positions] = parameters
        return actions.clamp(-1.0, 1.0)

    def _local_tone(
        self,
        image: torch.Tensor,
        pre_exposure_log_luminance: torch.Tensor,
        exposure_ev: torch.Tensor,
        highlights: torch.Tensor,
        shadows: torch.Tensor,
    ) -> torch.Tensor:
        long_edge = max(image.shape[1], image.shape[2])
        radius = max(1, round(0.015 * long_edge))
        base = _guided_base(pre_exposure_log_luminance, radius) + exposure_ev
        shadow_weight = 1.0 - _smoothstep(-4.8, 0.3, base)
        highlight_weight = _smoothstep(-1.0, 2.8, base)
        delta_ev = (
            shadows * 1.7 * shadow_weight * shadow_weight.sqrt()
            + highlights * 1.7 * highlight_weight
        )
        return image * torch.exp2(delta_ev).unsqueeze(-1)

    def _tone_map(
        self,
        image: torch.Tensor,
        contrast: torch.Tensor,
        whites: torch.Tensor,
        blacks: torch.Tensor,
    ) -> torch.Tensor:
        luminance = _luminance(image, self._working_luma).clamp_min(0.0)
        perceptual = luminance.clamp_min(1e-10).pow(1.0 / 2.2)
        middle = _GREY ** (1.0 / 2.2)

        below_white = perceptual <= 1.0
        adjusted = perceptual + contrast * 0.35 * (perceptual - middle) * (
            1.0 - (2.0 * perceptual - 1.0).square()
        )
        upper_weight = _smoothstep(0.45, 1.0, adjusted)
        adjusted = adjusted + whites * 0.12 * upper_weight * (1.0 - adjusted * 0.5)
        lower_weight = 1.0 - _smoothstep(0.0, 0.45, adjusted)
        adjusted = adjusted + blacks * 0.07 * lower_weight * (adjusted * 2.0).clamp_max(
            1.0
        )
        above_adjusted = perceptual + whites * 0.12 * 0.5
        adjusted = torch.where(below_white, adjusted, above_adjusted)

        output_luminance = adjusted.clamp_min(0.0).pow(2.2)
        shoulder_distance = (output_luminance - 0.95).clamp(0.0, 0.1)
        shoulder = 0.95 + shoulder_distance - shoulder_distance.square() / 0.2
        output_luminance = torch.where(
            output_luminance > 0.95, shoulder, output_luminance
        ).clamp(0.0, 1.0)
        scale = output_luminance / luminance.clamp_min(1e-9)
        mapped = image * scale.unsqueeze(-1)
        mapped = torch.where(
            (luminance > 1e-9).unsqueeze(-1), mapped, torch.zeros_like(mapped)
        )

        # Exactly neutral controls must be an identity for already-rendered JPEGs.
        active = ((contrast.abs() + whites.abs() + blacks.abs()) > 1e-8).unsqueeze(-1)
        return torch.where(active, mapped, image)

    def _colour(
        self, image: torch.Tensor, vibrance: torch.Tensor, saturation: torch.Tensor
    ) -> torch.Tensor:
        lab = self._working_to_oklab(image)
        a = lab[..., 1]
        b = lab[..., 2]
        chroma = torch.sqrt(a * a + b * b)

        low_chroma = 1.0 - (chroma / 0.22).clamp(0.0, 1.0)
        hue = torch.atan2(b, a)
        hue_delta = (
            torch.remainder(hue - self._skin_hue + torch.pi, 2.0 * torch.pi) - torch.pi
        )
        skin_protection = torch.where(
            vibrance > 0.0,
            1.0 - 0.6 * torch.exp(-((hue_delta / 0.35).square())),
            torch.ones_like(hue),
        )
        vibrance_scale = (
            1.0 + vibrance * low_chroma.square() * skin_protection * 1.2
        ).clamp_min(0.0)
        saturation_scale = (1.0 + saturation).clamp_min(0.0)
        scale = vibrance_scale * saturation_scale

        output_lab = torch.stack((lab[..., 0], a * scale, b * scale), dim=-1)
        return self._oklab_to_working(output_lab)

    def _gamut_map_srgb(self, image: torch.Tensor) -> torch.Tensor:
        luminance = _luminance(image, self._srgb_luma).clamp(0.0, 1.0)
        y = luminance.unsqueeze(-1)
        low_scale = y / (y - image).clamp_min(1e-9)
        high_scale = (1.0 - y) / (image - y).clamp_min(1e-9)
        candidates = torch.where(
            image < 0.0,
            low_scale,
            torch.where(image > 1.0, high_scale, torch.ones_like(image)),
        )
        chroma_scale = candidates.amin(dim=-1, keepdim=True).clamp(0.0, 1.0)
        return y + (image - y) * chroma_scale

    @torch.no_grad()
    def forward(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("images must have shape (B, H, W, 3)")
        if not images.is_floating_point():
            raise TypeError(
                "images must be a floating-point tensor in the [0, 1] range"
            )
        if parameters.ndim != 2:
            raise ValueError("parameters must have shape (B, num_parameters)")
        if parameters.shape != (images.shape[0], self.num_parameters):
            raise ValueError(
                "parameters must have shape "
                f"({images.shape[0]}, {self.num_parameters}), got {tuple(parameters.shape)}"
            )
        if parameters.device != images.device:
            raise ValueError("images and parameters must be on the same device")

        # Buffers follow the first input automatically. Subsequent calls on the
        # same device do not allocate or transfer any constants.
        if self._srgb_to_working.device != images.device:
            self.to(device=images.device)

        image = images.to(dtype=torch.float32).clamp(0.0, 1.0)
        actions = self._unpack_actions(parameters.to(dtype=torch.float32))
        values = {
            name: actions[:, i, None, None] for i, name in enumerate(SLIDER_ORDER)
        }

        image = _apply_matrix(_srgb_to_linear(image), self._srgb_to_working)

        if self._has_wb:
            wb = self._relative_wb_matrix(
                values["temp"][:, 0, 0], values["tint"][:, 0, 0]
            )
            image = _apply_matrix(image, wb).clamp_min(0.0)

        pre_exposure_log_luminance = None
        if self._has_local_tone:
            pre_exposure_log_luminance = _log_luminance(image, self._working_luma)

        exposure_ev = values["exposure"] * 5.0
        image = image * torch.exp2(exposure_ev).unsqueeze(-1)

        if self._has_local_tone and pre_exposure_log_luminance is not None:
            image = self._local_tone(
                image,
                pre_exposure_log_luminance,
                exposure_ev,
                values["highlights"],
                values["shadows"],
            )

        if self._has_tone_map:
            image = self._tone_map(
                image, values["contrast"], values["whites"], values["blacks"]
            )

        if self._has_colour:
            image = self._colour(image, values["vibrance"], values["saturation"])

        image = _apply_matrix(image, self._working_to_srgb)
        image = self._gamut_map_srgb(image)
        image = _linear_to_srgb(image)
        return torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
