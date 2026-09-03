"""Shape-contrast inversion for convex-to-concave transition sets."""

from .heteroskedastic import (
    GaussianHeteroskedasticEnvelope,
    balanced_residual_block_labels,
    gaussian_heteroskedastic_upper_envelope,
)
from .inference import (
    ContrastScale,
    GaussianShapeContrastCalibration,
    InflectionConfidenceSet,
    ShapeContrastBand,
    ShapeContrastFamily,
    build_shape_contrast_family,
    calibrate_gaussian_shape_contrast_max,
    dyadic_block_sizes,
    gaussian_bonferroni_shape_band,
    gaussian_calibrated_shape_band,
    invert_s_shaped_inflection,
)
from .noise_scale import (
    GaussianProjectionScaleBound,
    consecutive_block_design,
    gaussian_block_upper_scale,
    gaussian_projection_upper_scale,
)
from .replicated import ReplicatedShapeContrastBand, replicated_t_shape_band
from .projection import (
    PointwiseProjectionConfidenceSet,
    gaussian_pointwise_shape_projection,
)

__all__ = [
    "ContrastScale",
    "GaussianHeteroskedasticEnvelope",
    "GaussianProjectionScaleBound",
    "GaussianShapeContrastCalibration",
    "InflectionConfidenceSet",
    "PointwiseProjectionConfidenceSet",
    "ReplicatedShapeContrastBand",
    "ShapeContrastBand",
    "ShapeContrastFamily",
    "balanced_residual_block_labels",
    "build_shape_contrast_family",
    "calibrate_gaussian_shape_contrast_max",
    "consecutive_block_design",
    "dyadic_block_sizes",
    "gaussian_block_upper_scale",
    "gaussian_bonferroni_shape_band",
    "gaussian_calibrated_shape_band",
    "gaussian_heteroskedastic_upper_envelope",
    "gaussian_projection_upper_scale",
    "gaussian_pointwise_shape_projection",
    "invert_s_shaped_inflection",
    "replicated_t_shape_band",
]
