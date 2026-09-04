"""Shape-contrast inversion for convex-to-concave transition sets."""

from .heteroskedastic import (
    GaussianHeteroskedasticEnvelope,
    balanced_residual_block_labels,
    gaussian_heteroskedastic_upper_envelope,
)
from .identified_set import (
    DesignIdentifiedTransitionSet,
    design_identified_transition_set,
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
from .projection import (
    PointwiseProjectionConfidenceSet,
    gaussian_pointwise_shape_projection,
)
from .replicated import ReplicatedShapeContrastBand, replicated_t_shape_band

__all__ = [
    "ContrastScale",
    "DesignIdentifiedTransitionSet",
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
    "design_identified_transition_set",
    "dyadic_block_sizes",
    "gaussian_block_upper_scale",
    "gaussian_bonferroni_shape_band",
    "gaussian_calibrated_shape_band",
    "gaussian_heteroskedastic_upper_envelope",
    "gaussian_pointwise_shape_projection",
    "gaussian_projection_upper_scale",
    "invert_s_shaped_inflection",
    "replicated_t_shape_band",
]
