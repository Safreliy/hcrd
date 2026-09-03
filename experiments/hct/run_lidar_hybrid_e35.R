args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: run_lidar_hybrid_e35.R OUTPUT_DIR BOOTSTRAP_LOOPS SEED")
}

output_dir <- args[[1]]
bootstrap_loops <- as.integer(args[[2]])
bootstrap_seed <- as.integer(args[[3]])
if (!is.finite(bootstrap_loops) || bootstrap_loops < 1) {
  stop("BOOTSTRAP_LOOPS must be positive")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(SemiPar))
suppressPackageStartupMessages(library(Sshaped))
suppressPackageStartupMessages(library(ShapeChange))
data(lidar)

x <- as.numeric(lidar$range)
y <- -as.numeric(lidar$logratio)
x_normalized <- (x - min(x)) / (max(x) - min(x))

sshaped_fit <- sshapedreg(x, y)
set.seed(bootstrap_seed)
shapechange_fit <- tryCatch(
  withCallingHandlers(
    changept(
      y ~ ip(x_normalized, sh = 1),
      family = gaussian(),
      fir = TRUE,
      ci = TRUE,
      nloop = bootstrap_loops
    ),
    warning = function(warning) invokeRestart("muffleWarning")
  ),
  error = function(error) error
)

write.csv(
  data.frame(
    range = x,
    x_normalized = x_normalized,
    minus_logratio = y,
    sshaped_fitted = as.numeric(sshaped_fit$fitted)
  ),
  file.path(output_dir, "lidar_data_and_fit.csv"),
  row.names = FALSE
)

if (inherits(shapechange_fit, "error")) {
  shapechange_status <- paste("error:", conditionMessage(shapechange_fit))
  shapechange_point <- NA_real_
  shapechange_left <- NA_real_
  shapechange_right <- NA_real_
} else {
  shapechange_status <- "ok"
  shapechange_point <- as.numeric(shapechange_fit$chpt)
  shapechange_left <- as.numeric(shapechange_fit$cibt[[1]])
  shapechange_right <- as.numeric(shapechange_fit$cibt[[2]])
}

write.csv(
  data.frame(
    sshaped_point_metres = as.numeric(sshaped_fit$inflection),
    sshaped_rss = as.numeric(sshaped_fit$rss),
    shapechange_status = shapechange_status,
    shapechange_point_normalized = shapechange_point,
    shapechange_left_normalized = shapechange_left,
    shapechange_right_normalized = shapechange_right,
    bootstrap_loops = bootstrap_loops,
    bootstrap_seed = bootstrap_seed
  ),
  file.path(output_dir, "lidar_external_fits.csv"),
  row.names = FALSE
)

