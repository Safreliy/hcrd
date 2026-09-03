args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(paste(
    "usage: run_shape_contrast_comparators_e33.R",
    "OBSERVATIONS_CSV DESIGNS_CSV OUTPUT_CSV BOOTSTRAP_LOOPS"
  ))
}

observation_path <- args[[1]]
design_path <- args[[2]]
output_path <- args[[3]]
bootstrap_loops <- as.integer(args[[4]])
if (!is.finite(bootstrap_loops) || bootstrap_loops < 1) {
  stop("BOOTSTRAP_LOOPS must be a positive integer")
}

suppressPackageStartupMessages(library(Sshaped))
suppressPackageStartupMessages(library(ShapeChange))

observations <- read.csv(observation_path, stringsAsFactors = FALSE)
designs <- read.csv(design_path, stringsAsFactors = FALSE)

completed <- character(0)
if (file.exists(output_path)) {
  previous <- read.csv(output_path, stringsAsFactors = FALSE)
  completed <- paste(previous$cell, previous$trial, sep = "::")
}

pending_output <- list()
flush_output <- function() {
  if (length(pending_output) == 0) {
    return(invisible(NULL))
  }
  frame <- do.call(rbind, pending_output)
  write.table(
    frame,
    file = output_path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(output_path),
    append = file.exists(output_path),
    quote = TRUE
  )
  pending_output <<- list()
  invisible(NULL)
}

for (row_index in seq_len(nrow(observations))) {
  cell <- observations$cell[[row_index]]
  trial <- observations$trial[[row_index]]
  key <- paste(cell, trial, sep = "::")
  if (key %in% completed) {
    next
  }

  n <- observations$n[[row_index]]
  design_name <- observations$design[[row_index]]
  selected_design <- designs[
    designs$design == design_name & designs$n == n,
  ]
  selected_design <- selected_design[order(selected_design$index), ]
  x <- selected_design$x
  y <- as.numeric(strsplit(observations$y[[row_index]], "|", fixed = TRUE)[[1]])
  if (length(x) != n || length(y) != n) {
    stop(paste("length mismatch for", key))
  }

  sshaped_started <- proc.time()[["elapsed"]]
  sshaped_fit <- tryCatch(
    withCallingHandlers(
      sshapedreg(x, y),
      warning = function(warning) invokeRestart("muffleWarning")
    ),
    error = function(error) error
  )
  sshaped_seconds <- proc.time()[["elapsed"]] - sshaped_started
  if (inherits(sshaped_fit, "error")) {
    sshaped_status <- paste("error:", conditionMessage(sshaped_fit))
    sshaped_inflection <- NA_real_
    sshaped_rss <- NA_real_
  } else {
    sshaped_status <- "ok"
    sshaped_inflection <- as.numeric(sshaped_fit$inflection)
    sshaped_rss <- as.numeric(sshaped_fit$rss)
  }

  set.seed(as.integer(observations$bootstrap_seed[[row_index]]))
  shapechange_started <- proc.time()[["elapsed"]]
  shapechange_fit <- tryCatch(
    withCallingHandlers(
      changept(
        y ~ ip(x, sh = 1),
        family = gaussian(),
        fir = TRUE,
        ci = TRUE,
        nloop = bootstrap_loops
      ),
      warning = function(warning) invokeRestart("muffleWarning")
    ),
    error = function(error) error
  )
  shapechange_seconds <- proc.time()[["elapsed"]] - shapechange_started
  if (inherits(shapechange_fit, "error")) {
    shapechange_status <- paste("error:", conditionMessage(shapechange_fit))
    shapechange_inflection <- NA_real_
    shapechange_left <- NA_real_
    shapechange_right <- NA_real_
  } else {
    shapechange_status <- "ok"
    shapechange_inflection <- as.numeric(shapechange_fit$chpt)
    shapechange_left <- as.numeric(shapechange_fit$cibt[[1]])
    shapechange_right <- as.numeric(shapechange_fit$cibt[[2]])
  }

  pending_output[[length(pending_output) + 1]] <- data.frame(
    cell = cell,
    trial = trial,
    bootstrap_seed = observations$bootstrap_seed[[row_index]],
    bootstrap_loops = bootstrap_loops,
    sshaped_status = sshaped_status,
    sshaped_inflection = sshaped_inflection,
    sshaped_rss = sshaped_rss,
    sshaped_runtime_seconds = sshaped_seconds,
    shapechange_status = shapechange_status,
    shapechange_inflection = shapechange_inflection,
    shapechange_left = shapechange_left,
    shapechange_right = shapechange_right,
    shapechange_runtime_seconds = shapechange_seconds,
    stringsAsFactors = FALSE
  )

  if (length(pending_output) >= 25) {
    flush_output()
    cat(sprintf("completed %d/%d in %s\n", row_index, nrow(observations), observation_path))
    flush.console()
  }
}

flush_output()
