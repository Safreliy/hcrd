#!/usr/bin/env Rscript

# Convert the deeply nested RData lists to a compact, memory-mappable cache.
# This is a format conversion only: ordering, intensities and retention times
# are preserved exactly, with NaN padding beyond each recorded length.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: flatten_lcms_eic.R EIC_data.RData OUTPUT_DIR")
}
source_path <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

load(source_path)
if (!exists("EIC_list_7") || !exists("EIC_list_14")) {
  stop("expected EIC_list_7 and EIC_list_14")
}
if (!identical(names(EIC_list_7), names(EIC_list_14))) {
  stop("sample ordering differs between windows")
}
if (!all(vapply(
  seq_along(EIC_list_7),
  function(i) identical(names(EIC_list_7[[i]]), names(EIC_list_14[[i]])),
  logical(1)
))) {
  stop("peak ordering differs between windows")
}

sample_names <- names(EIC_list_7)
peak_names <- names(EIC_list_7[[1]])
if (!all(vapply(EIC_list_7, function(x) identical(names(x), peak_names), logical(1)))) {
  stop("peak ordering differs among short-window samples")
}
if (!all(vapply(EIC_list_14, function(x) identical(names(x), peak_names), logical(1)))) {
  stop("peak ordering differs among long-window samples")
}

writeLines(sample_names, file.path(output_dir, "sample_names.txt"), useBytes = TRUE)
writeLines(peak_names, file.path(output_dir, "peak_names.txt"), useBytes = TRUE)

flatten_window <- function(object, prefix) {
  cases <- unlist(object, recursive = FALSE, use.names = FALSE)
  lengths <- vapply(cases, ncol, integer(1))
  if (any(vapply(cases, nrow, integer(1)) != 2L)) {
    stop(paste(prefix, "contains a matrix without two rows"))
  }
  max_length <- max(lengths)
  count <- length(cases)
  intensity <- matrix(NaN, nrow = max_length, ncol = count)
  retention_time <- matrix(NaN, nrow = max_length, ncol = count)
  for (index in seq_len(count)) {
    length_i <- lengths[[index]]
    intensity[seq_len(length_i), index] <- cases[[index]][1L, ]
    retention_time[seq_len(length_i), index] <- cases[[index]][2L, ]
  }
  intensity_connection <- file(
    file.path(output_dir, paste0(prefix, "_intensity.f32")), "wb"
  )
  writeBin(as.vector(intensity), intensity_connection, size = 4L, endian = "little")
  close(intensity_connection)
  time_connection <- file(
    file.path(output_dir, paste0(prefix, "_retention_time.f32")), "wb"
  )
  writeBin(
    as.vector(retention_time), time_connection, size = 4L, endian = "little"
  )
  close(time_connection)
  length_connection <- file(
    file.path(output_dir, paste0(prefix, "_length.i32")), "wb"
  )
  writeBin(as.integer(lengths), length_connection, size = 4L, endian = "little")
  close(length_connection)
  c(count = count, max_length = max_length)
}

short_shape <- flatten_window(EIC_list_7, "short")
rm(EIC_list_7)
gc()
long_shape <- flatten_window(EIC_list_14, "long")
rm(EIC_list_14)
gc()

manifest <- c(
  "format=hcrd-e1-flat-v1",
  paste0("sample_count=", length(sample_names)),
  paste0("peak_count=", length(peak_names)),
  paste0("case_count=", unname(short_shape[["count"]])),
  paste0("short_max_length=", unname(short_shape[["max_length"]])),
  paste0("long_max_length=", unname(long_shape[["max_length"]])),
  "dtype=float32-little-endian",
  "ordering=sample-major-then-peak-major",
  "rows=intensity,retention_time"
)
writeLines(manifest, file.path(output_dir, "cache_manifest.txt"))
cat(paste(manifest, collapse = "\n"), "\n")
