# Dump the bixverse devforge param specs to JSON, flattened the way the
# generated `params_*()` constructors flatten them before Rust sees the list.
#
# Usage: Rscript dev/dump_param_specs.R <bixverse checkout> <out.json> <name>...
#
# Every merge block is resolved recursively (base spec, then `drop`, then
# `overrides`, then the merge field's own default), exactly as
# devforge::emit_merge() layers them. Overrides that are expressions of other
# fields (`k = neighbours_within_batch * 2L`) cannot be evaluated here and are
# passed through as R source in `derived`. Every resolved default is then
# checked against the checkout's generated `params_*()`, so the dump cannot
# drift from what R actually hands to Rust.

suppressMessages(library(devforge))

args <- commandArgs(trailingOnly = TRUE)
bixverse_dir <- args[[1]]
out_path <- args[[2]]
wanted <- args[-(1:2)]

spec_env <- new.env()
for (f in list.files(file.path(bixverse_dir, "inst", "params"), full.names = TRUE)) {
  sys.source(f, envir = spec_env)
}
specs <- mget(grep("^spec_", ls(spec_env), value = TRUE), envir = spec_env)
names(specs) <- vapply(specs, `[[`, character(1), "name")

gen_env <- new.env()
for (f in c("params-prelude-generated.R", "params-generated.R")) {
  sys.source(file.path(bixverse_dir, "R", f), envir = gen_env)
}

# Is this merge resolvable from spec data alone? A computed base (SCENIC picks
# its learner block at call time) is not.
is_static_merge <- function(field) {
  is.character(field$from) && length(field$from) == 1L
}

# Split an override object into constant defaults and derived expressions.
split_overrides <- function(ov) {
  consts <- list()
  derived <- list()
  if (is.null(ov)) {
    return(list(consts = consts, derived = derived))
  }
  if (is.language(ov)) {
    stopifnot(identical(ov[[1]], as.name("list")))
    items <- as.list(ov)[-1]
  } else {
    items <- ov
  }
  for (k in names(items)) {
    v <- items[[k]]
    if (is.language(v)) {
      derived[[k]] <- paste(deparse(v), collapse = " ")
    } else {
      consts[k] <- list(v)
    }
  }
  list(consts = consts, derived = derived)
}

field_record <- function(name, field, origin) {
  list(
    name = name,
    type = field$type,
    default = field$default,
    range = field$range,
    null_ok = isTRUE(field$null_ok),
    len = as.character(field$len),
    choices = field$choices,
    letter = field$letter,
    check_as = field$check_as,
    required = isTRUE(field$required),
    doc = field$doc %||% "",
    origin = origin,
    derived = NULL
  )
}

resolve_fields <- function(spec, origin = NULL) {
  out <- list()
  for (fname in names(spec$fields)) {
    field <- spec$fields[[fname]]
    if (!identical(field$type, "merge")) {
      out[[fname]] <- field_record(fname, field, origin)
      next
    }
    if (!is_static_merge(field)) {
      stop(sprintf("dynamic merge %s$%s", spec$name, fname))
    }
    block <- resolve_fields(specs[[field$from]], origin = origin %||% fname)
    for (d in field$drop) {
      block[[d]] <- NULL
    }
    ov <- split_overrides(field$overrides)
    # modifyList adds keys the base lacks (sc_vision's knn gains nn_max_iter),
    # so they become fields typed from their value.
    extra <- c(ov$consts, field$default)
    for (k in setdiff(names(extra), names(block))) {
      v <- extra[[k]]
      type <- if (is.logical(v)) "lgl" else if (is.integer(v)) "int" else if (is.numeric(v)) "dbl" else "chr"
      block[[k]] <- field_record(k, list(type = type, len = 1L), origin %||% fname)
    }
    if (length(setdiff(names(ov$derived), names(block))) > 0L) {
      stop(sprintf("%s$%s derives an unknown element", spec$name, fname))
    }
    for (k in names(ov$consts)) {
      block[[k]]$default <- ov$consts[[k]]
    }
    for (k in names(ov$derived)) {
      block[[k]]$derived <- ov$derived[[k]]
    }
    # The merge field's own default is the caller's list, applied last.
    for (k in names(field$default)) {
      block[[k]]$default <- field$default[[k]]
      block[[k]]$derived <- NULL
    }
    clash <- intersect(names(block), names(out))
    if (length(clash) > 0L) {
      stop(sprintf("name clash in %s: %s", spec$name, paste(clash, collapse = ", ")))
    }
    out <- c(out, block)
  }
  out
}

check_against_generated <- function(spec, fields) {
  if (any(vapply(fields, `[[`, logical(1), "required"))) {
    return(invisible(NULL))
  }
  # Some defaults fail their own extra_ctor on purpose (sc_empty_droplets needs
  # a column for its default method); nothing to compare against then.
  actual <- tryCatch(
    suppressWarnings(do.call(gen_env[[paste0("params_", spec$name)]], list())),
    error = function(e) NULL
  )
  if (is.null(actual)) {
    return(invisible(NULL))
  }
  stopifnot(setequal(names(actual), names(fields)))
  for (k in names(fields)) {
    f <- fields[[k]]
    if (!is.null(f$derived)) {
      next
    }
    if (!identical(unname(actual[[k]]), unname(f$default)) &&
      !isTRUE(all.equal(actual[[k]], f$default))) {
      stop(sprintf("default drift %s$%s", spec$name, k))
    }
  }
}

res <- list()
for (name in wanted) {
  spec <- specs[[name]]
  if (is.null(spec)) {
    stop(sprintf("unknown spec %s", name))
  }
  fields <- tryCatch(resolve_fields(spec), error = function(e) {
    message(conditionMessage(e))
    NULL
  })
  if (is.null(fields)) {
    res[[name]] <- list(name = name, handwritten = TRUE)
    next
  }
  check_against_generated(spec, fields)
  # Vectors declared as `+` / `*` stay arrays in the JSON even at length 1.
  fields <- lapply(fields, function(f) {
    if (!identical(f$len, "1") && !is.null(f$default)) {
      f$default <- I(f$default)
    }
    f
  })
  res[[name]] <- list(
    name = name,
    title = spec$title %||% "",
    description = spec$description %||% "",
    handwritten = FALSE,
    fields = unname(fields)
  )
}

writeLines(
  jsonlite::toJSON(res, auto_unbox = TRUE, null = "null", digits = I(17), pretty = TRUE),
  out_path
)
