plot_spec_db_index <- function(S_db, t, f_hz,
                               title = NULL,
                               show_axes = TRUE,
                               show_legend = TRUE,
                               title_size = NULL) {
  stopifnot(is.matrix(S_db))
  n_mel <- nrow(S_db); img_w <- ncol(S_db)
  stopifnot(length(t) == img_w, length(f_hz) == n_mel)

  df <- as.data.frame(t(S_db))
  colnames(df) <- paste0("m", seq_len(n_mel))
  df$Time <- t
  long <- tidyr::pivot_longer(df, -Time, names_to = "mel", values_to = "dB")
  long$mel <- as.integer(sub("m", "", long$mel))

  # --- fasta Hz-ticks, men etiketter = TARGET-värdena ---
  targets <- c(100, 1000, 3000, 8000)
  # Hittar närmaste pixel till target
  brks_idx <- sapply(targets, function(h) which.min(abs(f_hz - h))) 
  brks_idx <- pmax(1L, pmin(n_mel, as.integer(brks_idx)))  # clamp till [1, n_mel]
  y_breaks <- unique(brks_idx)
  # etiketter = target-värden, inte f_hz[brks]
  y_labels <- as.character(targets[match(y_breaks, brks_idx)])

  p <- ggplot2::ggplot(long, ggplot2::aes(x = Time, y = mel, fill = dB)) +
    ggplot2::geom_raster() +
    ggplot2::scale_y_continuous(breaks = y_breaks, labels = y_labels) +
    ggplot2::scale_fill_viridis_c(limits = c(-CLIP_DB, 0),
                                  breaks = seq(-CLIP_DB, 0, by = 20),
                                  name   = "dB.FS") +
    ggplot2::labs(x = if (show_axes) "Tid (s)" else NULL,
                  y = if (show_axes) "Frekvens (Hz)" else NULL,
                  title = title) +
    ggplot2::theme_minimal()

  if (!show_axes) {
    p <- p + ggplot2::theme(
      axis.title = ggplot2::element_blank(),
      axis.text  = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank()
    )
  }
  if (!show_legend) p <- p + ggplot2::theme(legend.position = "none")
  if (!is.null(title_size)) p <- p + ggplot2::theme(plot.title = ggplot2::element_text(size = title_size))
  p
}

# Exempel:
# viz <- compute_spec_for_plot(df$path[1], n_mel=64L, img_w=64L)
# plot_spec_db_index(viz$S_db, viz$t, viz$f, title = "Log-mel (64x64), dB")
