# ---- setup-constants ----
SAMPLE_RATE  <- 48000.0
NUM_CLASSES  <- 10L
FRAME_LENGTH <- 1024L
FRAME_STEP   <- 256L
FMIN         <- 80.0
FMAX         <- 8000.0
CLIP_DB      <- 80.0

AUTOTUNE <- tryCatch(tf$data$AUTOTUNE,
  error = function(...) tf$data$experimental$AUTOTUNE
)

tf_preprocess_factory <- function(frame_step = 256L,
                                  n_mel = 64L,
                                  fmin = 80.0,
                                  fmax = 8000.0,
                                  img_w = 64L) {
  
  function(elem) {
    file  <- elem$path
    label <- elem$label
    
    # Läs WAV (TF) och säkerställ fast SR (48 kHz)
    audio_bin <- tf$io$read_file(file)
    dec <- tf$audio$decode_wav(audio_bin, desired_channels = 1L)
    tf$debugging$assert_equal(dec$sample_rate, tf$cast(SAMPLE_RATE, tf$int32))
    wav <- tf$squeeze(dec$audio, axis = -1L)   # [samples]
    
    # STFT
    stft <- tf$signal$stft(
      wav,
      frame_length = as.integer(FRAME_LENGTH),
      frame_step   = as.integer(frame_step),
      fft_length   = as.integer(FRAME_LENGTH),
      window_fn    = tf$signal$hann_window
    )
    mag <- tf$abs(stft)                         # [time, freq_lin]
    
    # Lin -> mel (power)
    num_spec_bins <- as.integer(FRAME_LENGTH/2 + 1L)
    mel_w <- tf$signal$linear_to_mel_weight_matrix(
      num_mel_bins         = as.integer(n_mel),
      num_spectrogram_bins = num_spec_bins,
      sample_rate          = SAMPLE_RATE,
      lower_edge_hertz     = fmin,
      upper_edge_hertz     = fmax,
      dtype                = tf$float32
    )
    mel_power <- tf$matmul(tf$square(mag), mel_w)    # [time, n_mel]
    
    # dB + normalisering per klipp: [-CLIP_DB, 0]
    ln10 <- tf$math$log(10.0)
    S_db <- 10.0 * tf$math$log(mel_power + 1e-10) / ln10
    S_db <- S_db - tf$reduce_max(S_db)               # topp = 0 dB
    S_db <- tf$maximum(S_db, -CLIP_DB)               # klipp
    
    # Skala till [0,1] för nätet
    S01 <- (S_db + CLIP_DB) / CLIP_DB                # [-C,0] -> [0,1]
    
    # [time, mel] -> [mel, time, 1] och resiza till [n_mel, img_w, 1]
    img <- tf$expand_dims(S01, axis = -1L)           # [time, mel, 1]
    img <- tf$transpose(img, perm = c(1L, 0L, 2L))   # [mel, time, 1]
    img <- tf$image$resize(img, size = as.integer(c(as.integer(n_mel), as.integer(img_w))))
    img <- tf$clip_by_value(img, 0, 1)
    img <- tf$ensure_shape(img, shape = reticulate::tuple(
      as.integer(n_mel), as.integer(img_w), 1L
    ))
    
    # One-hot label
    y <- tf$one_hot(tf$cast(label, tf$int32), depth = as.integer(NUM_CLASSES))
    
    reticulate::tuple(img, y)
  }
}
