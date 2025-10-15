compute_spec_for_plot <- function(file,
                                  frame_step = 256L,
                                  n_mel      = 64L,
                                  fmin       = 80.0,
                                  fmax       = 8000.0,
                                  img_w      = 64L) {
  
  # Läs WAV via TF (vi antar fast SR = 48 kHz för konsekvens)
  audio_bin <- tf$io$read_file(as.character(file))
  dec <- tf$audio$decode_wav(audio_bin, desired_channels = 1L)
  tf$debugging$assert_equal(dec$sample_rate, tf$cast(SAMPLE_RATE, tf$int32))
  wav <- tf$squeeze(dec$audio, axis = -1L)
  
  # STFT
  stft <- tf$signal$stft(
    wav,
    frame_length = as.integer(FRAME_LENGTH),
    frame_step   = as.integer(frame_step),
    fft_length   = as.integer(FRAME_LENGTH),
    window_fn    = tf$signal$hann_window
  )
  mag <- tf$abs(stft)  # [T, F_lin]
  
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
  mel_power <- tf$matmul(tf$square(mag), mel_w)  # [T, n_mel]
  
  # dB-normalisering per klipp: [-CLIP_DB, 0]
  ln10 <- tf$math$log(10.0)
  S_db_tf <- 10.0 * tf$math$log(mel_power + 1e-10) / ln10
  S_db_tf <- S_db_tf - tf$reduce_max(S_db_tf)      # topp=0 dB
  S_db_tf <- tf$maximum(S_db_tf, -CLIP_DB)
  
  # Skala till [0,1], transponera till [mel, time, 1], resize till [n_mel, img_w]
  S01 <- (S_db_tf + CLIP_DB) / CLIP_DB             # [-C,0] -> [0,1]
  img <- tf$expand_dims(S01, axis = -1L)           # [T, n_mel, 1]
  img <- tf$transpose(img, perm = c(1L, 0L, 2L))   # [n_mel, T, 1]
  img <- tf$image$resize(img, size = as.integer(c(as.integer(n_mel), as.integer(img_w))))
  img <- tf$squeeze(img, axis = -1L)               # [n_mel, img_w]
  
  # Tillbaka till dB för plotting (samma färgskala som övrigt)
  S_db_res <- as.array(img) * CLIP_DB - CLIP_DB    # [n_mel, img_w], i dB
  
  # Axlar: tid (sek) och mel-centers i Hz
  # Total längd i sek = (#ramar - 1) * frame_step / SR (ramstarts-approxim.)
  T_raw <- dim(as.array(mag))[1]
  total_sec <- if (T_raw > 0) (T_raw - 1) * (frame_step / SAMPLE_RATE) else 0
  t_sec <- seq(0, total_sec, length.out = img_w)
  
  mel_edges   <- seq(hz_to_mel(fmin), hz_to_mel(fmax), length.out = n_mel + 1L)
  mel_centers <- 0.5 * (mel_edges[-1L] + mel_edges[-length(mel_edges)])
  f_hz <- mel_to_hz(mel_centers)
  
  # Returnera dB-matris som matchar modellens indata-dimensioner + axlar
  list(S_db = S_db_res,            # [n_mel x img_w], dB i [-CLIP_DB, 0]
       t    = t_sec,               # längd img_w
       f    = f_hz)                # längd n_mel
}
