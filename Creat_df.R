## Index files + labels (speaker from folder, digit from filename)
library(tfdatasets)
data_dir   <- normalizePath("ML_kurs/Project/Audio_MNIST", winslash = "/")

# list .wav files
wav_files <- list.files(data_dir, pattern = "\\.(wav|WAV)$", recursive = TRUE, full.names = TRUE)
stopifnot(length(wav_files) > 0)

get_digit   <- function(p) as.integer(strsplit(basename(p), "_", fixed = TRUE)[[1]][1])
get_speaker <- function(p) basename(dirname(p))  # "03" from .../03/5_03_12.wav

digits   <- vapply(wav_files, get_digit,   integer(1))
speakers <- vapply(wav_files, get_speaker, character(1))

df <- data.frame(path = wav_files, y = digits, spk = speakers, stringsAsFactors = FALSE)

# Checks
# table(df$y)      # expect ~3000 per digit (60 spk * 50 utt)
# length(unique(df$spk))  # expect 60
# nrow(df)         # expect ~30000


# Meta_df
meta <- jsonlite::fromJSON(file.path(data_dir, "audioMNIST_meta.txt"))
meta_df <- do.call(rbind, lapply(names(meta), function(id) {
  cbind(speaker=id, as.data.frame(meta[[id]], stringsAsFactors=FALSE))
}))
meta_df <- meta_df[,c("speaker","age", "gender")]
colnames(meta_df) <- c("spk","age", "gender")
rm(meta)
