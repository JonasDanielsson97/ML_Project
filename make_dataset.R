make_dataset <- function(df_split, preprocess_fn, batch_size = 32L, shuffle = TRUE) {
  ds <- tensor_slices_dataset(list(
    path  = as.character(df_split$path),
    label = as.integer(df_split$y)
  )) %>% dataset_map(preprocess_fn, num_parallel_calls = AUTOTUNE)
  
  # Lägger till shuffle möjlighet
  if (shuffle) ds <- ds %>% dataset_shuffle(buffer_size = as.integer(min(nrow(df_split), 10000L)))
  ds %>% dataset_batch(as.integer(batch_size)) %>% dataset_prefetch(AUTOTUNE)
}
