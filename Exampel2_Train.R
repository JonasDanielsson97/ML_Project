if (file.exists("results_exp1.rds")) {
  results_exp1 <- readRDS("results_exp1.rds")
} else {

# Initiera resultatlista
results_exp1 <- vector("list", nrow(configs_exp1))
# Loopning av A till D
for (i in seq_len(nrow(configs_exp1))) {
  id    <- configs_exp1$ID[i]
  n_mel <- configs_exp1$n_mel[i]
  img_w <- configs_exp1$img_w[i]

  # Förbehandling för aktuell konfiguration (A-D) med tf_preprocess_factory()
  prep_fn <- tf_preprocess_factory(n_mel = n_mel,
                                   frame_step = 256L,
                                   fmin = 80.0,
                                   fmax = 8000.0,
                                   img_w = img_w)

  # Skapar mina Datasets med prep_fn Batch_size=32
  train_ds <- make_dataset(df_train, prep_fn, batch_size = 32L, shuffle = TRUE)
  val_ds   <- make_dataset(df_val,   prep_fn, batch_size = 32L, shuffle = FALSE)
  test_ds  <- make_dataset(df_test,  prep_fn, batch_size = 32L, shuffle = FALSE)

  # Modell, använder Base_model
  model <- Base_model(img_h = as.integer(n_mel),
                      img_w = as.integer(img_w),
                      num_classes = NUM_CLASSES)

  # Mina Callbacks
  cb_list <- list(
    callback_early_stopping(monitor = "val_loss",
                            patience = 3,
                            restore_best_weights = TRUE)
  )

  # Börjar mäta tid
  t0 <- Sys.time()
  #Tränar Data (20 epok)
  history <- model %>% fit(
    train_ds,
    validation_data = val_ds,
    epochs = 20L,
    callbacks = cb_list,
    verbose = 2
  )
  # slutar mäta tid och tar diffen.
  t1 <- Sys.time()
  train_time <- as.numeric(difftime(t1, t0, units = "secs"))

  # Hämta bästa validerings-accuracy (lägst valideringsförlust)
  val_loss <- as.numeric(history$metrics$val_loss)
  val_acc  <- as.numeric(history$metrics$val_accuracy)
  best_idx <- which.min(val_loss)
  best_val_acc <- val_acc[best_idx]

  # Utvärdera på testmängden
  test_res <- model %>% evaluate(test_ds, verbose = 0)
  test_acc <- as.numeric(test_res["accuracy"])

  # Antal parametrar i modellen
  param_count <- sum(model$count_params())

  # Sparar mina resultat i results_exp1
  results_exp1[[i]] <- tibble(
    ID              = id,
    n_mel           = n_mel,
    img_w           = img_w,
    val_accuracy    = best_val_acc,
    test_accuracy   = test_acc,
    epochs_trained  = length(val_loss),
    train_time_sec  = train_time,
    parameters      = param_count
  )
}

results_exp1 <- dplyr::bind_rows(results_exp1)
saveRDS(results_exp1, file = "results_exp1.rds")
}
