if (file.exists("results_exp2.rds")) {
  results_exp2 <- readRDS("results_exp2.rds")
} else {

  results_exp2 <- vector("list", length(models_list))
  names(results_exp2) <- names(models_list)

  for (m in names(models_list)) {
    # fixa förbehandlingsfunktion för 16x16
    prep_fn <- tf_preprocess_factory(
      n_mel = 16L,
      frame_step = 256L,
      fmin = 80.0,
      fmax = 8000.0,
      img_w = 16L
    )
    # data
    train_ds <- make_dataset(df_train, prep_fn, batch_size = 32L, shuffle = TRUE)
    val_ds   <- make_dataset(df_val,   prep_fn, batch_size = 32L, shuffle = FALSE)
    test_ds  <- make_dataset(df_test,  prep_fn, batch_size = 32L, shuffle = FALSE)
    # modell
    model <- models_list[[m]](img_h = 16L, img_w = 16L, num_classes = NUM_CLASSES)
    # callbacks
    cb_list <- list(
      callback_early_stopping(monitor = "val_loss", patience = 3, restore_best_weights = TRUE)
    )
    # träna och mät tid
    t0 <- Sys.time()
    history <- model %>% fit(
      train_ds,
      validation_data = val_ds,
      epochs = 20L,
      callbacks = cb_list,
      verbose = 2
    )
    train_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    # bästa valideringsaccuracy
    val_loss <- as.numeric(history$metrics$val_loss)
    val_acc  <- as.numeric(history$metrics$val_accuracy)
    best_idx <- which.min(val_loss)
    best_val_acc <- val_acc[best_idx]
    # testprecision
    test_res <- model %>% evaluate(test_ds, verbose = 0)
    test_acc <- as.numeric(test_res["accuracy"])
    # param count
    param_count <- sum(model$count_params())
    # lagra
    results_exp2[[m]] <- tibble(
      Modell         = m,
      val_accuracy   = best_val_acc,
      test_accuracy  = test_acc,
      epochs_trained = length(val_loss),
      train_time_sec = train_time,
      parameters     = param_count
    )
    # rensa
    #clear_tf()
  }
  results_exp2 <- dplyr::bind_rows(results_exp2)
  saveRDS(results_exp2, "results_exp2.rds")
}
