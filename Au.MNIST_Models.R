
## Base_Model
Base_model <- function(img_h, img_w, num_classes = NUM_CLASSES) {
  inputs <- layer_input(shape = c(as.integer(img_h), as.integer(img_w), 1L))
  
  # NN
  x <- inputs %>%
    layer_conv_2d(16L, c(3,3), activation="relu", padding="same") %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(32L, c(3,3), activation="relu", padding="same") %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(64L, c(3,3), activation="relu", padding="same") %>%
    layer_global_average_pooling_2d() %>%
    layer_dropout(0.2)
  outputs <- layer_dense(x, units = num_classes, activation = "softmax")
  model <- keras_model(inputs, outputs)
  
  # Compiler
  model %>% compile(optimizer="adam", loss="categorical_crossentropy", metrics=list("accuracy"))
  model
}

## Model_Deep
Model_Deep <- function(img_h, img_w, num_classes = NUM_CLASSES) {
  inputs <- layer_input(shape = c(as.integer(img_h), as.integer(img_w), 1L))

  x <- inputs %>%
    layer_conv_2d(16L, c(3,3), activation="relu", padding="same") %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(32L, c(3,3), activation="relu", padding="same") %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(32L, c(3,3), activation="relu", padding="same") %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(64L, c(3,3), activation="relu", padding="same") %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(64L, c(3,3), activation="relu", padding="same") %>%
    layer_global_average_pooling_2d() %>%
    layer_dropout(0.2)

  outputs <- layer_dense(x, units = num_classes, activation = "softmax")
  model <- keras_model(inputs, outputs)

  model %>% compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )
  model
}


## Model_BN
Model_BN <- function(img_h, img_w, num_classes = NUM_CLASSES) {
  inputs <- layer_input(shape = c(as.integer(img_h), as.integer(img_w), 1L))

  x <- inputs %>%
    layer_conv_2d(16L, c(3,3), padding="same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(32L, c(3,3), padding="same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(64L, c(3,3), padding="same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_global_average_pooling_2d() %>%
    layer_dropout(0.2)

  outputs <- layer_dense(x, units = num_classes, activation = "softmax")
  model <- keras_model(inputs, outputs)

  model %>% compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )
  model
}


## Model_Stride
Model_Stride <- function(img_h, img_w, num_classes = NUM_CLASSES) {
  inputs <- layer_input(shape = c(as.integer(img_h), as.integer(img_w), 1L))

  x <- inputs %>%
    layer_conv_2d(16L, c(3,3), strides=2, activation="relu", padding="same") %>%
    layer_conv_2d(32L, c(3,3), strides=2, activation="relu", padding="same") %>%
    layer_conv_2d(64L, c(3,3), strides=2, activation="relu", padding="same") %>%
    layer_global_average_pooling_2d() %>%
    layer_dropout(0.2)

  outputs <- layer_dense(x, units = num_classes, activation = "softmax")
  model <- keras_model(inputs, outputs)

  model %>% compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )
  model
}


