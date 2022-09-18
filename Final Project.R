rm(list=ls())
library(keras)
library(tensorflow)

# Need to do this to prevent weird bug issues
tf$compat$v1$disable_eager_execution()

# Load tensorflow backend to objectt K
K <- backend()

# Data Preparation -----------------------------------------------------

batch_size <- 128
num_classes <- 3
epochs <- 12

# Input image dimensions
img_rows <- 32
img_cols <- 32

# Load the data
load('deers_frogs_trucks.Rdata')

# Define input shape
input_shape <- c(img_rows, img_cols, 3)

# Print number of rows
cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Define inputs
inputs = layer_input(shape = input_shape)

# Model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu',name='Conv_last') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = num_classes, activation = 'softmax')

summary(model)


# Specify optimizer, loss function, metrics
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 10,
                                            verbose = 0, mode = "auto"),
                    callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch' ,save_best_only = TRUE)
)

# Running optimization
history <- model %>% fit(
  x_train, y_train,
  epochs = 50, batch_size = 256,
  validation_split = 0.2,
  callbacks = callback_specs
)

# Load the saved best mode
model_best = load_model_hdf5('best_model.hdf5',compile=TRUE)

model_best %>% evaluate(x_test,y_test)

# Compute the predicted value
p_hat_test = model_best %>% predict(x_test)
y_hat_test = apply(p_hat_test,1,which.max)

# Evaluate the model performance
model %>% evaluate(x_test, y_test)
y_true = apply(y_test,1,which.max)
sum(y_hat_test==y_true)/length(y_true) #do we get the same result?

# Multi-class ROC
library(pROC)

multiclass.roc(y_true,y_hat_test)
model %>% evaluate(x_test, y_test)

sum(y_hat_test==y_true)/length(y_true)

multiclass.roc(y_true,y_hat_test)


# Look at case 17
test_case_to_look <- 17 # Case 17
number_of_filters <- 64 #number of filers for last layer

last_conv_layer <- model_best %>% get_layer("Conv_last") 

target_output <- model_best$output[, which.max(y_test[test_case_to_look,])] 

grads <- K$gradients(target_output, last_conv_layer$output)[[1]]

pooled_grads <- K$mean(grads, axis = c(1L, 2L))
compute_them <- K$`function`(list(model_best$input), 
                             list(pooled_grads, last_conv_layer$output[1,,,])) 

# The input image has to be a 4D array
x_test_example <- x_test[test_case_to_look,,,]
dim(x_test_example) <- c(1,dim(x_test_example))

# True Label
which.max(y_test[test_case_to_look,])

# Original Label
which.max(model_best %>% predict(x_test_example))

# Computing the importance and gradient map for each filter
c(pooled_grads_value, conv_layer_output_value) %<-% compute_them(list(x_test_example))

# Computing the Activation Map
for (i in 1:number_of_filters) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)

# Normalizing the activation map
heatmap <- pmax(heatmap, 0) 
heatmap <- (heatmap - min(heatmap))/ (max(heatmap)-min(heatmap)) 

library(magick) 
library(viridis) 

write_heatmap <- function(heatmap, filename, width = 32, height = 32,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

image <- image_read(x_test[test_case_to_look,,,]) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue =255)
write_heatmap(heatmap, "overlay.png", 
              width = 32, height = 32, bg = NA, col = pal_col) 

###########################
## Creating the Final Image
###########################
par(mfrow=c(1,2))
image %>% plot()
image_read("overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "35") %>%
  plot() 

# Look at case 13
test_case_to_look <- 13 # Case 13
number_of_filters <- 64 #number of filers for last layer

last_conv_layer <- model_best %>% get_layer("Conv_last") 

target_output <- model_best$output[, which.max(y_test[test_case_to_look,])] 

grads <- K$gradients(target_output, last_conv_layer$output)[[1]]

pooled_grads <- K$mean(grads, axis = c(1L, 2L))
compute_them <- K$`function`(list(model_best$input), 
                             list(pooled_grads, last_conv_layer$output[1,,,])) 

# The input image has to be a 4D array
x_test_example <- x_test[test_case_to_look,,,]
dim(x_test_example) <- c(1,dim(x_test_example))

# True Label
which.max(y_test[test_case_to_look,])

# Original Label
which.max(model_best %>% predict(x_test_example))

# Computing the importance and gradient map for each filter
c(pooled_grads_value, conv_layer_output_value) %<-% compute_them(list(x_test_example))

# Computing the Activation Map
for (i in 1:number_of_filters) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)

# Normalizing the activation map
heatmap <- pmax(heatmap, 0) 
heatmap <- (heatmap - min(heatmap))/ (max(heatmap)-min(heatmap)) 

library(magick) 
library(viridis) 

write_heatmap <- function(heatmap, filename, width = 32, height = 32,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

image <- image_read(x_test[test_case_to_look,,,]) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue =255)
write_heatmap(heatmap, "overlay.png", 
              width = 32, height = 32, bg = NA, col = pal_col) 

###########################
## Creating the Final Image
###########################
par(mfrow=c(1,2))
image %>% plot()
image_read("overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "35") %>%
  plot() 

# Look at case 11
test_case_to_look <- 11 # Case 11
number_of_filters <- 64 #number of filers for last layer

last_conv_layer <- model_best %>% get_layer("Conv_last") 

target_output <- model_best$output[, which.max(y_test[test_case_to_look,])] 

grads <- K$gradients(target_output, last_conv_layer$output)[[1]]

pooled_grads <- K$mean(grads, axis = c(1L, 2L))
compute_them <- K$`function`(list(model_best$input), 
                             list(pooled_grads, last_conv_layer$output[1,,,])) 

# The input image has to be a 4D array
x_test_example <- x_test[test_case_to_look,,,]
dim(x_test_example) <- c(1,dim(x_test_example))

# True Label
which.max(y_test[test_case_to_look,])

# Original Label
which.max(model_best %>% predict(x_test_example))

# Computing the importance and gradient map for each filter
c(pooled_grads_value, conv_layer_output_value) %<-% compute_them(list(x_test_example))

# Computing the Activation Map
for (i in 1:number_of_filters) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)

# Normalizing the activation map
heatmap <- pmax(heatmap, 0) 
heatmap <- (heatmap - min(heatmap))/ (max(heatmap)-min(heatmap)) 

library(magick) 
library(viridis) 

write_heatmap <- function(heatmap, filename, width = 32, height = 32,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

image <- image_read(x_test[test_case_to_look,,,]) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue =255)
write_heatmap(heatmap, "overlay.png", 
              width = 32, height = 32, bg = NA, col = pal_col) 

###########################
## Creating the Final Image
###########################
par(mfrow=c(1,2))
image %>% plot()
image_read("overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "35") %>%
  plot() 


# Question 2
# Load the data
load('deers_frogs_trucks.Rdata')

# Print number of rows
cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(tensorflow)
library(keras)

# Parameters --------------------------------------------------------------
batch_size <- 100L
latent_dim <- 32L
intermediate_dim <- 64L
epochs <- 75L
epsilon_std <- 1.0

# Define input shape
input_shape <- c(img_rows, img_cols, 3)


# Model definition --------------------------------------------------------
x <- layer_input(shape = input_shape)
h <- layer_conv_2d(x,filters = 32, kernel_size = c(3,3), activation = 'relu',
              padding='same') %>%
     layer_max_pooling_2d(pool_size = c(2, 2)) %>%
     layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
     layer_max_pooling_2d(pool_size = c(2, 2)) %>%
     layer_flatten() %>%
     layer_dense(intermediate_dim, activation = "relu") %>%
     layer_dense(intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)
sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean=0.,
    stddev=epsilon_std
  )
  z_mean + k_exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

# Decoder from Z to Y
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_p <- layer_dense(units = 3, activation = 'softmax')
h_decoded <- decoder_h(z)
y_decoded_p <- decoder_p(h_decoded)

# End-to-end variational model
vnn <- keras_model(x, y_decoded_p)

# Encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# Defining loss and complie
vnn_loss <- function(x, x_decoded_mean){
  cat_loss <- loss_categorical_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  cat_loss + kl_loss
}

vnn %>% compile(optimizer = "adam", loss = vnn_loss, metrics = c('accuracy'))

# Model training ----------------------------------------------------------
vnn %>% fit(
  x_train, y_train,
  shuffle = TRUE,
  epochs = epochs,
  batch_size = batch_size,
  validation_data = list(x_train,y_train)
)

# Evaluate the test data
vnn %>% evaluate(x_test,y_test)

# Compute the predicted value
p_hat_test = vnn %>% predict(x_test)
y_hat_test = apply(p_hat_test,1,which.max)

# Evaluate the model performance
vnn %>% evaluate(x_test, y_test)
y_true = apply(y_test,1,which.max)
sum(y_hat_test==y_true)/length(y_true) #do we get the same result?

# Multi-class ROC
library(pROC)

multiclass.roc(y_true,y_hat_test)
vnn %>% evaluate(x_test, y_test)

sum(y_hat_test==y_true)/length(y_true)

multiclass.roc(y_true,y_hat_test)

# vnn prediction
x_test_decoded <- array(NA,dim=c(3000,3,1000))
for(i in 1:1000)
{
  x_test_decoded[,,i] <- predict(vnn, x_test)
}

# Look at Case #18
par(mfrow=c(1,2))

test_case_to_look <- 18 
image <- image_read(x_test[test_case_to_look,,,]) 
image %>% plot()

prob_to_plot <- t(x_test_decoded[18,,])
boxplot(prob_to_plot)

# Look at Case #24
par(mfrow=c(1,2))

test_case_to_look <- 24 
image <- image_read(x_test[test_case_to_look,,,]) 
image %>% plot()

prob_to_plot <- t(x_test_decoded[24,,])
boxplot(prob_to_plot)


# MC Dropout
# Setting up tuning parameters
DropoutRate <- 0.5
tau <- 0.5
keep_prob <- 1-DropoutRate
n_train <- nrow(x_train)
penalty_weight <- keep_prob/(tau* n_train)
penalty_intercept <- 1/(tau* n_train)

# Setting up dropout from the beginning
dropout_1 <- layer_dropout(rate = DropoutRate)
dropout_2 <- layer_dropout(rate = DropoutRate)

# Define input shape
input_shape <- c(img_rows, img_cols, 3)

# Setting up Neural Network Model
inputs = layer_input(shape = input_shape)
output <- inputs %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu',
              kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  dropout_1(training=TRUE) %>%
  layer_dense(units = 64, activation = 'relu',
              kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept)) %>%
  dropout_2(training=TRUE) %>%
  layer_dense(units = 3, activation = 'softmax',
              kernel_regularizer=regularizer_l2(penalty_weight), bias_regularizer=regularizer_l2(penalty_intercept))
model <- keras_model(inputs, output)
summary(model)

# Specify optimizer, loss function, metrics
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Set up early stopping
callback_specs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 10,
                                            verbose = 0, mode = "auto"),
                    callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch' ,save_best_only = TRUE)
)

# Running optimization
history <- model %>% fit(
  x_train, y_train,
  epochs = 50, batch_size = 200,
  validation_split = 0.2,
  callbacks = callback_specs
)

# Load the saved best model
model_best = load_model_hdf5('best_model.hdf5',compile=TRUE)

# Evaluate the test data
model_best %>% evaluate(x_test,y_test)

# Compute the predicted value
p_hat_test = model_best %>% predict(x_test)
y_hat_test = apply(p_hat_test,1,which.max)

# Evaluate the model performance
model_best %>% evaluate(x_test, y_test)
y_true = apply(y_test,1,which.max)
sum(y_hat_test==y_true)/length(y_true) #do we get the same result?

# Multi-class ROC
library(pROC)

multiclass.roc(y_true,y_hat_test)
model_best %>% evaluate(x_test, y_test)

sum(y_hat_test==y_true)/length(y_true)

multiclass.roc(y_true,y_hat_test)

# Prediction via mcdropout sampling
mc.sample=1000

testPredict=array(NA,dim=c(nrow(x_test),3,mc.sample))
for(i in 1:mc.sample)
{
  testPredict[,,i]=model_best %>% predict(x_test)
}

par(mfrow=c(1,2))

# Look at Case #16
test_case_to_look <- 16 
image <- image_read(x_test[test_case_to_look,,,]) 
image %>% plot()

prob_to_plot <- t(testPredict[16,,])
boxplot(prob_to_plot)


par(mfrow=c(1,2))

# Look at Case #17
test_case_to_look <- 17
image <- image_read(x_test[test_case_to_look,,,]) 
image %>% plot()

prob_to_plot <- t(testPredict[17,,])
boxplot(prob_to_plot)



