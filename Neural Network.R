library(tidyr)
library(dplyr) #for utube_us manipulation
library(readr)
library(stringr)
library(ggplot2)
library(tidytext)
library(utube_us.table)
library(wordcloud) #for visualizing word clouds
library(qdap) #for word frequencies
library(tm)
library(qdap)
library(rjson)
library(keras)
library(ISLR)
library(neuralnet)
# General Goal: 
# Get the top 5 most common video categorys in the US and try to predict them using our NN model

# Todo:
# remove rows that are not top 5 from our data set. 
# Create a model with 5 input, 5 hidden and 5 output layers. 
# Test accuracy results

# Loading data in
utube_us <- read.csv('C:/Users/JD/Desktop/Trending-Youtube-MLTrending-Youtube-ML/USvideos.csv', 
                     encoding="UTF-8",
                     stringsAsFactors=FALSE,
                     na.strings=c("", "NA"))

# Data preprocessing
utube_us$video_id <- NULL # Removing video id because it does nothing
utube_us$category_id <- factor(utube_us$category_id) # Most of these variables are not gonna be used. May remove in the future. 
utube_us$channel_title <- factor(utube_us$channel_title)
utube_us$comments_disabled <- factor(utube_us$comments_disabled)
utube_us$ratings_disabled <- factor(utube_us$ratings_disabled)
utube_us$video_error_or_removed <- factor(utube_us$video_error_or_removed)
utube_us$trending_date <- as.Date(utube_us$trending_date, format = '%y.%d.%m')
utube_us$publish_time <- as.Date(utube_us$publish_time, format = '%Y-%m-%d')
utube_us$pub_to_trend <- as.numeric(utube_us$trending_date - utube_us$publish_time)


filtered_set <- utube_us[c(7,8,9,10,4)] # New dataset containing views, likes, dislikes, comment_count, and category.

# Standard test and training data splitting
n <- nrow(filtered_set)
p <- ncol(filtered_set)-1

set.seed(1) 

train <- sample(n, 0.8*n)
train_data <- filtered_set[train, 1:p]

train_labels <- filtered_set[train, p+1]
test_data <- filtered_set[-train, 1:p]
test_labels <- filtered_set[-train, p+1]

train_data <- scale(train_data)

col_means_train <- attr(train_data, "scaled:center")
col_stddevs_train <- attr(train_data, "scaled:scale")

test_data <- scale(test_data,
                   center=col_means_train,
                   scale=col_stddevs_train)

tensorflow::tf$random$set_seed(1)

train_labels <- to_categorical(as.numeric(train_labels)-1)
test_labels <- to_categorical(as.numeric(test_labels)-1)


# This Neural network is going to be used to visualize on our report. Still not finished
nn <- neuralnet(category_id~views+likes+comment_count+dislikes,
                data=train_data,
                hidden=4,
                err.fct='sse',
                linear.output=FALSE
                )
plot(nn)


# This neural network is going to be used to test for accuracy and we will also use it's plot on the report. 
model <- keras_model_sequential(layers = list(layer_dense(units = 16, activation = "relu", 
                                              input_shape = dim(train_data)[2]),
                                              layer_dense(units = ncol(train_labels), activation = "softmax")))

compile(model,
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = 'accuracy')

# Early stop function
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)


history_earlystop <- fit(model,
                         train_data, 
                         train_labels, 
                         epochs = 1000,
                         batch_size = 32, 
                         validation_split = 0.2,
                         callbacks = list(early_stop))

# Plot the history
plot(history, smooth=F)
history
score <- evaluate(model,  test_data, test_labels)