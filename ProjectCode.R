#Library
library(dplyr)

#Read csv file and set youtube data
youtube <- read.csv(file.choose(), header = TRUE)

#Get overall data type and dim
glimpse(youtube)

#Put youtube columns and rows into n and p objects
n <- nrow(youtube)
p <- ncol(youtube)-1

#Training and Test Data
set.seed(1)
train <- sample(n, 0.8*n)
train_data <- youtube[train, -1]
train_labels <- youtube[train, 1]
test_data <- youtube[-train, -1]
test_labels <- youtube[-train, 1]

#Create a Random Forest
