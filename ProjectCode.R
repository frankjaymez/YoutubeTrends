#Library
library(dplyr)
library(tree)
library(randomForest)

#Read csv file and set youtube data
youtube <- read.csv(file.choose(), header = TRUE)

#Randomly select a subset of data
youtube_sample <- youtube[sample(1:nrow(youtube), 500, replace = FALSE),]

#Predata Processing
youtube_sample$category_id <- as.factor(youtube_sample$category_id)

#Get unique category ID
unique_id <- youtube[, c(5)]
unique_id <- unique(unique_id)

#Get overall data type and dim
glimpse(youtube)

#Get variables likes, dislikes, count, category, and views
youtube_var <- youtube_sample[, c(5, 8, 9, 10, 11)]

#Classification Tree
tree.youtube = tree(category_id ~ ., data = youtube_var)
summary(tree.youtube)
plot(tree.youtube)
text(tree.youtube, pretty = 0)
tree.youtube

#Test Classification Error
RNGkind(sample.kind = "Rounding")
set.seed(2)
train = sample(1:nrow(youtube_var), nrow(youtube_var)/2)
youtube.test = youtube_var[-train,]
category.test = youtube_var$category_id[-train]
tree.youtube = tree(category_id ~ ., data = youtube_var, subset = train)
tree.pred = predict(tree.youtube, youtube.test, type = "class")
table(tree.pred, category.test)

#Cost-Complexity Pruning
RNGkind(sample.kind = "Rounding")
set.seed(3)
cv.youtube = cv.tree(tree.youtube, FUN = prune.misclass)
names(cv.youtube)
cv.youtube
par(mfrow=c(1,2))
plot(cv.youtube$)

#Random Forest
rf.youtube <- randomForest(category_id ~., data = trainset, importance = TRUE)
rf.youtube
