#Library
library(dplyr)
library(tree)
library(randomForest)

#Read csv file and set youtube data
youtube <- read.csv(file.choose(), header = TRUE)

#Randomly select a subset of data
youtube_sample <- youtube[sample(1:nrow(youtube), 500, replace = FALSE),]

str(youtube_sample)

#Predata Processing. Convert category_id to factor
youtube_sample$category_id <- as.factor(youtube_sample$category_id)

#Get variables likes, dislikes, count, category, and views
youtube_var <- youtube_sample[, c(5, 8, 9, 10, 11)]

########################
#Classification Tree
tree.youtube = tree(category_id ~ ., data = youtube_var)
summary(tree.youtube)
plot(tree.youtube)
text(tree.youtube, pretty = 0)

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
plot(cv.youtube$size, cv.youtube$dev, type = "b")
plot(cv.youtube$k, cv.youtube$dev, type = "b")
par(mfrow=c(1,1))
prune.youtube = prune.misclass(tree.youtube, best = 8)
plot(prune.youtube)
text(prune.youtube, pretty = 0)

tree.pred = predict(prune.youtube, youtube.test, type = "class")
table(tree.pred, category.test)

#Random Forest
RNGkind(sample.kind = "Rounding")
set.seed(1)
rf.youtube = randomForest(category_id ~., data = youtube_var, subset = train, importance = TRUE)

importance(rf.youtube)
varImpPlot(rf.youtube)

#Test Random Forest
for(i in 1:10){
  train <- sample(nrow(youtube_var), 0.8*nrow(youtube_var), replace = FALSE)
  trainset <- youtube_var[train,]
  validset <- youtube_var[-train,]
  #summary(trainset)
  #summary(validset)
  
  model1 <- randomForest(category_id ~ ., data = trainset, importance = TRUE)
  
  #model2 <- randomForest(category_id ~ ., data = trainset, ntree = 500, mtry = 3, importance = TRUE)
  
  predTrain <- predict(model1, trainset, type = "class")
  #table(predTrain, trainset$category_id)
  
  predValid <- predict(model1, validset, type = "class")
  print(mean(predValid == validset$category_id))
  #table(predValid, validset$category_id)
}
  print(test_error)
  importance(model2)
  varImpPlot(model2)
  



