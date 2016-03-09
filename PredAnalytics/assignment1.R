install.packages("caret")
install.packages("rpart")
install.packages("tree")
install.packages("randomForest")
install.packages("e1071")
install.packages("ggplot2")
install.packages("caTools")

require("caret")
require("rpart")
require("tree")
require("randomForest")
require("e1071")
require("ggplot2")
require("caTools")

#file_id: The data arrives in files, where each file represents a three-minute window; this field represents which file the data came from. The number is ordered by time, but is otherwise not significant.
#time: This is an integer representing the time the particle passed through the instrument. Many particles may arrive at the same time; time is not a key for this relation.
#cell_id: A unique identifier for each cell WITHIN a file. (file_id, cell_id) is a key for this relation.
#d1, d2: Intensity of light at the two main sensors, oriented perpendicularly. These sensors are primarily used to determine whether the particles are properly centered in the stream. Used primarily in preprocesssing; they are unlikely to be useful for classification.
#fsc_small, fsc_perp, fsc_big: Forward scatter small, perpendicular, and big. These values help distingish different sizes of particles.
#pe: A measurement of phycoerythrin fluorescence, which is related to the wavelength associated with an orange color in microorganisms
#chl_small, chl_big: Measurements related to the wavelength of light corresponding to chlorophyll.
#pop: This is the class label assigned by the clustering mechanism used in the production system.

seaflow <- read.csv('seaflow_21min.csv')
summary(seaflow)


set.seed(144)
split <- sample.split(seaflow$pop, SplitRatio = 0.5)
train <- subset(seaflow, split == TRUE)
test <- subset(seaflow, split == FALSE)

#q4
mean(train$time)

#q5
ggplot(seaflow, aes(x=pe, y=chl_small, color=pop)) + geom_point(shape=1)

#q6,7,8
fol <- pop ~ fsc_small + fsc_perp + fsc_big + pe + chl_big + chl_small
model <- rpart(fol, method="class", data=train)

print(model)

#q9
model.pred <- predict(model, test, type="class")
model.pred.acc <- sum(model.pred == test$pop) / length(test$pop)
model.pred.t <- table(test$pop,model.pred)
model.pred.t <- table(pred = model.pred, true = test$pop)

#q10
model2 <- randomForest(fol, data=train) 
model2.pred <- predict(model2, test, type="class")
model2.pred.acc <- sum(model2.pred == test$pop) / length(test$pop)
model2.pred.t <- table(pred = model2.pred, true = test$pop)

#q11
importance(model2)

#q12
model3 <- svm(fol, data=train)
model3.pred <- predict(model3, test, type="class")
model3.pred.acc <- sum(model3.pred == test$pop) / length(test$pop)
model3.pred.t <- table(pred = model3.pred, true = test$pop)

#q14
seaflow2 <- subset(seaflow, file_id != 208)
set.seed(1440)
split2 <- sample.split(seaflow2$pop, SplitRatio = 0.5)
train2 <- subset(seaflow2, split2 == TRUE)
test2 <- subset(seaflow2, split2 == FALSE)

model3b <- svm(fol, data=train2)
model3b.pred <- predict(model3b, test2, type="class")
model3b.pred.acc <- sum(model3b.pred == test2$pop) / length(test2$pop)
model3b.pred.t <- table(pred = model3b.pred, true = test2$pop)
model3b.pred.acc - model3.pred.acc

#q15
plot(seaflow$fsc_big)
