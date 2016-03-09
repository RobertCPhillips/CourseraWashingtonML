require("caTools")
require("ggplot2")
require(rpart)
require(rpart.plot)
require(randomForest)

#VARIABLE DESCRIPTIONS
#survival        Survival
#                (0 = No; 1 = Yes)
#pclass          Passenger Class (proxy for socio-economic status)
#                (1 = 1st; 2 = 2nd; 3 = 3rd)
#name            Name
#sex             Sex
#age             Age
#sibsp           Number of Siblings/Spouses Aboard
#parch           Number of Parents/Children Aboard
#ticket          Ticket Number
#fare            Passenger Fare
#cabin           Cabin
#embarked        Port of Embarkation
#                (C = Cherbourg; Q = Queenstown; S = Southampton)

train <- read.csv("titanic_train.csv")
test <- read.csv("titanic_test.csv")

str(train)
str(test)

train$Pclass <- as.factor(train$Pclass)
train$Survived <- as.factor(train$Survived)
test$Pclass <- as.factor(test$Pclass)

summary(train)

set.seed(1446)
split <- sample.split(train$Survived, SplitRatio = 0.65)
train.tr <- subset(train, split == TRUE)
train.te <- subset(train, split == FALSE)


#---------------------------------
# naive model - assume largest proportion
#---------------------------------
naive <- table(train.tr$Survived)
naive #most do not survive

naive.pred <- train.te$Survived == 0
naive.acc <- sum(naive.pred) / length (naive.pred)

#---------------------------------
#model 1 - gender
#---------------------------------
model1 <- Survived ~ Sex

# cart model
cart1 <- rpart(model1, data=train.tr, method="class")
prp(cart1)

cart1.predict <- predict(cart1, newdata=train.te, type="class")
cart1.predict.t <- table(train.te$Survived, cart1.predict)
cart1.acc <- (cart1.predict.t[1,1]+cart1.predict.t[2,2])/sum(cart1.predict.t)

# random forrest
set.seed(1000)
forrest1 <- randomForest(model1, data=train.tr)

forrest1.predict <- predict(forrest1, newdata=train.te, type="class")
forrest1.predict.t <- table(train.te$Survived, forrest1.predict)
forrest1.acc <- (forrest1.predict.t[1,1]+forrest1.predict.t[2,2])/sum(forrest1.predict.t)

#---------------------------------
#model 2 - gender and class
#---------------------------------
model2 <- Survived ~ Sex + Pclass

# cart model
cart2 <- rpart(model2, data=train.tr, method="class")
prp(cart2)

cart2.predict <- predict(cart2, newdata=train.te, type="class")
cart2.predict.t <- table(train.te$Survived, cart2.predict)
cart2.acc <- (cart2.predict.t[1,1]+cart2.predict.t[2,2])/sum(cart2.predict.t)

# random forrest
set.seed(1020)
forrest2 <- randomForest(model2, data=train.tr)

forrest2.predict <- predict(forrest2, newdata=train.te, type="class")
forrest2.predict.t <- table(train.te$Survived, forrest2.predict)
forrest2.acc <- (forrest2.predict.t[1,1]+forrest2.predict.t[2,2])/sum(forrest2.predict.t)

#---------------------------------
#model 3 - gender and siblings
#---------------------------------
model3 <- Survived ~ Sex + Age + Pclass + Parch + SibSp

train.tr.imp <- rfImpute(model3, train.tr)
train.te.imp <- rfImpute(model3, train.te)

# cart model
cart3 <- rpart(model3, data=train.tr.imp, method="class")
prp(cart3)

cart3.predict <- predict(cart3, newdata=train.te.imp, type="class")
cart3.predict.t <- table(train.te.imp$Survived, cart3.predict)
cart3.acc <- (cart3.predict.t[1,1]+cart3.predict.t[2,2])/sum(cart3.predict.t)

# random forrest
set.seed(2020)
forrest3 <- randomForest(model3, data=train.tr.imp)

forrest3.predict <- predict(forrest3, newdata=train.te.imp, type="class")
forrest3.predict.t <- table(train.te.imp$Survived, forrest3.predict)
forrest3.acc <- (forrest3.predict.t[1,1]+forrest3.predict.t[2,2])/sum(forrest3.predict.t)

#---------------------------------
#test
#---------------------------------
forrest3.predict2 <- predict(forrest3, newdata=test, type="class")
test$Survived <- forrest3.predict2
