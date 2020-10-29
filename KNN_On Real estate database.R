install.packages("pacman")
library(pacman)
p_load("rsample","dplyr", "caTools","caret", "e1071", "FNN") # to read dataset on Attrition

# Read the data file
real_es <- read.csv("realEstate.csv", stringsAsFactors = TRUE)


dim(real_es)
# Define input variables
X = real_es[,2:9]
# Define target variable
y = real_es[,10]

set.seed(101) 

# Normalize the inputs 
norm.values <- preProcess(X, method=c("center", "scale")) 
X.norm <- predict(norm.values, X) # Normalized input 
head(X.norm)

# train test split 
sample = sample.split(real_es, SplitRatio = 0.80)# select a random sample of 80%
X_train = subset(X.norm, sample==TRUE) # input for training
X_test = subset(X.norm, sample==FALSE) # input for prediction accuracy
dim(X_train)
dim(X_test)

y_train = subset(y,sample==TRUE) # labels for training
y_test = subset(y, sample==FALSE)# labels for prediction accuracy


# Use first three inputs and run a KNN for k =5 
nn_model <- knn(train = X_train[,1:3], test=X_test[,1:3], cl = y_train, k=5)
summary(nn_model)
confusionMatrix(nn_model, y_test)$overall[1]# accuracy of prediction on test data 
# Accuracy 
# 0.67

# How do we know K =5 is the best?
# To select K, we will check accuracy as various values of K between 1 and 20. 

# define a dataframe in which we will save accuracy for different values of K

accuracy.df <- data.frame(k = seq(1, 20, 1), accuracy = rep(0, 20))
accuracy.df # right now we have filled accurcay to be 0 for all values of K

# compute knn for different k on validation by looping 
for(i in 1:20) { # we will loop through K= 1 to 20
  knn_model <- knn(train = X_train[1:3], test=X_test[1:3], cl = y_train, k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn_model, y_test)$overall[1]
}

plot(accuracy.df) # plot accuracy for different values of K 
lines(accuracy.df)

which.max(accuracy.df$accuracy) # optimal K
# K = 16
# Evaluate model at K =16
accuracy.df[16,] # read 16th row
# accuracy  0.69


# Model improvement, add more inputs. We will use all inputs. 

# compute knn for different k on validation for all input
for(i in 1:20) {
  knn_model <- knn(train = X_train, test=X_test, cl = y_train, k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn_model, y_test)$overall[1]
}

plot(accuracy.df)
lines(accuracy.df)

which.max(accuracy.df$accuracy) # optimal K
# K = 15
# Evaluate model at K =15
accuracy.df[15,] # read 15th row
# accuracy 0.74


########################################################################################



