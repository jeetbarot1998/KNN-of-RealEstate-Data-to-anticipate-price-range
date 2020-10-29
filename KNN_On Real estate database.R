real_es <- read.csv("realEstate.csv", stringsAsFactors = TRUE)


# Define input variables
X = real_es[,2:9]
# Define target variable
y = real_es[,10]

set.seed(110) 

# Normalize the inputs 
norm.values <- preProcess(X, method=c("center", "scale")) 
X.norm <- predict(norm.values, X)  
head(X.norm)

# split The data 
sample = sample.split(real_es, SplitRatio = 0.80)
X_train = subset(X.norm, sample==TRUE) 
X_test = subset(X.norm, sample==FALSE) 
dim(X_train)
dim(X_test)

y_train = subset(y,sample==TRUE) 
y_test = subset(y, sample==FALSE)
 
nn_model <- knn(train = X_train[,1:3], test=X_test[,1:3], cl = y_train, k=5)
summary(nn_model)

confusionMatrix(nn_model, y_test)$overall[1] # Accuracy = 0.67
