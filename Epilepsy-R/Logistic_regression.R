
Logistic_regression <- function(file_name){
  
  sum(is.na(file_name))
  # df1<-read.csv("D:/data/output_1.csv")
  # df1<-select(df1,-subject)
  # Store the data frame to disk
  # write.fst(df1, "D:/data/output_1.fst")
  
  # Retrieve the data frame again
  df1 <- read.fst(file_name)
  
  
  # Encoding the target feature as factor
  df1$seizure <- factor(df1$seizure, levels = c(0, 1))
  
  
# logistic regression

# scaling df1

  df1[, -c(3)] <- scale(df1[, -c(3)])


# # Splitting the dataset into the Training set and Test set
 train_indx1 <- sample (1:nrow( df1), floor((0.75) * nrow ( df1)))
 train_set1 <- df1[train_indx1, ]
 test_set1 <- df1[-train_indx1, ]



 fmla <- as.formula("seizure ~.")



 classifier_lr <- glm(formula = fmla,
                     family = binomial,
                     data = train_set1)



# Predicting the Test set results

 prob_pred_lr <-  predict(classifier_lr, type = "response", newdata = test_set1[-3])
 y_pred_lr <- ifelse(prob_pred_lr > 0.5, 1, 0)



# Making the Confusion Matrix ,accuracy ,recall , precision and roc

 cm_lr <- table(test_set1[, 3], y_pred_lr > 0.5)
 print("Logistic Regression cm : ")
 print(cm_lr)
 accuracy_lr1 <- sum(diag(cm_lr)) / sum(cm_lr)
 print(paste0("Logistic Regression accuracy : ", accuracy_lr1))
 recall_lr <- cm_lr[2, 2] / (cm_lr[2, 1] + cm_lr[2, 2])
 print(paste0("Logistic Regression recall : ", recall_lr))
 precision_lr <- cm_lr[2, 2] / (cm_lr[1, 2] + cm_lr[2, 2])
 print(paste0("Logistic Regression precision : ", precision_lr))
 roc_lr <- roc(test_set1$seizure, order( y_pred_lr > 0.5))
 print("Logistic Regression roc : ")
 print(roc_lr)

# Visualising the Training set results
 library(ElemStatLearn)
 set <- train_set1
 X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
 X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
 grid_set <- expand.grid(X1, X2)
 colnames(grid_set) <- c("mean", "std")
 prob_set <- predict(classifier_lr, type = "response", newdata = grid_set)
 y_grid <- ifelse(prob_set > 0.5, 1, 0)
 plot(set[, -3],
     main = "Logistic Regression (Training set)",
     xlab = "mean", ylab = "std",
     xlim = range(X1), ylim = range(X2))
 contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
 points(grid_set, pch = ".", col = ifelse(y_grid == 1, "springgreen3", "tomato"))
 points(set, pch = 21, bg = ifelse(set[, 3] == 1, "green4", "red3"))

# Visualising the Test set results
 library(ElemStatLearn)
 set <- test_set1
 X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
 X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
 grid_set <- expand.grid(X1, X2)
 colnames(grid_set) <- c("mean", "std")
 prob_set <- predict(classifier_lr, type = "response", newdata = grid_set)
 y_grid <- ifelse(prob_set > 0.5, 1, 0)
 plot(set[, -3],
     main = "Logistic Regression (Test set)",
     xlab = "mean", ylab = "std",
     xlim = range(X1), ylim = range(X2))
 contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
 points(grid_set, pch = ".", col = ifelse(y_grid == 1, "springgreen3", "tomato"))
 points(set, pch = 21, bg = ifelse(set[, 3] == 1, "green4", "red3"))
 
 return(file_name)
}

Logistic_regression("D:/data/output_1.fst")