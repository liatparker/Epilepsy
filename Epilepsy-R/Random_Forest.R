# Random Forest 

Random_Forest  <- function(file_name){
  
  df1 <- read.fst(file_name)
  
  # Encoding the target feature as factor
  df1$seizure <- factor(df1$seizure, levels = c(0, 1))
  
  # scaling df
  df1[, -c(3)] <- scale(df1[, -c(3)])
  
  
  # # Splitting the dataset into the Training set and Test set
  train_indx1 <- sample (1:nrow( df1), floor((0.75) * nrow ( df1)))
  train_set1 <- df1[train_indx1, ]
  test_set1 <- df1[-train_indx1, ]
  
  
  
  fmla <- as.formula("seizure ~.")
  
  
  rf_model <- randomForest(fmla, train_set1)
  rf_model$importance 
  
  # Predicting the Test set results
  rf_predict <- predict(rf_model, test_set1)
  
  
  # Making the Confusion Matrix ,accuracy ,recall , precision and roc
  confusion_matrix_rf <- table(test_set1[, 3], rf_predict)
  print("Random Forest cm : " )
  print(confusion_matrix_rf)      
  accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
  print(paste0("Random Forest accuracy : ", accuracy_rf))
  recall.rf <- confusion_matrix_rf[2, 2] / (confusion_matrix_rf[2, 1] + confusion_matrix_rf[2, 2]) 
  print(paste0("Random Forest recall : ", recall.rf))
  precision.rf <- confusion_matrix_rf[2, 2] / (confusion_matrix_rf[1, 2] + confusion_matrix_rf[2, 2])
  print(paste0("Random Forest precision : ", precision.rf))
  roc.rf <- roc(test_set1$seizure, order( rf_predict))
  print("Random Forest roc : ")
  print(roc.rf)
  
  
  # visualization
  # 
  library(ElemStatLearn)
  # 
  set <- train_set1
  X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
  X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  grid_set <- expand.grid(X1, X2)
  colnames(grid_set) <- c("mean", "std")
  y_grid <- predict(rf_model, grid_set)
  plot(set[, -3], main = "Random Forest Classification (Training test )",
       xlab = "mean", ylab = "std",
       xlim = range(X1), ylim = range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  points(grid_set, pch = ".", col = ifelse(y_grid == 1, "springgreen3", "tomato"))
  points(set, pch = 21, bg = ifelse(set[, 3] == 1, "green4", "red3"))
  
  
  set <- test_set1
  X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
  X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  grid_set <- expand.grid(X1, X2)
  colnames(grid_set) <- c("mean", "std")
  y_grid <- predict(rf_model, grid_set)
  plot(set[, -3], main = "Random Forest Classification (Test set)",
       xlab = "mean", ylab = "std",
       xlim = range(X1), ylim = range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  points(grid_set, pch = ".", col = ifelse(y_grid == 1, "springgreen3", "tomato"))
  points(set, pch = 21, bg = ifelse(set[, 3] == 1, "green4", "red3"))
  
  
  
  return(file_name)
}

Random_Forest("D:/data/output_1.fst")