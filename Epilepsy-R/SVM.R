
# svm

# Fitting Kernel SVM to the Training set


SVM<- function(file_name){
  
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
  
  classifier_svm <- svm(formula = fmla,
                        data = train_set1,
                        type = "C-classification",
                        kernel = "radial")
  
  # Predicting the Test set results
  y_pred_svm <- predict(classifier_svm, newdata = test_set1[-3])
  
  # Making the Confusion Matrix ,accuracy ,recall , precision and roc
  cm_svm <- table(test_set1[, 3], y_pred_svm)
  print("svm  cm : ")
  print(cm_svm)
  accuracy_svm <- sum(diag(cm_svm )) / sum(cm_svm )
  print(paste0("svm accuracy : ", accuracy_svm))
  recall.svm <- cm_svm [2, 2] / (cm_svm [2, 1] + cm_svm [2, 2]) 
  print(paste0("svm recall : ", recall.svm))
  precision.svm <- cm_svm [2, 2] / (cm_svm [1, 2] + cm_svm [2, 2])   
  print(paste0("svm precision : ", precision.svm))
  roc_svm <- roc(test_set1$seizure, order( y_pred_svm))
  print("svm roc : ")
  print(roc_svm)
  
  
  
  # ROCRpred<-prediction(pred,obs)
  # plot(performance(ROCRpred,"tpr","fpr"))
  
  
  # # Visualising the Training set results
  # library(ElemStatLearn)
  set <- train_set1
  X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
  X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  grid_set <- expand.grid(X1, X2)
  colnames(grid_set) <- c("mean", "std")
  y_grid <- predict(classifier_svm, newdata = grid_set)
  plot(set[, -3],
       main = "Kernel SVM (Training set)",
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
  y_grid <- predict(classifier_svm, newdata = grid_set)
  plot(set[, -3], main = "Kernel SVM (Test set)",
       xlab = "mean", ylab = "std",
       xlim = range(X1), ylim = range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  points(grid_set, pch = ".", col = ifelse(y_grid == 1, "springgreen3", "tomato"))
  points(set, pch = 21, bg = ifelse(set[, 3] == 1, "green4", "red3"))
  
  return(file_name)
}

SVM("D:/data/output_1.fst")