rm(list = ls())
require(graphics)
require(ggplot2)
require(reshape2)
require(ggplot2)
require(caret)
require(e1071) 
require(class)  
require(e1071)
require(fst)
require (matrixStats)
require (pROC)
library(reshape2)
require(dplyr)
library(plotROC)
library(readr)
library(forecast)
library(tibble)
library(lintr)
require(randomForest)
library(ElemStatLearn)
# The USI dataset is made up of 500 subjects each with 23 seconds divided to 178 of 1/178 seconds  of EEG values labled 
# reading csv file of Epilepsy of USI , taking all the 1- result and equaliy taking 0 result for balanced data for the machine learning modouls later


# orgenized_origin_df <- read.csv("D:/data/epilepcia1.csv")
# # Store the data frame to disk
# write.fst(orgenized_origin_df, "D:/data/epilepcia1.fst")
orgenized_origin_df <- read.fst("D:/data/epilepcia1.fst") 


# best_sample_df <- read.csv("D:/data/Epii.csv")
# # Store the data frame to disk
# write.fst(best_sample_df,"D:/data/Epii.fst")
# Retrieve the data frame again
best_sample_df <- read.fst("D:/data/Epii.fst")



# looping the subjects (part of 500 subjects , each with 178 of 1/178 seconds for 23 seconds of EEG test )
pre_processing <- function(best_sample_df){
for(i in 1:1000) {
  dfi<-filter(best_sample_df,subject==i)
  dfi<-dfi[order(dfi$id),]
  dfi_new<-select(dfi,-y,-subject,-isseasonal,-seizure)
  if (nrow(dfi)>0){       #in order to ignor empty dataframes
#     
#     
#     #melting
#     # to reshape the data 
    mmi = melt(dfi_new, id.vars=1)
    mmi<-mmi[order(as.numeric(mmi$id)),]
    mmi$variable <- gsub("X", "0", mmi$variable)
    mmi$variable<- as.numeric(mmi$variable)

# 
#     # difining as time series
    EEGi<-mmi$value
    dfi.ts<-ts(EEGi,start = c(1,0.0056179),frequency =178)
#     plot(dfi.ts)
    tsi <- ts(dfi.ts, frequency=178)

#     
#     # locating seasonality
#     fit <- tbats(tsi)
#     seasonal <- !is.null(fit$seasonal)
#     
#     # creating new dataframe by looping  of mean and sd of every second of every subject each second includes 178 of 1/178 seconds
    datai<-as.data.frame(tapply(tsi, rep(1:(length(tsi)/178), each = 178), mean))
    datai$std<-as.vector(tapply(tsi, rep(1:(length(tsi)/178), each = 178), sd))

# 
#     # including the target 
    if (any(dfi$y==1)){

      datai$seizure=1;
    }
    else
      datai$seizure=0;
#   # including the subject
    datai$subject=i
#  
#    
    colnames(datai) <- c("mean","std","seizure","subject")

#     plot(tsi,ylab="EEGi")

#     zoom in 1 second
#     plot(tsi,ylab="EEGi",xlim=c(1,2))

    filename="D:/data/output_1.csv"

   # creating csv file for the process of machine learning , including only mean annd std of each second
    write.table(datai, filename,  append=T, sep=",", row.names=F, col.names=!file.exists(filename))
  }
 
}
  return(filename)
}
# pre_processing(best_sample_df)


my.choice<- function(choice){
  choice <- readline("choose your modoule 1. Logistic Regression 2. Random Forest 3. SVM =" )
  
  if (choice == 1){
    source("Epilepsy-R/Logistic_regression.R")
  } else if (choice== 2){
    source("Epilepsy-R/Random_Forest.R")
  } else if (choice== 3){
    source("Epilepsy-R/SVM.R")
  }
  return(choice) 
}
my.choice(choice)
