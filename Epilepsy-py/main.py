import pre_process
import logistic_reg
import svm
import random_forest
output_file = "D:/data/output_1.csv"
pre_process.pre_processing("D:/data/Epi.pkl")


# Modoule selection

choice = input("choose your modoule : 1.Logistic Regression  2.SVM  3.Random Forest :")
if choice == '1':
    logistic_reg.logistic_reg("D:/data/output_3.pkl")
elif choice == '2':
    svm.svm("D:/data/output_3.pkl")
elif choice == '3':
    random_forest.random_forest("D:/data/output_3.pkl")
