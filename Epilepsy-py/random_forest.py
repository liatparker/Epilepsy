import pandas as pd
from matplotlib import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score


# Splitting the dataset into the Training set and Test set
def random_forest(filename):

# Importing the dataset
    dataset = pd.read_pickle(filename)
    # dataset.to_pickle("D:/data/output_1.pkl")
    # dataset = pd.read_pickle("D:/data/output_1.pkl")
    X = dataset.iloc[:, [0, 1]].values
    y = dataset.iloc[:, 2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

# Random Forest Classification


# Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

# Predicting the Test set results
    y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    print("Random forest cm:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
# get the accuracy , recall and precision :
    print("Random forest accuracy:")
    print(accuracy_score(y_test, y_pred))
    acc_train = accuracy_score(y_test, y_pred)
    print("Random forest recall:")
    print(recall_score(y_test, y_pred))
    print("Random forest precision:")
    print(precision_score(y_test, y_pred))
    auc_train = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print("auc Random Forest : ")
    print(roc_auc)


# skplt.metrics.plot_roc_curve(y_test, y_pred)
# plt.show()

# Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Random Forest Classification (Training set)')
    plt.xlabel('mean')
    plt.ylabel('std')
    plt.legend()
    plt.show()

# Visualising the Test set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c =ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Random Forest Classification (Test set)')
    plt.xlabel('mean')
    plt.ylabel('std')
    plt.legend()
    plt.show()

# ROC PLOT
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic- Random Forest')
    plt.legend(loc="lower right")
    plt.show()