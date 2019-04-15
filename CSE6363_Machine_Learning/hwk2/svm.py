from sklearn import svm
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

def plot_hyperplane(clf, X, y, 
                    h=0.01, 
                    draw_sv=True, 
                    title='hyperplane'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # hyperplane
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='Accent', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    # draw support vector
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')

        
X = np.array([[1, 2],[2, 3],[2, 1],[3, 4],[1, 3],[4, 4]])
y = [-1, 1, -1, 1, -1, 1]
clf = svm.SVC(C=10, kernel='linear')
clf.fit(X, y)

plt.figure(figsize=(12, 4), dpi=80)

plot_hyperplane(clf, X, y, h=0.01, 
                title='Maximum Margin Hyperplane')
