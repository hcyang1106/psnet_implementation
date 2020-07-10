import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def draw_curve(lines,model,target): # draw loss curves


    plt.figure()

    for e in range(len(lines)):

        plt.plot(lines[e]['epoch'], lines[e][target], label = lines[e]['name'])
    
    epoch = []
    line = []
    for e in range(1,len(lines[0]['epoch'])+1):
        epoch.append(e)
        line.append(0.82)

    plt.plot(epoch, line, label = '0.82')
    plt.xlabel('epoch')
    plt.ylabel(target)
    
    plt.title(target)
    plt.legend()
    plt.savefig('./figure/'+ str(model) +"_" +str(target) + ".png")
    plt.show()
    #plt.close()

def plot_confusion_matrix(y_true, y_pred, labels=None,
                          sample_weight=None, normalize=None,
                          title = None,cmap='viridis',
                          epoch = None):


    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels, normalize=normalize)

    plt.matshow(cm, cmap=cmap) # imshow
    plt.colorbar()
    l = cm.shape[0]
    tick_marks = np.arange(cm.shape[0])
    #plt.xticks(tick_marks, df_confusion.shape[0], rotation=45)
    #plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    for i in range(l):
        for j in range(l):
            plt.text(j, i, cm[i][j],
                     horizontalalignment="center",
                     color="black" )#if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion matrix'+ ".jpg")
    plt.close()
    #plt.show()

def show_point_cloud(point_cloud, filename): 
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    for e in point_cloud:
        xdata = e[0]
        ydata = e[1]
        zdata = e[2]
        ax.scatter3D(xdata, ydata, zdata, s=2, c=(e[3:]/256).reshape(1, 3));#, c=zdata

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    for e in point_cloud:
        xdata = e[0]
        ydata = e[2]
        zdata = e[1]
        ax.scatter3D(xdata, ydata, zdata, s=2, c=(e[3:]/256).reshape(1, 3));#, c=zdata

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    for e in point_cloud:
        xdata = e[1]
        ydata = e[2]
        zdata = e[0]
        ax.scatter3D(xdata, ydata, zdata, s=2, c=(e[3:]/256).reshape(1, 3));#, c=zdata

    fig.savefig(filename)
    plt.close(fig)   