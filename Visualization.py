import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement
from random import sample
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

def check_in_range(num):
    if num > 255:
        return 255
    if num < 0:
        return 0
    return num
def show_point_cloud_three_view(point_cloud, filename): 
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    for e in point_cloud:
        e[3] = check_in_range(e[3])
        e[4] = check_in_range(e[4])
        e[5] = check_in_range(e[5])
        xdata = e[0]
        ydata = e[1]
        zdata = e[2]
        ax.scatter3D(xdata, ydata, zdata, s=2, c=(e[3:]/255).reshape(1, 3));#, c=zdata

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    for e in point_cloud:
        e[3] = check_in_range(e[3])
        e[4] = check_in_range(e[4])
        e[5] = check_in_range(e[5])
        xdata = e[0]
        ydata = e[2]
        zdata = e[1]
        ax.scatter3D(xdata, ydata, zdata, s=2, c=(e[3:]/255).reshape(1, 3));#, c=zdata

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    for e in point_cloud:
        e[3] = check_in_range(e[3])
        e[4] = check_in_range(e[4])
        e[5] = check_in_range(e[5])
        xdata = e[1]
        ydata = e[2]
        zdata = e[0]
        ax.scatter3D(xdata, ydata, zdata, s=2, c=(e[3:]/255).reshape(1, 3));#, c=zdata

    fig.savefig(filename)
    plt.close(fig)   


def show_point_cloud(point_cloud):
    fig = plt.figure()
    ax = plt.axes(projection="3d")  
    for e in point_cloud:
        xdata = e[0]
        ydata = e[1]
        zdata = e[2]
        ax.scatter3D(xdata, ydata, zdata, s=2, c=(e[3:]/256).reshape(1, 3))
    ax.set_aspect(aspect = 'equal')
    plt.show()


if __name__ == '__main__':
    content_point_cloud_data = PlyData.read('1a1a5facf2b73e534e23e9314af9ae57.ply')
    content_point_cloud_temp = []
    for e in content_point_cloud_data.elements[0]:
        content_point_cloud_temp.append([e[i] for i in range(0, 6)])

    content_point_cloud_temp = sample(content_point_cloud_temp, 5000)
    content_point_cloud = np.asarray(content_point_cloud_temp)
    show_point_cloud(content_point_cloud)