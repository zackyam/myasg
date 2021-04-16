import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.decomposition import PCA
import imageio
from sklearn.preprocessing import StandardScaler

def load_data(digits, num):
    totalsize = 0
    for digit in digits:
        totalsize += min([len(next(os.walk('train%d' % digit))[2]), num])
    print('We will load %d images' % totalsize)
    X = np.zeros((totalsize, 784), dtype = np.uint8)   #784=28*28
    for index in range(0, len(digits)):
        digit = digits[index]
        print('\nReading images of digit %d' % digit)
        for i in range(num):
            pth = os.path.join('train%d' % digit,'%05d.pgm' % i)
            image = imageio.imread(pth).reshape((1, 784))
            X[i + index * num, :] = image
        print('\n')
    return X

def plot_mean_image(X, digits = [0]):
    ''' example on presenting vector as an image
    '''
    plt.close('all')
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.imshow(np.reshape(meanrow,(28,28)))
    plt.title('Mean image of digit ' + str(digits))
    plt.gray(), plt.xticks(()), plt.yticks(()), plt.show()
    
def main():
    digits = [0, 1, 2]
    # load handwritten images of digit 0, 1, 2 into a matrix X
    # for each digit, we just use 300 images
    # each row of matrix X represents an image
    X = load_data(digits, 300)
    # plot the mean image of these images!
    # you will learn how to represent a row vector as an image in this function
    plot_mean_image(X, digits)

    ####################################################################
    # you need to
    #   1. do the PCA on matrix X;
    #
    #   2. plot the eigenimages (reshape the vector to 28*28 matrix then use
    #   the function ``imshow'' in pyplot), save the images of eigenvectors
    #   which correspond to largest 9 eigenvalues. Save them in a single file
    #   ``eigenimages.jpg''.
    #
    #   3. plot the POV (the Portion of variance explained v.s. the number of
    #   components we retain), save the figure in file ``pov.jpg''
    #
    #   4. report how many dimensions are need to preserve 0.9 POV, describe
    #   your answers and your undestanding of the results in the plain text
    #   file ``description2.txt''
    #
    #   5. remember to submit file ``eigenimages.jpg'', ``pov.jpg'',
    #   ``description2.txt'' and ``ex2.py''.
    #
    # YOUR CODE HERE!
    ####################################################################
    X_std = StandardScaler().fit_transform(X)
    Xcov = np.cov(X_std.T)
    d, V = np.linalg.eig(Xcov)
    idx = d.argsort()[::-1] 
    d = d[idx] 
    V = V[:,idx]
    Vreal = V.real
    fig, axes = plt.subplots(3,3,figsize=(10,10))
            
    for ax, data in zip(axes.ravel(), Vreal[:9]):
        Vec = np.reshape(data,(28,28))
        ax.imshow(Vec)
    plt.savefig("eigenimages.jpg")
 
    plot2 = plt.figure(2)
    dim = np.arange(1, len(d)+1)
    pov_array=[]
    sum_d = d.sum()
    new_d = [sum(d[:i+1]) for i in range(len(d))]
    for i in new_d: 
        pov = i/sum_d
        pov_array.append(pov)
    plt.plot(dim, pov_array, linewidth=1, markersize=12, markeredgewidth=4, markeredgecolor='navy')
    plt.title('PoV v.s. eigenvalues ')
    plt.legend(['PoV'])
    plt.xlabel('order of eigenvalues'); plt.ylabel('PoV')
    plt.savefig("pov.jpg")
    plt.show()
    
    for i in pov_array:
        if i > 0.9:
            print(np.where(pov_array==i))
            break
    
if __name__ == '__main__':
    main()

