import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt


class UnionFindOpt:
    """
    Optimized version of union find that
    implements path compression and
    weight-based merging
    """
    
    def __init__(self,N):
        self.N = N
        self.parents = []
        self.weights = []
        self._operations = 0
        self._calls = 0
        
        for i in range(N):
            self.parents.append(i)
            self.weights.append(1)
    
    def root(self,i):
        self._calls += 1
        if i != self.parents[i]:
            self._operations += 1
            self.parents[i] = self.root(self.parents[i])    
        
        return self.parents[i]
    
    def find(self,i,j):
        self._calls +=1
        return self.root(i) == self.root(j)
    
    
        
    
    def union(self,i,j):
        self._calls +=1
        root_i = self.root(i)
        root_j = self.root(j)
        
        if self.weights[root_i] < self.weights[root_j]:
            self._operations += 1
            self.parents[root_i] = self.parents[root_j]
            self.weights[root_j] += self.weights[root_i]
        else:
            self._operations +=1
            self.parents[root_j] = self.parents[root_i]
            self.weights[root_i] += self.weights[root_j]
            
   

def load_cells_grayscale(filename, n_pixels = 0):
    """
    Load in a grayscale image of the cells, where 1 is maximum brightness
    and 0 is minimum brightness

    Parameters
    ----------
    filename: string
        Path to image holding the cells
    n_pixels: int
        Number of pixels in the image
    
    Returns
    -------
    ndarray(N, N)
        A square grayscale image
    """
    I = plt.imread(filename)
    cells_gray = 0.2125*I[:, :, 0] + 0.7154*I[:, :, 1] + 0.0721*I[:, :, 2]
    # Denoise a bit with a uniform filter
    cells_gray = ndimage.uniform_filter(cells_gray, size=10)
    cells_gray = cells_gray - np.min(cells_gray)
    cells_gray = cells_gray/np.max(cells_gray)
    N = int(np.sqrt(n_pixels))
    if n_pixels > 0:
        # Resize to a square image
        cells_gray = misc.imresize(cells_gray, (N, N))
    return cells_gray


def permute_labels(labels):
    """
    Shuffle around labels by raising them to a prime and
    modding by a large-ish prime, so that cells are easier
    to see against their backround
    Parameters
    ----------
    labels: ndarray(M, N)
        An array of labels for the pixels in the image
    Returns
    -------
    labels_shuffled: ndarray(M, N)
        A new image where the labels are different but still
        the same within connected components
    """
    return (labels**31) % 833


def get_cluster_centers(labels):
    """
    Parameters
    ----------
    labels : list
        table that holds the unionfind
        roots of each pixel

    Returns
    -------
    list_matches : list
        list that holds the average
        pixels of each cluster

    """
    #initialize an empty list for each pixel in image
    list_labels = [[] for _ in range((len(labels))**2)]
    #loop through pixels in image
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            
            #add coordinates of pixel to the appropriate indexed list
            #root of pixels will be index in list_labels table and convert back to pixel rather than root
            
            list_labels[int(labels[i][j])].append([i, j])
            
    #get the average of each cell (each group of pixels with same root)
    list_matches = []
    for i in range(len(list_labels)):
        if len(list_labels[i]) > 1:
            
            total_x = 0
            total_y = 0
            counter = 0
            
            for j in range(len(list_labels[i])):
                total_x += list_labels[i][j][0]
                total_y += list_labels[i][j][1]
                counter += 1
                
            x = total_x//counter
            y = total_y//counter
            avg_pixel = (x, y)
            list_matches.append(avg_pixel)
            
    return list_matches
    

def get_cell_labels(image,thresh):
    """
    Cluster grayscale pixels together by 
    unioning adjacent pixels that meet the brightness threshold
    Parameters
    ----------
    image : list
        table of pixel grayscale values
    thresh : float
        minimum brightness that we're looking for

    Returns
    -------
    labels : list
        table that holds the unionfind 
        roots of each pixel

    """
    
    #initialize list of pixels and create 2d array of labels based on image size
    #create union find object with number of pixels as size
    matches = UnionFindOpt(len(image)**2)
    labels = np.zeros((len(image), len(image)))
    
    for i in range(len(image) - 1):
        for j in range(len(image) - 1):
            #if a pixel and its adjacent pixel are bright enough, union them together and add their root to the labels list
            if image[i][j] > thresh and image[i + 1][j] > thresh and i < len(image) - 1:
                
                #len(image) = 400, so i * 400 translates proper row of 2d array to 1d for union find
                #adding j is the column offset
                
                matches.union(i*len(image)+j, (i+1)*len(image)+j)
            
            #if a pixel and the other adjacent pixel are bright enough, union them together
            if image[i][j] > thresh and image[i][j + 1] > thresh and j < len(image) - 1:
                matches.union(i*len(image)+j, i*len(image)+j+1)
    
    for i in range(len(image)):
        for j in range(len(image)):
            #add the root of each pixel to the labels list to create 'clusters'
            labels[i][j] = matches.root(j + len(image) * i)
                
    return labels
    


if __name__ == '__main__':
    
    thresh = 0.7
    I = load_cells_grayscale("Cells.jpg")
    labels = get_cell_labels(I, thresh)
    #plt.imshow(permute_labels(labels))     #Uncomment this to show the labeled image
    #plt.show()
    
    get_cluster_centers(labels)
    cells_original = plt.imread("Cells.jpg")
    X = get_cluster_centers(labels)
    X = np.array(X)
    plt.imshow(cells_original)
    plt.scatter(X[:, 1], X[:, 0], c='C4')
    plt.show()