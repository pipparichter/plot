import numpy as np
import scipy.sparse
import plot
import sklearn.manifold
import sklearn.decomposition
import matplotlib as plt

import popalign as PA


class SubpopPlot(plot.Plot):    
    '''
    This class creates handles analysis of cell-type subpopulations. It assumes the pop object passed in as an argument
    is already normalized. 
    '''
    def __init__(self, obj,
        is_subplot=False,
        filename=None,
        # samples=None,
        # celltype=None,
        **kwargs):
        '''
        This function intantiates an object belonging to the SubpopPlot class.

        Parameters
        ----------
        obj : dict
            A PopAlign oject.
        is_subplot : bool
            Whether or not the plot is a subplot.


        '''
        self.merge_sample = obj.merge_samples 

        # Parent class initialization --------------------------------------------------------
        super().__init__(obj, 
                is_subplot=is_subplot,
                color=None, 
                filename='subpop', 
                plotter=self._plotter)

        # if self.samples is None:
        self.samples = obj.samples # If no sample is specified, use all samples. 

        # assert(celltype is not None, "A celltype must be specified for this Plot type.")
        self.celltype = obj.celltype

        # Relevant values stored in the class -----------------------------------------------
        
        # Values to be populated by the __get_data function.
        self.log_cvs = None
        self.log_means = None
        offset = kwargs.get('offset', 1.5)    
        self.mtx = None
        self.sample_labels = None # Stores the names of the samples in the matrix (for graphing). 
        self.filtered_mtx = None
        self.filtered_genes = None

        self.__get_data(offset)

        # Values to be populated when dimensionality reduction is carried out.
        self.nfeats = kwargs.get("nfeats", 20)
        self.onmf_A = None
        self.onmf_B = None # Cells by features. 
        self.pca = None

        # Values to be populated when clustering is carried out.
        self.kmeans = None
        self.nclusters = None
        self.cluster_method = None
        self.mtx_by_cluster = None

    def __get_data(self, offset):
        '''
        Collect data for the specified celltype across all given samples. 
        '''
        # Build a matrix with each row representing a sample, and each column representing data for
        # a particular cell. 
        mtx = []
        sample_labels = []
        for sample in self.samples:
            celltype_idxs = np.where(self.pop['samples'][sample]['cell_type'] == self.celltype)
            # Get the data from the original M matrix, which is normalized but not filtered. 
            # The M matrix has a column corresponding to each cell, and a row for each gene.
            mtx.append(self.pop['samples'][sample]['M'][:, celltype_idxs])
            sample_labels.append([sample] * len(celltype_idxs))
        mtx = scipy.sparse.hstack(mtx) # This concatenates the columns of all columns stored in the mtx list.
        
        self.sample_labels = np.hstack(sample_labels)
        self.mtx = mtx
        self.filtered_mtx = self.__filter_data(mtx, offset)

    # NOTE: What is a Poisson approach to filtering?
    def __filter_data(self, mtx, offset):
        '''
        This function filters the genes using a Poisson filtering approach.

        Parameters 
        ----------
        offset : [WHAT DOES THIS DO?]
        '''
 
        # NOTE: What exactly does this function do?
        slope, intercept, r, p, stderr = PA.linregress(self.mean, self.cv)
        # Update the intercept. NOTE: Why do we do this?
        intercept += np.log10(offset)
        # xlims = [min(self.log_means), max(self.log_means)]
        # y = [slope * x + intercept for x in xlims]

        idxs = np.where(self.log_cvs > self.log_means*slope + intercept)
        filter_idxs = self.nonzero_idxs[idxs]
        # Store the list of filtered genes as an attribute. 
        self.filtered_genes = self.pop['genes'][filter_idxs]
        
        return mtx[filter_idxs, :] # Filter the matrix stored in self.data.

    # NOTE: mu is the symbol for mean, sigma is the symbol for coefficients of variance.
    def __mu_sigma(self):
        '''
        '''
        self.log_cvs, self.log_means, self.nonzero_idxs = mu_sigma(self.mtx)

    def pca(self, nfeats):
        '''
        Run PCA on the data and store the results.
        '''
        self.nfeats = nfeats # Store the number of features.

        # Run principal component analysis.
        model = sklearn.decomposition.PCA(n_components=nfeats,
                                        copy=True,
                                        whiten=False,
                                        svd_solver='auto',
                                        tol=0.0,
                                        iterated_power='auto',
                                        random_state=None)
        # Isn't data already an array? Why do we take the transpose?
        # The matrix stored is like the original data matrix, but with the gene axis replaced by features. 
        self.pca = model.fit_transform(self.data.to_array().T)

    def onmf(self, nfeats):
        '''
        Run orthogonal nonnegatice matrix factorization (oNMF) on the data and store the results. 
        '''
        self.nfeats = nfeats # Store the number of features.

        # NOTE: Need to learn more about the oNMF algorithm to figure out what these parameters are doing.
        # n_iter is the maximum number of iterations (for what?)
        # The algorithm converged if the reconstruction value is less than residual.
        # tof is the tolerance of the stopping condition.
        A, B = PA.oNMF(self.mtx, nfeats, n_iter=500, verbose=0, residual=1e-4, tof=1e-4)
        
        # Rescale A and recompute B. NOTE: Why do we bother doing this?
        A = PA.scale_W(A) # This rescales the A column (?) vectors so that each has a unit size of 1
        new_B = []
        for i in range(self.mtx.shape[1]):
            # Call the nnls function, iterating over each column.
            new_B.append(PA.nnls(A, self.filtered_mtx[:,i].toarray().flatten()))
        B = np.vstack(new_B) # NOTE: Don't take transpose here (as is done in the pipeline) for consistency with PCA. 
        
        self.onmf_A = A
        self.onmf_B = B

            
    # NOTE: Figure out how to get this to work with oNMF -- see if the results are any different. 
    # I'm pretty sure the second matrix from oNMF is the same as the matrix from PCA. 
    def cluster(self, nclusters=13, method='pca'):
        '''
        Runs a kmeans clustering algorithm on the data.
        '''
        if method == 'pca':
            arr = self.pca # Cells by features array. 
        elif method == 'onmf':
            if self.onmf_B is not None:
                arr = self.onmf_B # Cells by features array.
        else:
            raise Exception('A valid method must be entered (either pca or onmf)')
        if arr is None:
            answer = input(f"{method.upper} has not yet been run. Proceed with {method.upper} (y/n)? ", end='')
            print()
            if answer == 'n':
                return
            elif answer == 'y':
                eval(f"self.{method}({self.nfeats})")
        self.kmeans = sklearn.cluster.KMeans(n_clusters=nclusters, random_state=0).fit(arr)
        
        mtx_by_cluster = {}
        # Store each cluster as its own matrix.
        for cluster in range(nclusters):
            # Get the indices of the cells belonging to cluster `cluster`
            cluster_idxs = np.where(self.kmeans.labels_ == cluster)
            # NOTE: self.filtered_mtx has the raw data (with filtered genes)
            mtx_by_cluster[str(cluster)] = self.filtered_mtx[:, cluster_idxs]

        self.mtx_by_cluster = mtx_by_cluster # Store the resulting dictionary 


    def _plotter(self):
        '''
        '''
        
        pass

    def __plot_tsne(self, by='cluster', samples=None):
        '''
        Creates a tSNE plot for all cells in the experiment, which color overlayed demarcating the specified samples. 
        If samples is None, then all samples are used by default.
        '''
        if by == 'cluster':
            if self.kmeans is None: # If the clustering algorithm has not yet been run...
                print("Default clustering settings used. nclusters=13, method='pca'.")
                self.cluster()
            self.__plot_cluster_tsne()
        elif by == 'sample':
            if samples is None:
                samples = self.samples
            self.__plot_sample_tsne(samples)


    def __plot_sample_tsne(self, axes, samples, cmap):
        '''
        Creates a tSNE plot with points labelled according to sample. 
        '''
        idxs = [] # This will be a list of numpy arrays.
        for sample in samples:
            sample_idxs = np.where(self.sample_labels == sample)
            idxs.append(sample_idxs)

        cmap = plt.get_cmap(cmap, len(idxs)) # Use len(idxs) instead of len(samples) to account for possible rep merging. 
        axes.scatter(self.tsne[:, 0], self.tsne[:, 1], cmap=cmap)


    def __plot_cluster_tsne(self, axes, cmap):
        '''
        '''
        pass


    def tsne(self, method='pca'):
        '''
        Generated TSNE coordinates using the previously generated PCA projection or oNMF.
        The coordinates are stored in the object.
        '''
        if method == 'pca':
            arr = self.pca # Cells by features array. 
        if method == 'onmf':
            if self.onmf_B is not None:
                arr = self.onmf_B # Cells by features array.
        else:
            raise Exception('A valid method must be entered (either pca or onmf)')
        if arr is None:
            answer = input(f"{method.upper} has not yet been run. Proceed with {method.upper} (y/n)? ", end='')
            print()
            if answer == 'n':
                return
            elif answer == 'y':
                eval(f"self.{method}({self.nfeats})")

        self.tsne = sklearn.decomposition.TSNE(n_components=2).fit_transform(arr)


    def __plot_heatmap():
        '''
        '''
        pass
 

def mu_sigma(mtx):
    '''
    This function computes the mu and sigma values of the inputted sparse matrix.
    It returns the 
    Parameters
    ----------
    mtx : sparse matrix
    Matrix for which genes mu and sigma values will be computed
    '''
    cellcount = mtx.shape[1] # Get the number of cells (i.e. number of columns).
    genecount = mtx.shape[0] # Get the number of genes (i.e. the number of rows).

    # Calculating the means across the columnis, i.e. for each cell. 
    means, var = np.mean_variance_axis(mtx, axis=1) # Means and variances. 
    std = np.sqrt(var) # Standard deviations. 
    # NOTE: What does each part of this function do?
    cvs = np.divide(std, means, out=np.zeros_like(std), where=(means != 0)) # Coefficient of variation.

    # NOTE: What is the purpose of converting to Compressed Sparse Row format?
    mtx_csr = mtx.tocsr() 
    # Count how many cells have nonzero expression for each gene. 
    # NOTE: What does this do?
    presence = np.array([mtx_csr.indptr[i + 1] - mtx_csr.indptr[i] for i in range(genecount)])
    # Get indices of the genes that are expressed in more than 0.1 percent of the cells. 
    presence_idx = np.where(presence > cellcount*0.001)[0] 
    # mtx_csr = None # What is this for?

    # Get the indices of genes that have both a nonzero means and are present in more than 0.1 percent of cells.
    # NOTE: In what case would there be a zero means and a nonzero presence?
    nonzero_idxs= np.intersect1d(np.nonzero(means)[0], presence_idx) 

    # Get the values corresponding to the nonzero indices and take the common log.
    nonzero_cvs = cvs[nonzero_idxs] 
    nonzero_means = means[nonzero_idxs] 
    log_cvs = np.log10(nonzero_cvs) 
    log_means = np.log10(nonzero_means)
    
    return log_cvs, log_means, nonzero_idxs
