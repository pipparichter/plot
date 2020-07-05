import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import time
import scipy.cluster.hierarchy as sch

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

from plotpop import plot
from plotpop import barplot

class HeatmapPlot(plot.Plot):
    '''

    '''
    def __init__(self, pop, 
                 type_=None,
                 is_subplot=False,
                 cluster=True, 
                 **kwargs):
        '''
        Initializes a HeatmapPlot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        type_ : str
            One of 's_ct', 'sp_s', or 'sp_rp'. This specifies what will be plotted on the Heatmap.
        is_subplot : bool
            Whether or not the heatmap is a subplot.
        cluster : bool
            Whether or not to cluster the data. Clustering options can be specified in cluster_params.
        kwargs : N/A
            A series of keyword arguments, details of which are specified in the documentation. 
        '''
           
        # Initialize dictionary of genes and their corresponding indices.
        self.genedict = {gene:pop['filtered_genes'].index(gene) for gene in pop['filtered_genes']}

        self.merge_samples = kwargs.get('merge_samples', True)
        
        # Parent class initialization --------------------------------------------------------
        super().__init__(pop, is_subplot=is_subplot) 
        self.plotter = self._plotter
        self.color = 'bwr'
        
        # Clustering -----------------------------------------------------------------------
        self.cluster = cluster 
        self.cluster_plot_dendrograms = kwargs.get('cluster_plot_dendrograms', True)
        self.cluster_metric = kwargs.get('cluster_metric', 'euclidean') 
        self.cluster_linkage = kwargs.get('cluster_linkage', 'complete') 
        self.cluster_nclusters = kwargs.get('cluster_nclusters', 3) 
        self.cluster_axis = kwargs.get('cluster_axis', 'both')

        self.lms = [None, None] # A list which will store the linkage matrices.
        
        # Type-specific initialization ------------------------------------------------------
        options = ['s_ct', 'sp_s', 'sp_rp']
        assert type_ in options, f'The type_ parameter must be one of: {options}.'
        self.type_ = type_

        if type_ == 's_ct':
            celltype = kwargs.get('celltype', None)
            genes = kwargs.get('genes', None)
            samples = kwargs.get('samples', None)
            self.celltype = plot.check_celltype(pop, celltype)
            self.xlabels, self.ylabels = plot.check_genes(pop, genes), plot.check_samples(pop, samples)
            self.data = self.__s_ct_get_data() # Populate the data attribute.
                
        elif type_ == 'sp_s':
            pass
        
        elif type_ == 'sp_rp':
            pass
        
        # Adjusting the filepath -------------------------------------------------------------
        self.filepath.append('heatmap') # Assign plot to 'heatmap' directory.
        self.filename = f'heatmap.png'

    
    # S_CT ------------------------------------------------------------------------------------------------
    
    def __s_ct_get_data(self):
        '''
        Gets the data to be plotted on the heatmap for the given celltype. This function creates and
        returns a 2-D array with the rows corresponding to samples in self.samples and the columns corresponding
        to genes in self.genes. 
        
        Parameters
        ----------
        celltype : str
            The celltype for which to collect the data, i.e. the label of the subplot.
        '''
 
        genes, samples, celltype = self.xlabels, self.ylabels, self.celltype
        ngenes, nsamples = len(genes), len(samples)

        t0 = time.time() # Get the start time for performance info.
        print(f'Gathering {celltype} data...\r', end='')
        
        nans = [] # List to store the indices to be poplulated with NaNs.
        # Initialize a 2-D array, where there is a row for each sample and a column for each gene.
        data = np.zeros((nsamples, ngenes))
        for i in range(nsamples):
            sample = samples[i]
            rep = sample + '_rep'
            
            for j in range(ngenes):
                gene = genes[j]
                # ncells = self.pop['samples'][sample]['cell_type'].count(celltype)
                ncells = np.count_nonzero(self.pop['samples'][sample]['cell_type'] == celltype)

                if self.merge_samples: 
                    # ncells += self.pop['samples'][rep]['cell_type'].count(celltype)
                    ncells += np.count_nonzero(self.pop['samples'][rep]['cell_type'] == celltype)
                    params = {'gene':gene, 'celltype':celltype, 'sample':sample, 'merge_samples':True}
                    distribution = barplot.BarPlot(self.pop, type_='g_s', **params)
                else:
                    distribution = barplot.BarPlot(self.pop, gene=gene, celltype=celltype, sample=sample, merge_samples=False)
                mincells = 50 # The minimum number of cells needed to make a valid distribution
                if ncells < mincells:
                    nans.append((i, j))
                    
                l1 = distribution.calculate_l1() # Get the L1 metric for the distribution (reference is control by default).
                data[i, j] = l1
        
        t1 = time.time()
        print(f'{celltype} data gathered: {t1 - t0} seconds')

        if self.cluster: # If clustering is set to True...
            t0 = time.time()
            print(f'Clustering {celltype} data...\r', end='')
            
            if self.cluster_axis == 'x':
                data, genes = self.__cluster(data, axis='x')
            elif self.cluster_axis == 'y':
                data, samples = self.__cluster(data, axis='y')
            elif self.cluster_axis == 'both':
                data, genes  = self.__cluster(data, axis='x')
                data, samples = self.__cluster(data, axis='y')
            # Change the axes labels stored in the object to match the new ordering.
            self.xlabels = genes
            self.ylabels = samples

            t1 = time.time()
            print(f'{celltype} data clustered: {t1 - t0} seconds')

        # For the datapoints corresponding to distributions with ncells < mincells, replace the current value with NaN.
        for i, j in nans:
            data[i, j] = np.nan

        return data

    # SP_S -------------------------------------------------------------------------------------------------------

    def __sp_s_get_data(self, sample): # IN PROGRESS
        '''
        This function gets the data for a gene-by-subpopulation heatmap, for the subplot corresponding to the
        inputted sample.

        Parameters
        ----------
        sample : str
            The sample for which the data is being gathered. If 'allsamples', then the overarching GMM is used.
        index : int
            The index corresponding to the subplot for which the data is being gathered. 
        '''
        # Make sure feature data has been initialized, which is done in the onmf() function. 
        assert 'C' in self.pop['samples'][sample].keys(), 'Feature data has not been initialized.'

        genes = self.xlabels[sample].tolist()
        geneidxs = [self.genedict[gene] for gene in genes]

        if sample == 'allsamples':
            gmm, gmmtypes= self.pop['gmm'], self.pop['gmm_types']
            M = PA.cat_data(self.pop, 'M_norm')
            C = PA.cat_data(self.pop, 'C')
            gmmtypes = self.pop['gmm_types']
        else:
            gmm, gmmtypes = self.pop['samples'][sample]['gmm'], self.pop['samples'][sample]['gmm_types']
            M = self.pop['samples'][sample]['M_norm']
            C = self.pop['samples'][sample]['C']
        
        M = M[geneidxs, :]
        prediction = gmm.predict(C) # Get subpopulation assignments from the selected GMM.
        
        ncells = [] # Empty list to store number of cells per subpopulation.
        for i in range(gmm.n_components): # For each subpopulation in the GMM...
            cellidxs = np.where(prediction == i) # Get the indices of the cells in the ith subpopulation.

    # SP_RP -------------------------------------------------------------------------------------------

    def __sp_rp_get_data(self, refpop): # IN PROGRESS
        '''
        '''
        pass

    # ---------------------------------------------------------------------------------------------------

    def __cluster(self, data, axis=None):
        '''
        Runs an agglomerative hierarchical clustering algorithm on the inputted matrix. It uses
        the cluster settings inputted at Heatmap initialization. It returns the modified matrix, 
        the reordered data labels, and the clustering model generated.

        Parameters
        ----------
        matrix : numpy.ndarray
            A 2-D array which contains Heatmap data.
        labels : numpy.array
            An array which contains the data labels for the axis being sorted. 
        axis : str
            One of 'x' or 'y'. If 'y', the rows of the matrix are clustered; if 'x', 
            the columns of the matrix are clustered.
        '''
        if axis == 'x': # If clustering is to be carried out by column...
            matrix = data.T # Get the transpose of the inputted matrix.
            labels = self.xlabels
        elif axis == 'y':
            matrix = data
            labels = self.ylabels
            
        # NOTE: This function returns a  matrix such that element {i, j} is the distance between 
        # the ith and jth vectors of the given matrix. 
        X = skl.metrics.pairwise_distances(X=matrix, metric=self.cluster_metric) # Get the distance matrix.
        model = skl.cluster.AgglomerativeClustering(n_clusters=self.cluster_nclusters,
                                                    affinity='precomputed', # Distance matrix was precomputed.
                                                    linkage=self.cluster_linkage) # Create the clustering model.
        
        sorter = get_sorter(model, X=X) # Get the indices with which to sort the data and data labels.
        labels = labels[sorter] # Sort the labels to match the data.
        if axis == 'y':
            data = data[sorter, :] # Sort the data.
        elif axis == 'x':
            data = data[:, sorter] # Sort the data.

        # NOTE: Make sure to re-calculate the distance matrix for non-transposed and sorted data so that the 
        # dendrograms are accurate.
        if axis == 'x':
            X = skl.metrics.pairwise_distances(X=data.T, metric=self.cluster_metric) # Re-calculate the distance matrix.
            self.lms[0] = sch.linkage(X, method=self.cluster_linkage) # Get the linkage matrix.
        elif axis == 'y':
            X = skl.metrics.pairwise_distances(X=data, metric=self.cluster_metric) # Re-calculate the distance matrix.
            self.lms[1] = sch.linkage(X, method=self.cluster_linkage) # Get the linkage matrix.

        return data, labels

    def __plot_dendrogram(self, daxes):
        '''
        Plots dendrograms on the x, y, or both axes, depending on which linkage matrices have been
        calculated.
        '''
        xdax, ydax = daxes # Get the axes for the x and y-axis dendrograms.
        ydax.axis('off') # Turn off axes display.
        xdax.axis('off')
        if self.lms[0] is not None: # If the x-axis was clustered...
            lm = self.lms[0] # Retrieve the x-axis linkage matrix.
            sch.dendrogram(lm, ax=xdax, 
                           orientation='top', 
                           color_threshold=0,
                           above_threshold_color='black',
                           no_labels=True)
        if self.lms[1] is not None: # If the y-axis was clustered...
            lm = self.lms[1] # Retrieve the y-axis linkage matrix.
            sch.dendrogram(lm, ax=ydax, 
                           orientation='left',
                           color_threshold=0,
                           above_threshold_color='black',
                           no_labels=True)

    def _plotter(self, axes, color=None, fontsize=None):
        '''
        Plots a heatmap on the inputted axes.        
        
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes of the subplot on which the graph will be plotted.
        index : 
            The name of the subplot to be plotted.
        color: str
            The name of a matplotlib.colors.Colormap to color the heatmap data.
        fontsize : int
            Size of the font on the axes labels (not the title). 
        '''
 
        assert isinstance(color, str), 'Color must be a string for a HeatMap object.'
            
        # Retrieve the data and data labels for the subplot.
        data = self.data
        xlabels, ylabels = self.xlabels, self.ylabels
        
        axes.axis('off') # Turn off the axes frame.
        axes.set_title(f'Expression in {self.celltype}', fontdict={'fontsize':30})
        
        # Get information for creating a layout; if the HeatmapPlot is a subplot, the layout
        # will need to be constructed relative the the larger figure.
        if self.is_subplot:
            nplots = len(self.figure.axes)
            i = self.figure.axes.index(axes)
            nr, nc = PA.nr_nc(nplots) # Get the number of rows and columns.
            w, h = 1.0 / nc, 1.0 / nr
            x0, y0 =  ((i // nc) - 1) * h, (i % nc) * w
        else:
            w, h = 1, 1
            x0, y0 = 0, 0
        # Create a layout for all the subcomponents of the heatmap...
        # If cluster=True and plot_dendrograms is set to True (or left blank), plot the dendrograms.
        if self.cluster and self.cluster_plot_dendrograms:
            d = 0.075 * w # The thickness of the dendrogram plots.
            mainw, mainh = 0.7 * w, 0.9 * h # The dimensions of the main axes.
            
            # Create axes for the x-and-y dendrograms.
            ydax = self.figure.add_axes([x0, y0, d, mainh], frame_on=False)
            xdax = self.figure.add_axes([x0 + d, y0 + mainh, mainw, d], frame_on=False)
            self.__plot_dendrogram((xdax, ydax)) # Plot the dendrograms.
        else:
            d = 0
            mainw, mainh = 0.775 * w, 0.975 * h
           
        # Create axes for the main heatmap and the colorbar.
        c = 0.6 * h  # The length of the colorbar.
        mainax = self.figure.add_axes([x0 + d, y0, mainw, mainh], frame_on=False)
        cax = self.figure.add_axes([x0 + 0.95 * w, y0 + mainh / 2 - c / 2, 0.05 * w, c], frame_on=False)
       
        # Plot the heatmap on the main axes.
        cmap = plt.get_cmap(color) # Get the colormap.
        cmap.set_bad(color='gray') # Set NaN values to be gray.
        mainax.imshow(data, cmap=cmap, aspect='auto') # Auto scales the pixels according to the axes. 
        # Set the axes ticks. 
        mainax.set_xticks(np.arange(0, len(xlabels))) 
        mainax.set_yticks(np.arange(0, len(ylabels))) 
        # Set the axes labels, with the correct orientation and font size.
        mainax.set_yticklabels(ylabels, fontdict={'fontsize':fontsize})
        xlabels = mainax.set_xticklabels(xlabels, fontdict={'fontsize':fontsize})
        for label in xlabels: # Make x-axis labels vertical.
            label.set_rotation('vertical')
        # If dendrograms are plotted, move the y-tick labels to make room for the dendrograms.
        if self.cluster and self.cluster_plot_dendrograms: 
            mainax.yaxis.set_label_position('right')
            mainax.yaxis.tick_right() 
       
        # Add a colorbar to the colorbar axes.
        mappable = mpl.cm.ScalarMappable(cmap=cmap) # Turn the selected colormap into a ScalarMappable object.
        cbar = plt.colorbar(mappable, cax=cax, ticks=[0, 0.5, 1])
        cbar.ax.set_title('L1 norm', fontdict={'fontsize':fontsize})
        cbar.ax.set_yticklabels(['-2', '0', '2'], fontdict={'fontsize':fontsize})


# Accessory functions --------------------------------------------------------------------------------------

def get_sorter(model, X=None):
    '''
    Takes the indices generated by the AgglomerativeClustering model, which assigns each sample element to a 
    cluster, and produces a list of indices to sort each element according to its assigned cluster.
    
    Parameters
    ----------
    model : sklearn.cluster.AgglomerativeClustering
        The agglomerative clustering model generated using the data.
    X : np.array
        A 2-D array representing a distance matrix.
    '''
    clusteridxs = model.fit_predict(X=X) # Return the clustering labels.
    
    nelements = len(clusteridxs) # Get the number of elements to sort.
    idxs = list(range(nelements))
    sorter = [idx for (_, idx) in sorted(zip(clusteridxs, idxs))]
    
    return sorter

