import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import time
import scipy.cluster.hierarchy as sch
import re

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

from plotpop import plot
from plotpop import barplot

class HeatmapPlot(plot.Plot):
    '''

    '''
    # Class constructor.
    def __init__(self, pop, 
                 type_=None, 
                 x=None,
                 y=None,
                 merge_samples=False,
                 is_subplot=False,
                 cluster=True, 
                 cluster_settings={},
                 **kwargs):
        '''
        Initializes a HeatmapPlot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        genes : list
            A list of strings representing the official names of the genes to 
            be analyzed.
        type_ : str
            One of 's_ct', 'sp_s', or 'sp_rp'. This specifies what will be plotted on the Heatmap.
        samples : str, list
            A string or list of strings representing the samples to be analyzed. If None,
            all samples are used in the HeatmapPlot.
        celltypes : str, list
            A string or list of strings representing the celltypes to be analyzed. If None,
            all celltypes are used in the HeatmapPlot.
        refpops : list

        cluster : bool
            Whether or not to cluster the data. Clustering options can be specified in cluster_settings.
        merge_reps : bool
            Whether or not to merge replicate samples. 
        cluster_settings : dict
            Specific settings for clustering. Possible settings are as follows.
                axis : 'y', 'x', 'both'; 'both' by default.
                metric : e.g. 'l1', 'euclidean'; 'euclidean' by default.
                linkage : e.g. 'complete', 'single', 'ward'; 'complete by default.
                plot_dendrograms : True, False; True by default.
        '''
        super().__init__(pop, is_subplot=is_subplot) # Initialize underlying Plot object.
        self.plotter = self._plotter
        self.color = 'bwr'

        options = ['s_ct', 'sp_s', 'sp_rp']
        assert type_ in options, f'The type_ parameter must be one of: {options}.'
        self.type_ = type_

        self.xlabels, self.ylabels = self.__get_axes(x, y) # Get the labels for the x and y axes.
        
        self.merge_samples = merge_samples

        self.cluster = cluster # Whether or not the data will be clustered.
        # Initialize cluster settings.
        self.plot_dendrograms = self.cluster_settings.get('plot_dendrograms', True)
        self.metric = self.cluster_settings.get('metric', 'euclidean') 
        self.linkage = self.cluster_settings.get('linkage', 'complete') 
        self.nclusters = self.cluster_settings.get('nclusters', 3) 
        
        self.lms = [None, None] # A list which will store the linkage matrices.
        
        # Initialize dictionary of genes and their corresponding indices.
        self.genedict = {gene:pop['filtered_genes'].index(gene) for gene in pop['filtered_genes']}
        
        if type_ == 's_ct':
            self.celltype = kwargs.get('celltype', None)
            assert self.celltype is not None, 'For heatmap type s_ct, a celltype must be specified.'
            self.data = self.__s_ct_get_data() # Populate the data attribute.
        # elif type_ == 'sp_s':
        # elif type_ == 'sp_rp':
        
        self.filepath.append('heatmap') # Assign plot to 'heatmap' directory.
        self.filename = f'heatmap.png'

    
    # Private methods -------------------------------------------------
    
    def __get_axes(self, x, y):
        '''
        '''
        if self.type_ == 's_ct':
            # Filter the y labels (samples).
            controlstring = self.pop['controlstring']
            ylabels = [sample for sample in y if re.match(controlstring, sample) is None] 
            # Filter the x labels (genes).
            invalid, xlabels = [], []
            for gene in x:
                if gene in self.pop['filtered_genes']:
                    xlabels.append(gene)
                else:
                    invalid.append(gene)
            if len(invalid) > 1:
                print('The following genes are invalid and were removed: ' + ', '.join(invalid))
        # if self.type_ == 
        # if self.type_ ==
        
        return np.array(xlabels), np.array(ylabels)

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

                # t2 = time.time()
                if self.merge_samples: 
                    # ncells += self.pop['samples'][rep]['cell_type'].count(celltype)
                    ncells += np.count_nonzero(self.pop['samples'][rep]['cell_type'] == celltype)
                    distribution = barplot.BarPlot(self.pop, gene=gene, celltype=celltype, sample=sample, merge_samples=True)
                else:
                    distribution = barplot.BarPlot(self.pop, gene=gene, celltype=celltype, sample=sample, merge_samples=False)
                # t3 = time.time()
                # print(f'Barplot took {t3 - t2} seconds.')
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
            
            axis = self.cluster_settings.get('axis', 'both')
            if axis == 'x':
                data, genes = self.__cluster(data, axis='x')
            elif axis == 'y':
                data, samples = self.__cluster(data, axis='y')
            elif axis == 'both':
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

    def __sp_rp_get_data(self, refpop): # IN PROGRESS
        '''
        '''
        pass

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
            
        # Retrieve cluster settings.
        
        # NOTE: This function returns a  matrix such that element {i, j} is the distance between 
        # the ith and jth vectors of the given matrix. 
        X = skl.metrics.pairwise_distances(X=matrix, metric=self.metric) # Get the distance matrix.
        model = skl.cluster.AgglomerativeClustering(n_clusters=self.nclusters,
                                                    affinity='precomputed', # Distance matrix was precomputed.
                                                    linkage=self.linkage) # Create the clustering model.
        
        sorter = get_sorter(model, X=X) # Get the indices with which to sort the data and data labels.
        labels = labels[sorter] # Sort the labels to match the data.
        if axis == 'y':
            data = data[sorter, :] # Sort the data.
        elif axis == 'x':
            data = data[:, sorter] # Sort the data.

        # NOTE: Make sure to re-calculate the distance matrix for non-transposed and sorted data so that the 
        # dendrograms are accurate.
        if axis == 'x':
            X = skl.metrics.pairwise_distances(X=data.T, metric=self.metric) # Re-calculate the distance matrix with the sorted data.
            self.lms[0] = sch.linkage(X, method=self.linkage) # Get the linkage matrix.
        elif axis == 'y':
            X = skl.metrics.pairwise_distances(X=data, metric=self.metric) # Re-calculate the distance matrix with the sorted data.
            self.lms[1] = sch.linkage(X, method=self.linkage) # Get the linkage matrix.

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
            nplots = len(self.figure.get_axes())
            i = self.figure.get_axes().index(axes)
            nr, nc = PA.nr_nc(nplots) # Get the number of rows and columns.
            w, h = 1.0 / nc, 1.0 / nr
            x0, y0 =  ((i // nc) - 1) * h, (i % nc) * w
        else:
            w, h = 1, 1
            x0, y0 = 0, 0
        # Create a layout for all the subcomponents of the heatmap...
        # If cluster=True and plot_dendrograms is set to True (or left blank), plot the dendrograms.
        if self.cluster and self.plot_dendrograms:
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
        if self.cluster and self.plot_dendrograms: 
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

