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

class HeatmapPlot(plot.Plot):
    '''

    '''
    def __init__(self, obj,
                 is_subplot=False,
                 merge_samples=True,
                 genes=None, 
                 samples=None,
                 cluster=True, 
                 **kwargs):
        '''
        Initializes a HeatmapPlot object.

        Parameters
        ----------
        obj : data.Data
            The Data object.
        is_subplot : bool
            Whether or not the heatmap is a subplot.
        genes : list
            A list of genes to plot on the HeatmapPlot. If the object contains diffexp_genes, then this is 
            set as the default gene list.
        samples : list
            A list of samples to plot on the HeatmapPlot. This is optional. It allows the user to manually
            adjust sample ordering or exclude samples from the analysis. 
        cluster : bool
            Whether or not to cluster the data. Clustering options can be specified in cluster_params.
        kwargs : N/A
            A series of keyword arguments, details of which are specified in the documentation. 
        '''
        self.merge_samples = merge_samples
        
        # Parent class initialization --------------------------------------------------------
        super().__init__(obj, 
                         is_subplot=is_subplot, 
                         color='coolwarm', 
                         filename='heatmap', 
                         plotter=self._plotter) 
 
        # Clustering -----------------------------------------------------------------------
        self.cluster_ = cluster 
        self.cluster_plot_dendrograms = kwargs.get('cluster_plot_dendrograms', True)
        self.cluster_metric = kwargs.get('cluster_metric', 'euclidean') 
        self.cluster_linkage = kwargs.get('cluster_linkage', 'complete') 
        self.cluster_nclusters = kwargs.get('cluster_nclusters', 3) 
        self.cluster_axis = kwargs.get('cluster_axis', 'both')
        
        self.clusters = {} # A dictionary in which to store the cluster groups. 
        self.lms = [None, None] # A list which will store the linkage matrices.
        
        # Data inititalization ----------------------------------------------------------        
        self.celltype = obj.celltype
        self.gene_order = obj.genes # The gene order for obtaining corresponding gene indices. 
        self.samples_order = obj.samples # The sample order for obtaining corresponding sample indices.

        # Inititalize the samples attribute; if no sample list is specified, use obj.samples.
        if samples is None:
            self.samples = np.array(obj.samples)
        else:
            self.samples = np.array(samples)
       
        if genes is None: # If no genes list is specified, try using the differential expression list.
            assert obj.diffexp_genes is not None, 'A genes list must be specified.'
            self.genes = obj.diffexp_genes
        else: 
            genes = np.array(genes)
            # Remove the genes which are not also in the Data object.
            self.genes = genes[np.in1d(genes, self.gene_order)] 
            
        # Data collection ------------------------------------------------------------------------
        # NOTE: The Data object is passed in to avoid unnecessarily copying it to the Plot object.
        self.data = self.__get_data(obj)
        
        self.ylabels = self.samples
        self.xlabels = self.genes 
        self.title = f'Expression in {self.celltype}'

        # Adjusting the filepath -------------------------------------------------------------
        self.filepath.append('heatmap') # Assign plot to 'heatmap' directory.
        self.filename = f'heatmap.png'
    
    def __get_data(self, obj):
        '''
        Gets the data to be plotted on the heatmap. This function creates and
        returns a 2-D array with the rows corresponding to samples in self.samples and the columns corresponding
        to genes.
        
        Parameters
        ----------
        obj : data.Data
            A Data object.
        '''
        ngenes, nsamples = len(self.genes), len(self.samples)
        t0 = time.time() # Get the start time for performance info.
        data = np.zeros((nsamples, ngenes)) # Initialize an array where rows are samples and columns are genes.
        for i in range(nsamples):
                sample = self.samples[i]
                print(f'Gathering data for sample {i} of {nsamples}...    \r', end='')
                sample_data = self.__get_sample_data(obj, sample=sample)
                data[i]= sample_data # Add the L1 data to the data matrix.
        t1 = time.time()
        print(f'All sample data gathered: {int(t1 - t0)} seconds    ')
        
        self.xlabels = self.genes # Set the ylabels BEFORE clustering.
        if self.cluster_: # If clustering is set to True...
            data = self.cluster(data=data)     
        
        return data
     
    def __get_sample_data(self, obj, sample=None):
        '''
        This function uses the precomputed data in the inputted diffexp_data dictionary to retrieve data
        for the given sample. 

        Parameters
        ----------
        obj : data.Data
            The Data object.
        sample : str
            The sample for which to gather the data. 
        '''
        # NOTE: The L1 data is in the order of diffexp_data['genes'], so we must use that to get the
        # correct indices.
        geneidxs = np.array([np.where(self.gene_order == gene)[0][0] for gene in self.genes])
        sampleidx = np.where(np.array(self.samples_order) == sample)[0] # Get the index of the sample.
        sample_l1s = np.array(obj.l1s)[sampleidx, geneidxs] # Filter by gene indices.
        
        return sample_l1s.flatten()

    # Clustering ------------------------------------------------------------------------------------
    
    def cluster(self, data=None, **kwargs):
        '''
        Cluster the data. By default, clustering parameters set during initialization will be used, although
        they can be overriden with keyword arguments.

        Parameters
        ----------
        data : numpy.array
            Only for use if cluster is being called prior to population of the self.data attribute (i.e. 
            within the initializer). None by default.
        **kwargs : N/A
            Various cluster settings. These will be passed directly into the self.__cluster_axis() function.
        '''
        if data is None:
            data = self.data
       
        if self.cluster_axis == 'x':
            data, xlabels = self.__cluster_axis(data, axis='x', **kwargs)
            self.xlabels = xlabels
        elif self.cluster_axis == 'y':
            data, ylabels = self.__cluster_axis(data, axis='y', **kwargs)
            self.ylabels = ylabels
        elif self.cluster_axis == 'both':
            data, xlabels  = self.__cluster_axis(data, axis='x', **kwargs)
            data, ylabels = self.__cluster_axis(data, axis='y', **kwargs)
            # Change the axes labels stored in the object to match the new ordering.
            self.xlabels = xlabels
            self.ylabels = ylabels

        self.data = data # Reassign the data attribute.
        return data # Return data just cause.

    def __cluster_axis(self, data, axis=None, **kwargs):
        '''
        Runs an agglomerative hierarchical clustering algorithm on the inputted matrix along the 
        specified axis. It uses the cluster settings inputted at Heatmap initialization, which can 
        be overriden by keyword arguments. It returns the modified data and the reordered labels.

        Parameters
        ----------
        data : numpy.ndarray
            A 2-D array which contains Heatmap data.
        axis : str
            One of 'x' or 'y'. If 'y', the rows of the matrix are clustered; if 'x', 
            the columns of the matrix are clustered.
        '''
 
        # Load the cluster settings, which can be overriden with keyword arguments to the function.
        metric = kwargs.get('metric', self.cluster_metric) 
        linkage = kwargs.get('linkage', self.cluster_linkage) 
        nclusters = kwargs.get('nclusters', self.cluster_nclusters) 
        
        if axis == 'x': # If clustering is to be carried out by column...
            matrix = data.T # Get the transpose of the inputted matrix.
            labels = self.xlabels
        elif axis == 'y':
            matrix = data
            labels = self.ylabels
            
        # NOTE: This function returns a  matrix such that element {i, j} is the distance between 
        # the ith and jth vectors of the given matrix. 
        X = skl.metrics.pairwise_distances(X=matrix, metric=metric) # Get the distance matrix.
        model = skl.cluster.AgglomerativeClustering(n_clusters=nclusters,
                                                    affinity='precomputed', # Distance matrix was precomputed.
                                                    linkage=linkage) # Create the clustering model.
 
        # Get the indices with which to sort the data and data labels.
        clusteridxs = model.fit_predict(X=X) # Return the clustering labels.
        nelements = len(clusteridxs) # Get the number of elements to sort.
        sorter = [idx for (_, idx) in sorted(zip(clusteridxs, list(range(nelements))))]
        
        self.clusters[axis] = []
        for cluster in range(nclusters): # Store the clusters in an attribute.
            idxs = np.where(clusteridxs == cluster)[0]
            self.clusters[axis].append(np.array(labels)[idxs])
   
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

    def __plot_dendrogram(self, daxes, flip_axes=False):
        '''
        Plots dendrograms on the x, y, or both axes, depending on which linkage matrices have been
        calculated.
        
        Parameters
        ----------
        daxes : tuple
            A two-tuple of matplotlib.axes.Axes objects on which to plot the dendrograms.
        '''
        xdax, ydax = daxes # Get the axes for the x and y-axis dendrograms.
        ydax.axis('off') # Turn off axes display.
        xdax.axis('off')

        if flip_axes: # If flip_axes == True, then switch the linkage matrix used to create each dendrogram.
            xlm, ylm = self.lms[1], self.lms[0]
        else:
            xlm, ylm = self.lms[0], self.lms[1]

        if xlm is not None: # If the x-axis was clustered...
            sch.dendrogram(xlm, ax=xdax, 
                           orientation='top', 
                           color_threshold=0,
                           above_threshold_color='black',
                           no_labels=True)
        if ylm is not None: # If the y-axis was clustered...
            sch.dendrogram(ylm, ax=ydax, 
                           orientation='left',
                           color_threshold=0,
                           above_threshold_color='black',
                           no_labels=True)

    def _plotter(self, axes, 
                 color=None, 
                 cutoff=None,
                 flip_axes=True):
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
        **kwargs : N/A
            Other HeatmapPlot-specific plotting settings. More information can be found in the documentation.
        '''
        assert isinstance(color, str), 'Color must be a string for a HeatMap object.'

        # Retrieve the data and data labels for the subplot.
        if flip_axes:
            data = np.transpose(self.data)
            ylabels, xlabels = self.xlabels, self.ylabels
        else:
            data = self.data
            xlabels, ylabels = self.xlabels, self.ylabels

        if cutoff is not None: # If a filter is set, set all values below the cutoff to zero.
            # Set all values within the cutoff 'zone' to zero.
            data = np.where((data < cutoff) & (data > -1 * cutoff), 0.0, data)
        
        axes.axis('off') # Turn off the axes frame.
        
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
        if self.cluster_ and self.cluster_plot_dendrograms:
            d = 0.075 * w # The thickness of the dendrogram plots.
            mainw, mainh = 0.7 * w, 0.9 * h # The dimensions of the main axes.
            
            # Create axes for the x-and-y dendrograms.
            ydax = self.figure.add_axes([x0, y0, d, mainh], frame_on=False)
            xdax = self.figure.add_axes([x0 + d, y0 + mainh, mainw, d], frame_on=False)
            # Plot the dendrorgrams.
            self.__plot_dendrogram((xdax, ydax))
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
        mainax.imshow(data, cmap=cmap, aspect='auto', vmin=-2.0, vmax=2.0) 
        # Set the axes ticks. 
        mainax.set_xticks(np.arange(0, len(xlabels))) 
        mainax.set_yticks(np.arange(0, len(ylabels))) 
        # Set the axes labels, with the correct orientation and font size.
        mainax.set_yticklabels(ylabels, fontdict={'fontsize':self.y_fontsize})
        xlabels = mainax.set_xticklabels(xlabels, fontdict={'fontsize':self.x_fontsize})
        for label in xlabels: # Make x-axis labels vertical.
            label.set_rotation('vertical')
        # If dendrograms are plotted, move the y-tick labels to make room for the dendrograms.
        if self.cluster_ and self.cluster_plot_dendrograms: 
            mainax.yaxis.set_label_position('right')
            mainax.yaxis.tick_right() 
       
        # Add a colorbar to the colorbar axes.
        norm = mpl.colors.Normalize(vmin=-2.0, vmax=2.0)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) # Turn the selected colormap into a ScalarMappable object.
        cbar = plt.colorbar(mappable, cax=cax, ticks=[-2, 0, 2])
        cbar.ax.set_title('L1 norm', fontdict={'fontsize':20})
