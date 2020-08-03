import matplotlib as mpl
import re
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import time
import scipy.cluster.hierarchy as sch

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

from plotpop import plot
from plotpop import bar

class HeatmapPlot(plot.Plot):
    '''

    '''
    def __init__(self, 
                 pop=None,
                 diffexp_data=None,
                 type_=None,
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
        pop : dict
            The pop object.
        diffexp_data : dict
            The object storing information on differential expression. 
        type_ : str
            One of 'ct', rp'. This specifies what will be plotted on the Heatmap.
        is_subplot : bool
            Whether or not the heatmap is a subplot.
        genes : list
            A list of genes to plot on the HeatmapPlot. This must be specified if the plot is initialized
            with a pop obect. If initialized with diffexp_data, this defaults to the list of differentially-
            expressed genes (diffexp_data['diffexp']['all'].
        samples : list
            A list of samples to plot on the HeatmapPlot. This is optional. It allows the user to manually
            adjust sample ordering or exclude samples from the analysis. 
        cluster : bool
            Whether or not to cluster the data. Clustering options can be specified in cluster_params.
        kwargs : N/A
            A series of keyword arguments, details of which are specified in the documentation. 
        '''
        options = ['ct', 'rp']
        assert type_ in options, f'The type_ parameter must be one of: {options}.'
        self.type_ = type_
           
        self.merge_samples = merge_samples
        
        # Parent class initialization --------------------------------------------------------
        assert not (diffexp_data and pop), 'Only one of pop, diffexp_data may be given.'
        if diffexp_data is not None:
            obj = diffexp_data
        elif pop is not None:
            obj = pop
        else:
            raise Exception('At least one of pop, diffexp_data must be specified.')

        super().__init__(obj, is_subplot=is_subplot) 
        self.plotter = self._plotter
        self.color = 'coolwarm'
 
        # Clustering -----------------------------------------------------------------------
        self.cluster_ = cluster 
        self.cluster_plot_dendrograms = kwargs.get('cluster_plot_dendrograms', True)
        self.cluster_metric = kwargs.get('cluster_metric', 'euclidean') 
        self.cluster_linkage = kwargs.get('cluster_linkage', 'complete') 
        self.cluster_nclusters = kwargs.get('cluster_nclusters', 3) 
        self.cluster_axis = kwargs.get('cluster_axis', 'both')
        
        self.clusters = {} # A dictionary in which to store the cluster groups. 
        self.lms = [None, None] # A list which will store the linkage matrices.
        
        # Initialization with diffexp_data ----------------------------------------------------------        
        if diffexp_data is not None:
            # NOTE: All checks on the samples and genes were carried out in the get_diffexp_data()
            # method in the plot module.
            self.gene_order = diffexp_data['genes'] # The gene order for obtaining corresponding gene indices. 
            self.samples_order = diffexp_data['samples'] # The sample order for obtaining corresponding sample indices.
            if genes is None:
                self.genes = diffexp_data['diffexp']['all'] 
            else:
                self.genes = genes
            # Inititalize the samples attribute; if no sample list is specified, use diffexp['samples'].
            if samples is None:
                self.samples = np.array(diffexp_data['samples'])
            else:
                self.samples = samples

            # This function initializes the following attributes: diffexp, upreg, downreg, all_l1s. 
            self.__load_diffexp_data(diffexp_data)
            
            # Check that the inputted diffexp data is compatible with the plot type.
            fltr = diffexp_data['filter'][0]
            if self.type_ == 'ct':
                assert fltr == 'celltype', \
                        f'diffexp_data with filter {fltr} is incompatible with HeatmapPlot type ct.'
                self.celltype = diffexp_data['filter'][1]
            elif self.type_ == 'rp':
                assert fltr == 'refpop', \
                        f'diffexp_data with filter {fltr} is incompatible with HeatmapPlot type rp.'
                self.refpop = diffexp_data['filter'][1]
        
            unsupervised = True # Gene selection was carried out using an unsupervised algorithm.
            
        # Standard initialization ------------------------------------------------------
        elif pop is not None:
            self.pop = pop # Store the pop object as an attribute.
            self.ctrls = [s for s in pop['samples'].keys() if re.match(pop['controlstring'], s) is not None]
            
            assert genes is not None, 'A genes list must be specified when initializing with a pop object.'
            self.genes = plot.check_genes(pop, genes)            
            # Inititalize the samples attribute; if no sample list is specified, use pop['order'].
            if samples is None:
                samples = np.array(pop['order'])
            self.samples = plot.check_samples(pop, samples=samples, filter_reps=merge_samples, filter_ctrls=True)
            # If genes are specified in the arguments, assign them to self.genes. 

            if self.type_ == 'ct': # Samples filtered by a celltype.
                self.celltype = plot.check_celltype(pop, kwargs.get('celltype', None))
            elif self.type_ == 'rp': # Samples filtered by a reference population.
                self.refpop = kwargs.get('refpop', None)
                self.ref = pop['ref'] # Store the name of the reference sample.
                self.celltype = pop['samples'][self.ref]['gmm_types'][self.refpop] # Get the type of the subpopulation.
            
            unsupervised = False # A curated list of genes, and no precomputed L1 matrix, are being used.

        # Parent class initialization (continued) ----------------------------------------------------------
        # NOTE: These need to be set following the type-specific initialization for HeatmapPlots.
        self.ylabels = self.samples
        self.xlabels = self.genes 
        self.title = f'Expression in {self.celltype}'

        # Data collection ------------------------------------------------------------------------
        # Gather data without either using the precomputed diffexp object or with a pop object.
        self.data = self.__s_get_data(unsupervised=unsupervised)
        
        # Adjusting the filepath -------------------------------------------------------------
        self.filepath.append('heatmap') # Assign plot to 'heatmap' directory.
        self.filename = f'heatmap.png'
    
    def __load_diffexp_data(self, diffexp_data):
        '''
        Load relevant data from a diffexp_data object into the HeatmapPlot. It stores all differentially 
        up and down-regulated genes (in any sample), as well as the stairstep-order list of differentially
        expressed genes and the 2-D pop['order'] by pop['filtered_genes'] array of L1 values. 
        
        Parameters
        ----------
        diffexp_data : dict
            An object produced by the plot.get_diffexp_data() function which stores information
            about differentially-expressed genes. 
        '''
        upreg, downreg = np.array([]), np.array([])
        # Get lists of all genes differentially up or down-regulated in any sample.
        for sample in self.samples:
            upreg = np.append(upreg, diffexp_data['diffexp']['samples'][sample]['up'])
            downreg = np.append(downreg, diffexp_data['diffexp']['samples'][sample]['down'])
        
        # Remove duplicates.
        upreg = np.unique(upreg)
        downreg = np.unique(downreg)
        
        # Inititalize the attributes.
        self.diffexp = diffexp_data['diffexp']['all']
        self.upreg = upreg
        self.downreg = downreg
        self.all_l1s = diffexp_data['l1s']

    # S_* ------------------------------------------------------------------------------------------------ 
    
    def __s_get_data(self, unsupervised=False):
        '''
        Gets the data to be plotted on the heatmap. This function creates and
        returns a 2-D array with the rows corresponding to samples in self.samples and the columns corresponding
        to genes. If the unsupervised option is set, unsupervised analysis is use to calculate all differentially
        expressed genes.
        
        Parameters
        ----------
        unsupervised : bool
            Whether :or not genes should be selected by an unsupervised algorithm.
        '''
        if unsupervised: 
            data_getter = self.__get_sample_data_unsupervised
        else: # If a list of genes has been specified... 
            data_getter = self.__get_sample_data
        
        ngenes, nsamples = len(self.genes), len(self.samples)
        t0 = time.time() # Get the start time for performance info.
        data = np.zeros((nsamples, ngenes)) # Initialize an array where rows are samples and columns are genes.
        for i in range(nsamples):
                sample = self.samples[i]
                print(f'Gathering data for sample {i} of {nsamples}...    \r', end='')
                sampledata = data_getter(sample=sample)
                data[i]= sampledata # Add the L1 data to the data matrix.
        t1 = time.time()
        print(f'All sample data gathered: {int(t1 - t0)} seconds    ')
        
        self.xlabels = self.genes # Set the ylabels BEFORE clustering.
        if self.cluster_: # If clustering is set to True...
            data = self.cluster(data=data)     
        
        return data
     
    def __get_sample_data(self, sample=None):
        '''
        Retrieve the L1 norms for a certain sample for all specified genes.

        Parameters
        ----------
        sample : str
            The sample from which to collect data.
        cutoff : float
            Not for use in this function; argument is present for consistency.
        merge_samples : bool
        '''
        genes = self.genes
        if self.type_ == 'ct':
            bar_type = 'g_s_ct'
            bar_params = {'celltype':self.celltype}
        elif self.type_ == 'rp':
            bar_type = 'g_s_rp'
            bar_params = {'refpop':self.refpop}
        
        l1s = np.array([])
        for gene in genes:
            bar_params.update({'gene':gene, 'sample':sample, 'merge_samples':True})
            bp = bar.BarPlot(self.pop, type_=bar_type, **bar_params)    
            l1 = bp.calculate_l1() # Get the L1 metric for the distribution (reference is control by default).
            l1s = np.append(l1s, l1) # Add the L1 value.

        return l1s
    
    def __get_sample_data_unsupervised(self, sample=None):
        '''
        This function uses the precomputed data in the inputted diffexp_data dictionary to retrieve data
        for the given sample. 

        Parameters
        ----------
        sample : str
            The sample for which to gather the data. 
        '''
        # NOTE: The L1 data is in the order of diffexp_data['genes'], so we must use that to get the
        # correct indices.
        geneidxs = np.array([np.where(self.gene_order == gene)[0] for gene in self.genes])
        # NOTE: The L1 data is in the order of pop['order'], so we must use the indices from that to 
        # pull out the correct data.
        sampleidx = np.where(np.array(self.samples_order == sample))[0] # Get the index of the sample.
        l1s = self.all_l1s[sampleidx, geneidxs] # Filter by gene indices.
        
        return l1s.flatten()

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
                 flip_axes=True,
                 **kwargs):
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
