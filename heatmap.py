import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import sklearn as skl
import time
import scipy.cluster.hierarchy as sch

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

from plot import plot
from plot import barplot

class HeatmapPlot(plot.Plot):
    '''

    '''
    # Class constructor.
    def __init__(self, pop, genes=None, toplot=None, samples=None, celltypes=None, refpops=None, cluster=True, **kwargs):
        '''
        Initializes a HeatmapPlot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        genes : list
            A list of strings representing the official names of the genes to 
            be analyzed.
        toplot : str
            One of 's_ct', 'sp_s', or 'sp_rp'. This specifies what will be plotted on the Heatmap.
        samples : str, list
            A string or list of strings representing the samples to be analyzed. If None,
            all samples are used in the HeatmapPlot.
        celltypes : str, list
            A string or list of strings representing the celltypes to be analyzed. If None,
            all celltypes are used in the HeatmapPlot.
        refpops : list

        cluster : bool
            
        **kwargs : str
        
        '''
        self.cluster = cluster # Whether or not the data will be clustered.
        self.clusterinfo = kwargs # Save the cluster settings as an attribute.

        self.celltypes = plot._init_celltypes(pop, celltypes) # Initialize cell type list.
        
        self.allsamples = (samples is None) # If no samples are specified, all samples are used.
        self.samples = plot._init_samples(pop, samples) # Initialize list of samples.
        
        self.genedict = plot._init_genedict(pop, genes) # Initialize dictionary of genes and their corresponding indices.
        genelist = list(self.genedict.keys())

        self.refpops = refpops

        options = ['s_ct', 'sp_s', 'sp_rp']
        assert toplot in options, f'The toplot parameter must be one of: {options}.'
        self.toplot = toplot

        if toplot == 's_ct':
            self.subplots = self.celltypes
            nplots = len(self.subplots)

            # NOTE: self._axis needs to be a 2-D array to support clustering on a per-subplot basis.
            self.xlabels = np.array([genelist] * nplots)
            self.ylabels = np.array([self.samples] * nplots)
            
            data_getter = self.__s_ct_get_data # Set the data retrieval function.
        
        elif toplot == 'sp_s':
            if self.allsamples:
                self.subplots = ['allsamples']
                nplots = 1
            else:
                self.subplots = self.samples
                nplots = len(self.subplots)
            
            self.xlabels = np.ndarray([genelist] * nplots)
            self.ylabels = None # This will be a list of lists.
            
            data_getter = self.__sp_s_get_data # Set the data retrieval function.
        
        elif toplot == 'sp_rp':
            self.subplots = self.refpops
            nplots = len(self.subplots)

            self.xlabels = np.ndarray([genelist] * nplots)
            self.ylabels = None # This will be a list of lists.
            
            data_getter = self.__sp_rp_get_data # Set the data retrieval function.
        
        self.lms = [[None, None]] * nplots # A list in which to store linkage matrices.
    
        super().__init__(pop, nplots) # Initialize underlying plot object.
        
        self.__get_data(data_getter) # Populate the data attribute using the data_getter function.

        self.filepath.append('heatmap') # Assign plot to 'heatmap' directory.
        if nplots > 1: # Create the default filenames.
            self.filename = f'heatmap_grid.png'
        else:
            self.filename = f'heatmap.png'

    
    # Private methods -------------------------------------------------
    
    def __get_data(self, data_getter):
        '''
        Gets all data to be plotted on the heatmap, and stores it in the self.data parameter.
        After this function is called, self.data will be a dictionary where the keys are celltypes
        and values are 2-D numpy arrays.

        Parameters
        ----------
        data_getter : function
            
        '''
        self.data = {}
        
        if self.toplot == 'sp_s' and self.allsamples:
            self.data['allsamples'] = data_getter('allsamples')
        
        for i in range(self.nplots): # Get the data for each subplot.
            subplot = self.subplots[i]
            self.data[subplot] = data_getter(subplot, i)
                
        
    def __s_ct_get_data(self, celltype, index):
        '''
        Gets the data to be plotted on the heatmap for the given celltype. This function creates and
        returns a 2-D array with the rows corresponding to samples in self.samples and the columns corresponding
        to genes in self.genes. 
        
        Parameters
        ----------
        celltype : str
            The celltype for which to collect the data.
        '''
        t0 = time.time() # Get the start time for performance info.
        print(f'Gathering {celltype} data...\r', end='')
        
        nans = [] # List to store the indices to be poplulated with NaNs.

        genes, samples = self.xlabels[index], self.ylabels[index]
        ngenes, nsamples = len(genes), len(samples)

        # Initialize a 2-D array, where there is a row for each sample and a column for each gene.
        data = np.zeros((nsamples, ngenes))
        for i in range(nsamples):
            sample = samples[i]
            
            for j in range(ngenes):
                gene = genes[j]
                ncells = np.count_nonzero(self.pop['samples'][sample]['cell_type'] == celltype)
                mincells = 50 # The minumum number of cells needed to make a valid distribution
                
                # NOTE: If I can figure out how to get the skl.metrics.pairwise_distances() function to accept
                # NaNs without throwing an error (force_all_finite=False is not working), then I will change this
                # to directly set data[i, j] = np.nan.
                if ncells < mincells: # 
                    nans.append((i, j))
                    
                # Create an expression distribution for a particular gene, sample, and celltype.
                dist = barplot.BarPlot(self.pop, gene, celltype, sample, nbins=15)
                l1 = dist.calculate_l1()[sample] # Get the L1 metric for the distribution (reference is control by default).
                data[i, j] = l1
        
        t1 = time.time()
        print(f'{celltype} data gathered: {t1 - t0} seconds')

        if self.cluster: # If clustering is set to True...
            t0 = time.time()
            print(f'Clustering {celltype} data...\r', end='')

            axis = self.clusterinfo.get('axis', 'both') 
            if axis == 'x':
                data, genes = self.__cluster(data, index, axis='x')
            elif axis == 'y':
                data, samples = self.__cluster(data, index, axis='y')
            elif axis == 'both':
                data, genes  = self.__cluster(data, index, axis='x')
                data, samples = self.__cluster(data, index, axis='y')
            # Change the axes labels stored in the object to match the new ordering.
            self.xlabels[index] = genes
            self.ylabels[index] = samples

            t1 = time.time()
            print(f'{celltype} data clustered: {t1 - t0} seconds')

        # For the datapoints corresponding to distributions with ncells < mincells, replace the current value with NaN.
        for i, j in nans:
            data[i, j] = np.nan

        return data


    def __sp_s_get_data(self, sample, index): # IN PROGRESS
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

        genes = self.xlabels[index].tolist()
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

    def __cluster(self, data, index, axis=None):
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
            labels = self.xlabels[index]
        elif axis == 'y':
            matrix = data
            labels = self.ylabels[index]
            
        # Retrieve cluster settings.
        metric = self.clusterinfo.get('metric', 'euclidean') # Get the distance metric, euclidian by default.
        linkage = self.clusterinfo.get('linkage', 'complete') # Get the linkage type, complete by default.
        nclusters = self.clusterinfo.get('nclusters', 3) # Get the number of clusters, 3 by default.

        # NOTE: This function returns a  matrix such that element {i, j} is the distance between 
        # the ith and jth vectors of the given matrix. 
        dists = skl.metrics.pairwise_distances(X=matrix, metric=metric) # Get the distance matrix.
        model = skl.cluster.AgglomerativeClustering(n_clusters=nclusters,
                                                    affinity='precomputed', # Distance matrix was precomputed.
                                                    linkage=linkage) # Create the clustering model.

        lm = sch.linkage(dists, method=linkage) # Get the linkage matrix.
        # Store the linkage matrices for later use in plotting dendrograms.
        if axis == 'x':
            self.lms[index][0] = lm
        elif axis == 'y':
            self.lms[index][1] = lm

        clusteridxs = model.fit_predict(X=dists) # Return the clustering labels.
        sorter = get_sorter(clusteridxs) # Get the indice with which to sort the data and data labels.
        
        labels = labels[sorter] # Sort the labels to match the data.
        if axis == 'y':
            data = data[sorter, :] # Sort the data.
        elif axis == 'x':
            data = data[:, sorter] # Sort the data

        return data, labels

    def __plot_dendrogram(self, daxes, index):
        '''
        '''
        xdax, ydax = daxes # Get the axes for the x and y-axis dendrograms.
        lms = self.lms[index] # Get the linkage matrices for the selected subplot.

        if lms[0] is not None: # If the x-axis was clustered...
            lm = lms[0] # Retrieve the x-axis linkage matrix.
            xdax.set_yticks([])
            xdax.set_xticks([])
            sch.dendrogram(lm, ax=xdax, 
                           orientation='top', 
                           color_threshold=0,
                           above_threshold_color='black',
                           no_labels=True)
        if lms[1] is not None: # If the y-axis was clustered...
            lm = lms[1] # Retrieve the y-axis linkage matrix.
            ydax.set_yticks([])
            ydax.set_xticks([])
            sch.dendrogram(lm, ax=ydax, 
                           orientation='left',
                           color_threshold=0,
                           above_threshold_color='black',
                           no_labels=True)

    def __plotter(self, axes, index, color):
        '''
        Plots a heatmap on the inputted axes.        
        
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes of the subplot on which the graph will be plotted.
        index : 
            The index of the subplot to be plotted.
        color: str
            The name of a matplotlib.colors.Colormap to color the heatmap data. 
        '''
        assert isinstance(color, str), 'Color must be a string for a HeatMap object.'
        
        axes.axis('off') # Turn off the axes frame.

        # Create a layout for all the subcomponents of the heatmap...
        nr, nc = self.nrows, self.ncols
        sub = (index // nr, index % nc) # Get the coordinates for the subplot in the figure grid.
        w, h = 1.0 / nc, 1.0 / nr # Get the width of a subplot column and height of a subplot row.
        x0, y0 = (sub[0] * w, h * (nr - sub[1] - 1)) # Get the absolute coordinates of the origin of the subplot. 

        plot_dendrograms = self.clusterinfo.get('plot_dendrograms', True)
        # If cluster=True and plot_dendrograms is set to True (or left blank), plot the dendrograms.
        if self.cluster and plot_dendrograms:
            d = 0.075 * w # The thickness of the dendrogram plots.
            mainw, mainh = 0.7 * w, 0.8 * h # The dimensions of the main axes.
            
            # Create axes for the x-and-y dendrograms.
            ydax = self.figure.add_axes([x0, y0, d, mainh], frame_on=False)
            xdax = self.figure.add_axes([x0 + d, y0 + mainh, mainw, d], frame_on=False)
            self.__plot_dendrogram((xdax, ydax), index) # Plot the dendrograms.
        else:
            d = 0
            mainw, mainh = 0.775 * w, 0.875 * h
           
        # Create axes for the main heatmap and the colorbar.
        mainax = self.figure.add_axes([x0 + d, y0, mainw, mainh], frame_on=False)
        cax = self.figure.add_axes([x0 + 0.95 * w, y0 + 0.1 * h, 0.05 * w , 0.6 * h], frame_on=False)
           
        # Retrieve the data and data labels for the subplot.
        subplot = self.subplots[index]
        data = self.data[subplot]
        xlabels, ylabels = self.xlabels[index], self.ylabels[index]  
        
        # Plot the heatmap on the main axes.
        cmap = plt.get_cmap(color) # Get the colormap.
        cmap.set_bad(color='gray') # Set NaN values to be gray.
        mainax.imshow(data, cmap=cmap, aspect='auto') # Auto scales the pixels according to the axes. 
        # Set the axes ticks. 
        mainax.set_xticks(np.arange(0, len(xlabels))) 
        mainax.set_yticks(np.arange(0, len(ylabels))) 
        # Set the axes labels, with the correct orientation and font size.
        mainax.set_yticklabels(ylabels, fontdict={'fontsize':20})
        xlabels = mainax.set_xticklabels(xlabels, fontdict={'fontsize':20})
        for label in xlabels: # Make x-axis labels vertical.
            label.set_rotation('vertical')
        # If dendrograms are plotted, move the y-tick labels to make room for the dendrograms.
        if self.cluster and plot_dendrograms: 
            mainax.yaxis.set_label_position('right')
            mainax.yaxis.tick_right() 
       
        # Add a colorbar to the colorbar axes.
        mappable = mpl.cm.ScalarMappable(cmap=cmap) # Turn the selected colormap into a ScalarMappable object.
        cbar = plt.colorbar(mappable, cax=cax, ticks=[0, 0.5, 1])
        cbar.ax.set_title('L1 norm', fontdict={'fontsize':20})
        cbar.ax.set_yticklabels(['-2', '0', '2'], fontdict={'fontsize':20})

    # Public methods --------------------------------------------------

    def plot(self, color='bwr'):
        '''
        Uses the data to generate graphs.
        
        Parameters
        ----------
        color : str
            The name of the colormap with which to create the heatmap image.
            See matplotlib.colors module documentation for information on possible colors. 
        '''
        self._plot(self.__plotter, color)


# Accessory functions --------------------------------------------------------------------------------------

def get_sorter(clusteridxs):
    '''

    '''
    nelements = len(clusteridxs) # Get the number of elements to sort.
    idxs = list(range(nelements))
    sorter = [idx for (_, idx) in sorted(zip(clusteridxs, idxs))]
    
    return sorter


