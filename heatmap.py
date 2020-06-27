import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import sklearn as skl
import time

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
            self.xaxis = np.array([genelist] * nplots)
            self.yaxis = np.array([self.samples] * nplots)
            
            data_getter = self.__s_ct_get_data # Set the data retrieval function.
        
        elif toplot == 'sp_s':
            if self.allsamples:
                self.subplots = ['allsamples']
                nplots = 1
            else:
                self.subplots = self.samples
                nplots = len(self.subplots)
            
            self.xaxis = np.ndarray([[genelist], ] * nplots)
            self.yaxis = None # This will be a list of lists.
            
            data_getter = self.__sp_s_get_data # Set the data retrieval function.
        
        elif toplot == 'sp_rp':
            self.subplots = self.refpops
            nplots = len(self.subplots)

            self.xaxis = np.ndarray([[genelist], ] * nplots)
            self.yaxis = None # This will be a list of lists.
            
            data_getter = self.__sp_rp_get_data # Set the data retrieval function.
       
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
        
        genes, samples = self.xaxis[index], self.yaxis[index]
        ngenes, nsamples = len(genes), len(samples)

        # Initialize a 2-D array, where there is a row for each sample and a column for each gene.
        data = np.zeros((nsamples, ngenes))
        for i in range(nsamples):
            sample = samples[i]
            
            for j in range(ngenes):
                gene = genes[j]
                
                # Create an expression distribution for a particular gene, sample, and celltype.
                dist = barplot.BarPlot(self.pop, gene, celltype, sample, nbins=25)
                l1 = dist.calculate_l1()[sample] # Get the L1 metric for the distribution (reference is control by default).
                data[i, j] = l1
        
        t1 = time.time()
        print(f'{celltype} data gathered: {t1 - t0}')

        if self.cluster: # If clustering is set to True...
            t0 = time.time()
            print(f'Clustering {celltype} data...\r', end='')

            axis = self.clusterinfo.get('axis', 'y')
            clusteridxs = self.__cluster(data, axis=axis)
            sorter = get_sorter(clusteridxs)
            
            if axis == 'y':
                data = data[sorter, :]
                self.yaxis[index] = samples[sorter]

            elif axis == 'x':
                clusteridxs = self.__cluster(data, axis='x')
                data = data[:, sorter]
                self.xaxis[index] = genes[sorter]
            
            t1 = time.time()
            print(f'{celltype} data clustered: {t1 - t0}')

        return data


    def __sp_s_get_data(self, sample, index):
        '''
        '''
        genes = self.xaxis[index].tolist()
        geneidxs = [self.genedict[gene] for gene in genes]

        if sample == 'allsamples':
            gmm, gmmtypes= self.pop['gmm'], self.pop['gmm_types']
            genedata = PA.cat_data(self.pop, 'M_norm')
            featdata = PA.cat_data(self.pop, 'C')
            gmmtypes = self.pop['gmm_types']
        else:
            gmm, gmmtypes = self.pop['samples'][sample]['gmm'], self.pop['samples'][sample]['gmm_types']
            genedata = self.pop['samples'][sample]['M_norm']
            featdata = self.pop['samples'][sample]['C']
        
        genedata = genedata[geneidxs, :]
        
        prediction = gmm.predict(featdata)

    def __sp_rp_get_data(self, refpop):
        '''
        '''
        pass

    def __cluster(self, matrix, axis='y'):
        '''
        Runs an agglomerative hierarchical clustering algorithm on the inputted matrix. It uses
        the cluster settings inputted at Heatmap initialization.

        Parameters
        ----------
        matrix : numpy.ndarray
            A 2-D array which contains Heatmap data.
        along : str
            One of 'rows' or 'cols'. If 'rows', the rows of the matrix are clustered; if 'cols', 
            the columns of the matrix are clustered.
        '''
        if axis == 'x': # If clustering is to be carried out by column...
            matrix = matrix.T # Get the transpose of the inputted matrix.
            
        # Retrieve cluster settings.
        metric = self.clusterinfo.get('metric', 'euclidean') # Get the distance metric, euclidian by default.
        linkage = self.clusterinfo.get('linkage', 'complete' ) # Get the linkage type, complete by default.
        nclusters = self.clusterinfo.get('nclusters', 3) # Get the number of clusters, 3 by default.

        # NOTE: This function returns a  matrix such that element {i, j} is the distance between 
        # the ith and jth vectors of the given matrix. 
        # NOTE: A distance matrix is a 2-D  array) containing the distances, taken pairwise, 
        # between the elements of a set. If there are N elements, this matrix will have size N×N.
        dists = skl.metrics.pairwise_distances(X=matrix, metric=metric) # Get the distance matrix.
        model = skl.cluster.AgglomerativeClustering(n_clusters=nclusters,
                                                    affinity='precomputed',
                                                    linkage=linkage) # Create the clustering model.
        # NOTE: Because we are inputting a distance matrix, affinity='precomputed'.
        # [https://stackoverflow.com/questions/44834944/agglomerative-clustering-in-sklearn] 
        return model.fit_predict(X=dists) # Return the clustering labels.

    def __plotter(self, axes, index, color):
        '''
        Plots a heatmap on the inputted axes.        
        
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to add the barplot. If None, new axes are created.
        index : 
            The index of the celltype to be plotted (corresponds to a celltype in self.celltypes).
        color: str
            The name of a matplotlib.colors.Colormap to color the heatmap data. 
        '''
        assert isinstance(color, str), 'Color must be a string for a HeatMap object.'
        
        subplot = self.subplots[index]
        data = self.data[subplot]
        xaxis, yaxis = self.xaxis[index], self.yaxis[index]  

        axes.imshow(data, cmap=plt.get_cmap(color), aspect='auto') # Auto scales the pixels according to the axes. 

        # Make the graph prettier!
        axes.set_xticks(np.arange(0, len(xaxis))) # Genes will be plotted along the x-axis (columns of self.data)
        axes.set_yticks(np.arange(0, len(yaxis))) # Genes will be plotted along the y-axis (rows of self.data)

        xlabels = axes.set_xticklabels(xaxis)
        for label in xlabels: # Make x-axis labels vertical.
            label.set_rotation('vertical')
        axes.set_yticklabels(yaxis)

        axes.set_title(f'Expression in {subplot}')

        # Add a colorbar.
        mappable = mpl.cm.ScalarMappable(cmap=plt.get_cmap(color))
        cbar = plt.colorbar(mappable, ax=axes, ticks=[0, 0.5, 1])
        
        cbar.ax.set_title('L1 norm')
        cbar.ax.set_yticklabels(['-2', '0', '2'])

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

