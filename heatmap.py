import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

from plot import plot
from plot import barplot

class HeatmapPlot(plot.Plot):
    '''

    '''
    # Class constructor.
    def __init__(self, pop, genes=None, toplot=None, samples=None, celltypes=None, refpops=None, cluster=False, **kwargs):
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
        self.clustersettings = kwargs # Save the cluster settings as an attribute.

        self.celltypes = plot._init_celltypes(pop, celltypes) # Initialize cell type list.
        
        self.allsamples = (samples is None) # If no samples are specified, all samples are used.
        self.samples = plot._init_samples(pop, samples) # Initialize list of samples.
        
        self.genes, self.geneidxs = plot._init_genes(pop, genes) # Initialize list of genes and indices.
        
        self.refpops = refpops

        options = ['s_ct', 'sp_s', 'sp_rp']
        assert toplot in options, f'The toplot parameter must be one of: {options}.'
        self.toplot = toplot

        if toplot == 's_ct':
            self.subplots = self.celltypes
            self.variable = self.samples
            data_getter = self.__s_ct_get_data # Set the data retrieval function.
        
        elif toplot == 'sp_s':
            self.subplots = self.samples
            self.variable = None # This will be a list of lists.
            data_getter = self.__sp_s_get_data # Set the data retrieval function.
        
        elif toplot == 'sp_rp':
            self.subplots = self.refpops
            self.variable = None # This will be a list of lists.
            data_getter = self.__sp_rp_get_data # Set the data retrieval function.
        
        if toplot == 'sp_s' and self.allsamples:
            nplots = 1 # If plotting subpops over samples, and no sample is specified, use the global gmm data.
        else:
            nplots = len(self.subplots)
        
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
        
        for element in self.subplots: # Get the data for each subplot.
                self.data[element] = data_getter(element)
                
        
    def __s_ct_get_data(self, celltype):
        '''
        Gets the data to be plotted on the heatmap for the given celltype. This function creates and
        returns a 2-D array with the rows corresponding to samples in self.samples and the columns corresponding
        to genes in self.genes. 
        
        Parameters
        ----------
        celltype : str
            The celltype for which to collect the data.
        '''
        ngenes, nsamples = len(self.genes), len(self.samples)
        # Initialize a 2-D array, where there is a row for each sample and a column for each gene.
        data = np.empty((nsamples, ngenes))
        for i in range(nsamples):
            sample = self.samples[i]
            
            for j in range(ngenes):
                gene = self.genes[j]
                
                # Create an expression distribution for a particular gene, sample, and celltype.
                dist = barplot.BarPlot(self.pop, gene, celltype, sample, nbins=25)
                l1 = dist.calculate_l1()[sample] # Get the L1 metric for the distribution (reference is control by default).
                data[i, j] = l1

                progress = int(100 * float((i * ngenes +  j) / (ngenes * nsamples)))
                print(f'Gathering {celltype} data: {progress}%\r', end='')
        
        print(f'Gathering {celltype} data: 100%')

        return data

    def __sp_s_get_data(self, sample):
        '''
        '''
        if sample == 'allsamples':
            gmm, gmmtypes= self.pop['gmm'], self.pop['gmm_types']
            genedata = PA.cat_data(self.pop, 'M_norm')
            featdata = PA.cat_data(self.pop, 'C')
            gmmtypes = self.pop['gmm_types']
        else:
            gmm, gmmtypes = self.pop['samples'][sample]['gmm'], self.pop['samples'][sample]['gmm_types']
            genedata = self.pop['samples'][sample]['M_norm']
            featdata = self.pop['samples'][sample]['C']
        
        genedata = genedata[self.geneidxs, :]
        
        prediction = gmm.predict(featdata)

    def __sp_rp_get_data(self, refpop):
        '''
        '''
        pass

    def __cluster(self, matrix):
        '''
        '''
        settings = self.clustersettings
        method = settings.get('metric', 'correlation')
        linkage = settings.get('linkage', 'complete' )

        return matrix

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
        if self.toplot in ['sp_rp', 'sp_s']:
            variable = self.variable[index]
        else:
            variable = self.variable

        axes.imshow(data, cmap=plt.get_cmap(color), aspect='auto') # Auto scales the pixels according to the axes. 

        # Make the graph prettier!
        axes.set_xticks(np.arange(0, len(self.genes))) # Genes will be plotted along the x-axis (columns of self.data)
        axes.set_yticks(np.arange(0, len(variable))) # Genes will be plotted along the y-axis (rows of self.data)

        xlabels = axes.set_xticklabels(self.genes)
        for label in xlabels: # Make x-axis labels vertical.
            label.set_rotation('vertical')
        axes.set_yticklabels(self.samples)

        axes.set_title(f'Expression in {subplot}')

        # Add a colorbar.
        mappable = mpl.cm.ScalarMappable(cmap=plt.get_cmap(color))
        cbar = self.figure.colorbar(mappable, ticks=[0, 0.5, 1])
        
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
