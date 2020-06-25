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
    # Class constructor.
    def __init__(self, pop, genes, celltypes, samples=None, grid=False):
        '''
        Initializes a HeatmapPlot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        genes : str, list
            A string or list of strings representing the official names of the genes to 
            be analyzed. If isinstance(genes, list), then grid must be True.
        samples : str, list
            A string or list of strings representing the samples to be analyzed. If None,
            all samples are used in the HeatmapPlot.
        celltypes : str, list
            A string or list of strings representing the celltypes to be analyzed. If None,
            all celltypes are used in the HeatmapPlot.
        grid : bool
            Whether or not the Plot is a grid. If grid=True, more than one gene must be given.
        '''
        assert isinstance(celltypes, str), 'Right now, you can only plot one celltype at a time.'
        self.celltypes = plot._init_celltypes(pop, celltypes)
        self.samples = plot._init_samples(pop, samples)
        self.genes = plot._init_genes(pop, genes)
        
        self.nsamples, self.ngenes = len(self.samples), len(self.genes)
        
        super().__init__(pop, grid=False, nplots=1)
        
        self.filepath.append('heatmaps')
        if grid:
            filename = f'heatmap_grid.png'
        else:
            self.filepath.append(self.celltypes[0])
            filename = f'{self.celltypes[0]}_heatmap.png'
        self.filepath.append(filename)

        self.__get_data() # Populate the data attribute.

    # Private methods -------------------------------------------------
    
    def __get_data(self):
        '''
        Gets all data to be plotted on the heatmap, and stores it in the self.data parameter.
        After this function is called, self.data will be a dictionary where the keys are celltypes
        and values are 2-D numpy arrays.
        '''
        self.data = {}
        for celltype in self.celltypes: # Get the data for each celltype in self.celltypes.
            self.data[celltype] = self.__get_celltype_data(celltype)
            print(f'Gathering {celltype} data: 100%')
    
    def __get_celltype_data(self, celltype):
        '''
        Gets the data to be plotted on the heatmap for the given celltype. This function creates and
        returns a 2-D array with the rows corresponding to samples in self.samples and the columns corresponding
        to genes in self.genes. 
        
        Parameters
        ----------
        celltype : str
            The celltype for which to collect the data.
        '''
        # Initialize a 2-D array, where there is a row for each sample and a column for each gene.
        data = np.empty((self.nsamples, self.ngenes))
        for i in range(self.nsamples):
            sample = self.samples[i]
            
            for j in range(self.ngenes):
                gene = self.genes[j]
                
                # Create an expression distribution for a particular gene, sample, and celltype.
                dist = barplot.BarPlot(self.pop, gene, celltype, sample, grid=False, nbins=25)
                l1 = dist.calculate_l1()[sample] # Get the L1 metric for the distribution (reference is control by default).
                data[i, j] = l1

                progress = int(100 * float((i * self.ngenes +  j) / (self.ngenes * self.nsamples)))
                print(f'Gathering {celltype} data: {progress}%\r', end='')
        
        return data

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
        celltype = self.celltypes[index]
        data = (self.data[celltype] + 2) / 2 # Normalize the data; L1 values range between -2 and 2.

        # Data has been normalized to the range [0, 1], so can be mapped directly to a matplotlib.colors.Colormap.
        axes.imshow(data, 
                cmap=plt.get_cmap(color),
                aspect='auto') # Keeps the axes lengths constant and scales the pixels accordingly. 

        # Make the graph prettier!
        axes.set_xticks(np.arange(0, self.ngenes)) # Genes will be plotted along the x-axis (columns of self.data)
        axes.set_yticks(np.arange(0, self.nsamples)) # Genes will be plotted along the y-axis (rows of self.data)

        xlabels = axes.set_xticklabels(self.genes)
        for label in xlabels: # Make x-axis labels vertical.
            label.set_rotation('vertical')
        axes.set_yticklabels(self.samples)

        axes.set_title(f'Expression in {celltype}')

        # Add a colorbar.
        mappable = mpl.cm.ScalarMappable(cmap=plt.get_cmap(color))
        kw = {'label':'L1 norm', 'ticks':['-2', '-1', '0', '1', '2']}
        cbar = self.figure.colorbar(mappable, ax=axes, label='L1 norm', ticks=[-1, 0, 1])
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
