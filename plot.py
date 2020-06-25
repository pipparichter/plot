import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

# NOTE: A preceding underscore denotes a protected method or attribute. This doesn't prevent them from being
# accessed, it just means they probably shouldn't be. If an attribute or method is preceded by a double 
# underscore, trying to access it will result in an AttributeError.
# [https://www.tutorialsteacher.com/python/private-and-protected-access-modifiers-in-python]

# Why does super().__init__() not throw an AttributeError?

class Plot():
    # Class constructor.
    def __init__(self, pop, grid=False, nplots=1):
        '''
        Initializes a Plot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        grid : bool
            Whether or not the plot is a grid.
        nplots : int
            If the object is a grid, this specifies the number of plots in the 
            grid. If grid=False, nplots must be None.
        '''
        self.filepath = [pop['output'], 'plots'] # Initialize a filepath.
        
        self.pop = pop # Store the pop object.
        
        self.grid = grid
        self.nplots = nplots
        self.figure, self.axes = None, None # these attributes will store the figure and axes objects.
        self.nrows, self.ncols = PA.nr_nc(nplots)
                
        # Gets the names of all the control samples in the pop object.
        self.ctrls = [s for s in pop['samples'].keys() if re.match(pop['controlstring'], s) is not None]

        self.plotted = False # Once self.plot() has been called once, this becomes True.
        
        self.data = None # This will store the data for the Plot; format varies by subclass.

    # Private methods ----------------------------------------------------------------
    
    # NOTE: Removing this subprocess from __init__() reduces the computational cost of creating 
    # Plots (useful when creating BarPlots for use in a HeatmapPlot). 
    def __init_figure(self):
        '''
        Initializes the backing figure and axes. This function is called only when the graph needs
        to be displayed.
        '''
        if self.grid:
            self.figure, self.axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, 
                    sharex=True, sharey=True, figsize=(30, 30))
            self.axes = self.axes.flatten().tolist()
        elif not self.grid:
            self.figure = plt.figure(figsize=(20, 20))
            self.axes = [plt.axes([0, 0, 1, 1])] # Creates a list of axes with length of one.
        
    def _plot(self, plotter, color):
        '''
        Graphs the Plot object on the axes.

        Parameters
        ----------
        plotter : function
            The function to use to plot the data.
        color : variable
            The color data for the plotter function. The format of this data varies by subclass.
        '''
        assert self.data is not None, 'Data has not been initialized.'
        
        if self.plotted: # If plot() has been called before, clear the existing figure.
            self.figure.clear()
        self.__init_figure() # Initialize the figure and axes.

        for i in range(self.nplots):
            plotter(self.axes[i], i, color)
            # self.axes[i] = self.__barplot(ax, sample, color1, color2)
        
        self.figure.tight_layout() # Make sure axes aren't cut off.
        self.plotted = True

    # Public methods ----------------------------------------------------------------

    def save(self):
        '''
        Saves a plot to a subdirectory.
        '''
        assert self.plotted, 'A figure has not yet been created. The plot() method must be called.'

        for i in range(1, len(self.filepath)): # Make all necessary directories.
            filepath = os.path.join(*self.filepath[:i])
            PA.mkdir(filepath)
        
        # NOTE: See this link [https://docs.python.org/2/tutorial/controlflow.html#unpacking-argument-lists]
        # for more information on the '*' operator.
        filepath = os.path.join(*self.filepath) # Combine the filepath into a valid path name.
        self.figure.savefig(filepath, dpi=200) # Save the figure to the specified directory.

        print(f'Plot object saved to {filepath}.')
   

# Accessory functions ---------------------------------------------------------------

def _init_samples(pop, samples):
    '''
    Generates a list of samples for use in a Plot subclass.
    
    Parameters
    ----------
    pop : dict
        The pop object.
    samples : str, list
        The sample or list of samples to be analyzed. If None, all samples in the pop
        object are analyzed.
    '''
    if samples is None: # By default, make a plot for every sample.
        samples = list(pop['samples'].keys())
    elif isinstance(samples, str): # If only a single sample is given, put it into a list.
        samples = [samples]
    
    samples = [s for s in samples if re.match(pop['controlstring'], s) is None]  # Filter out controls.
    
    for sample in samples: # Make sure all samples are valid.
        assert sample in list(pop['samples'].keys()), f'Sample name {sample} is invalid'

    return samples


def _init_celltypes(pop, celltypes):
    '''
    Generates a list of samples for use in a Plot subclass.
    
    Parameters
    ----------
    pop : dict
        The pop object.
    celltypes : str, list
        The celltype or list of celltypes to be analyzed. If None, all samples in the pop
        object are analyzed.
    '''
    if celltypes is None: # By default, make a plot for every celltype.
        celltypes = list(set(pop['gmm_types'])) # Remove duplicates from gmm_types list.
    elif isinstance(celltypes, str): # If only a single sample is given, put it into a list.
        celltypes = [celltypes]
    
    for celltype in celltypes: # Make sure all samples are valid.
        assert celltype in pop['gmm_types'], f'Cell type {celltype} is invalid'

    return celltypes


def _init_genes(pop, genes):
    '''
    Filters out all invalid gene names (i.e. genes not in pop['filtered_genes'], and returns the filtered list.
    It also prints out all genes that were removed.

    Parameters
    ----------
    pop : dict
        The pop object.
    genes : list
        The list of genes to filter.
    '''
    invalid = []
    for gene in genes[:]: # Filter out invalid gene names.
        if gene not in pop['filtered_genes']:
            invalid.append(gene)
            genes.remove(gene)
    if len(invalid) > 0: # If any genes were removed, print the result
        print('The following gene names are invalid, and were removed: ' + ', '.join(invalid))    
    
    return genes

