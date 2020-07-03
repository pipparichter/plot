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
    def __init__(self, pop, is_subplot=False, filename=None):
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
        self.filename = filename
        
        self.figure, self.axes = None, None
        self.is_subplot = is_subplot

        self.pop = pop # Store the pop object.
        
        # Gets the names of all the control samples in the pop object.
        self.ctrls = [s for s in pop['samples'].keys() if re.match(pop['controlstring'], s) is not None]

        self.plotted = False # Once self.plot() has been called once, this becomes True.
        self.plotter = None # The plotting function, which will be initialized in a derived class initializer.

        self.data = None # This will store the data for the Plot; format varies by subclass.

    # Private methods ----------------------------------------------------------------
    
    # NOTE: Removing this subprocess from __init__() reduces the computational cost of creating 
    # Plots (useful when creating BarPlots for use in a HeatmapPlot). 
    def __init_figure(self, axes):
        '''
        Initializes the backing figure and axes. This function is called only when the graph needs
        to be displayed.
        '''
        if self.is_subplot:
            self.axes = axes
            self.figure = axes.get_figure # Get the figure associated with the inputted axes.
        else:
            self.figure = plt.figure(figsize=(20, 20))
            self.axes = self.figure.add_axes([0, 0, 1, 1])

    def plot(self, color=None, fontsize=20, axes=None):
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
        assert not self.plotted, 'The Plot object has already been plotted.'
        
        self.__init_figure(axes=axes)

        if color is None: # If color is not specified, use the default color. 
            color = self.color

        self.plotter(self.axes, color=color, fontsize=fontsize)
        self.plotted = True

    # Public methods ----------------------------------------------------------------

    def save(self, filename=None):
        '''
        Saves a plot to a subdirectory.

        Parameters
        ----------
        filename : str
            The name under which to save the plot. If None, the default filename (self.filename) is used.
        '''
        assert self.plotted, 'A figure has not yet been created. The plot() method must be called.'
        assert not self.is_subplot, 'A subplot cannot be saved directly.'
        
        for i in range(1, len(self.filepath) + 1): # Make all necessary directories.
            filepath = os.path.join(*self.filepath[:i])
            PA.mkdir(filepath)
        
        if filename is None: # If None, use the default filename.
            filename = self.filename

        # NOTE: See this link [https://docs.python.org/2/tutorial/controlflow.html#unpacking-argument-lists]
        # for more information on the '*' operator.
        loc = os.path.join(*self.filepath, filename) # Combine the filepath into a valid path name.
        self.figure.savefig(loc, dpi=200, bbox_inches='tight') # Save the figure to the specified directory.

        print(f'Plot object saved to {loc}.')
   

# Accessory functions ---------------------------------------------------

def get_ncells(pop, sample=None, celltype=None):
    '''
    Returns the number of cells in the specified sample of the specified celltype.

    Parameters
    ----------
    sample : str
        The sample for which to retrieve the number of cells. If None, all samples are used.
    celltype : str
        The celltype for which to retrieve the number of cells. If None, all celltypes are used.
    '''
    if sample is None:
        if celltype is None:
            ncells = pop['ncells']
        else:
            ncells = 0
            for s in pop['samples'].keys():
                ncells += np.count_nonzero(pop['samples'][s]['cell_type'] == celltype)
    else:
        if celltype is None:
            ncells = len(pop['samples'][sample]['cell_type'])
        else:
            ncells = np.count_nonzero(pop['samples'][sample]['cell_type'] == celltype)

    return ncells


