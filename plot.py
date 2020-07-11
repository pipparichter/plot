import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy
import barplot

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

# NOTE: A preceding underscore denotes a protected method or attribute. This doesn't prevent them from being
# accessed, it just means they probably shouldn't be. If an attribute or method is preceded by a double 
# underscore, trying to access it will result in an AttributeError.
# [https://www.tutorialsteacher.com/python/private-and-protected-access-modifiers-in-python]

# Why does super().__init__() not throw an AttributeError?

class Plot():
    '''
    '''
    def __init__(self, pop, 
                 is_subplot=False, 
                 filename=None):
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

    # NOTE: Removing this subprocess from __init__() reduces the computational cost of creating 
    # Plots (useful when creating BarPlots for use in a HeatmapPlot). 
    def __init_figure(self, axes=None):
        '''
        Initializes the backing figure and axes. This function is called only when the graph needs
        to be displayed.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            If the Plot object is_subplot, this is the subplot axes to which the Plot is assigned. 
        '''
        if self.is_subplot:
            self.axes = axes
            self.figure = axes.get_figure() # Get the figure associated with the inputted axes.
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
        axes : matplotlib.axes.Axes
            If plotting a subplot, this is the axes of the subplot. This parameter should not be 
            specified if not plotting a subplot.
        '''
        assert self.data is not None, 'Data has not been initialized.'
        if axes is not None:
            assert self.is_subplot == True, 'If a plotting axes is specified, the Plot must be a subplot.'

        if self.plotted:
            self.axes.clear()

        self.__init_figure(axes=axes)

        if color is None: # If color is not specified, use the default color. 
            color = self.color

        self.plotter(self.axes, color=color, fontsize=fontsize)
        self.plotted = True

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
        The sample for which to retrieve the number of cells. 
    celltype : str
        The celltype for which to retrieve the number of cells. If None, the total number of cells in 
        the sample is returned.
    '''
    assert sample is not None, 'A sample name must be specified.'

    celltypes = np.array(pop['samples'][sample]['cell_type'])
    
    if celltype is None:
        ncells = len(celltypes)
    else:
        ncells = np.count_nonzero(celltypes == celltype)

    return ncells

# Parameter checkers ---------------------------------------------------------------

def check_celltype(pop, celltype):
    '''
    '''
    assert celltype is not None, 'A cell type must be specified.'
    assert celltype in pop['gmm_types'], f'{celltype} is not a valid celltype.'
    
    return celltype


def check_gene(pop, gene):
    '''
    '''
    assert gene is not None, 'A gene name must be specified.'
    assert gene in pop['filtered_genes'], f'{gene} is an invalid gene name.'
    
    return gene


def check_genes(pop, genes):
    '''
    Checks a list of genes to make sure all are valid, and returns a numpy array of valid genes.
    Invalid genes are removed and printed.

    Parameters
    ----------
    genes : list
        A list of genes to check.
    '''
    assert genes is not None, 'A gene list must be specified.'
    invalid = []
    for gene in genes[:]:
        if gene not in pop['filtered_genes']:
            genes.remove(gene)
            invalid.append(gene)
    
    if len(invalid) > 1:
        print('The following genes are invalid and were removed: ' + ', '.join(invalid))    
    assert len(genes) > 1, 'At least one valid gene name must be given.'

    return np.array(genes)


def check_sample(pop, sample):
    '''
    '''
    assert sample is not None, 'A sample name must be specified.'
    assert sample in pop['samples'].keys(), f'{sample} is an invalid sample name.'

    return sample


def check_samples(pop, samples, filter_reps=True, filter_ctrls=True):
    '''
    '''
    samples = list(samples) # Make sure samples is a list.
    assert samples is not None, 'A list of samples must be specified.'
    for sample in samples[:]:
        check_sample(pop, sample)

        if filter_ctrls and re.match(pop['controlstring'], sample) is not None:
            samples.remove(sample)
        elif filter_reps and re.match('.*_rep', sample) is not None:
            samples.remove(sample)
    
    return np.array(samples)
        

# Differentially expressed gene selection ------------------------------------------------------------

def diffexp_(pop, merge_samples=True, tail=0.01):
    '''

    '''
    celltypes = np.unique(pop['gmm_types']).tolist()
    genes = pop['filtered_genes']
    samples = check_samples(pop['order'], pop['order'], filter_ctrls=True, merge_reps=merge_samples)
    ctrls = [s for s in pop['order'] if re.match(pop['controlstring'], s) is not None]
    
    diffexp = {}
    diffexp['upreg'] = np.array([])
    diffexp['downreg'] = np.array([])

    print(f'Calculating cutoff...    \r', end='')
    ctrl_l1s = np.array([])
    for ctrl in ctrls:
        # Turn off merge_samples when evaluating the controls. 
        for celltype in celltypes:
            for gene in genes:
                params = {'gene':gene, 'sample':ctrl, 'merge_samples':False, 'celltype':celltype}
                bar = barplot.BarPlot(pop, type_='g_s_ct', **params) 
                l1 = bar.calculate_l1()
                ctrl_l1s = np.append(ctrl_l1s, l1)
    distribution = scipy.stats.rv_histogram(np.histogram(ctrl_l1s, bins=100))
    cutoff = abs(distribution.ppf(tail)) # Sometimes this value is negative, so take the absolute value.
    print(f'Cutoff is {cutoff}.    ')
    
    for sample in samples:
        for celltype in celltypes: 
            for gene in genes:
                params = {'gene':gene, 'sample':sample, 'merge_samples':merge_samples, 'celltype':celltype}
                bar = barplot.BarPlot(pop, type_='g_s_ct', **params) 
                l1 = bar.calculate_l1()
                
                if l1 < -1 * cutoff:
                    diffexp['downreg'] = np.append(diffexp['downreg'], gene)
                elif l1 > cutoff:
                    diffexp['upreg'] = np.append(diffexp['upreg'], gene)
    return diffexp


