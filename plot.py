import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy
import time 
from plotpop import barplot
# import sklearn as skl
import pandas as pd
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
        is_subplot : bool
            Whether or not the plot is a subplot.
        filename : str
            The filename under which to save the Plot.
        '''
        self.filepath = [pop['output'], 'plotpop'] # Initialize a filepath.
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

    def plot(self, color=None, fontsize={}, axes=None, flip_axes=False):
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
        fontsize : dict
            Stores the font information. It allows variable setting of the x and y-axis font sizes,
            as well as the title.

        '''
        assert self.data is not None, 'Data has not been initialized.'
        if axes is not None:
            assert self.is_subplot == True, 'If a plotting axes is specified, the Plot must be a subplot.'

        if self.plotted:
            self.axes.clear()

        self.__init_figure(axes=axes)

        if color is None: # If color is not specified, use the default color. 
            color = self.color

        self.plotter(self.axes, color=color, fontsize=fontsize, flip_axes=flip_axes)
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

def get_ncells(pop, sample=None, celltype=None, refpop=None):
    '''
    Returns the number of cells in the pop object.

    Parameters
    ----------
    sample : str
        The sample for which to retrieve the number of cells. 
    celltype : str
        The celltype for which to retrieve the number of cells. If None, the total number of cells in 
        the sample is returned.
    refpop : int
        The index of the reference population for which to retrieve the number of cells.
    '''
    assert bool(celltype) ^ bool(refpop) or not (refpop or celltype), \
            'Only one of celltype or refpop can be specified, not both.'

    ncells = 0
    if sample is None:
        if celltype is None and refpop is None:
            ncells = pop['ncells']
        elif refpop is None:
            for sample in pop['order']:
                celltypes = np.array(pop['samples'][sample]['cell_type'])
                ncells += np.count_nonzero(celltypes == celltype)
        else: # If celltype is None...
            print('CODE INCOMPLETE IN plot.get_ncells()')
            pass

    else: # If a sample is specified...
        if celltype is None and refpop is None:
            ncells = pop['samples'][sample]['M_norm'].shape[1]
        elif refpop is None:
            celltypes = np.array(pop['samples'][sample]['cell_type'])
            ncells = np.count_nonzero(celltypes == celltype)
        else: # If celltype is None...
            print('CODE INCOMPLETE IN plot.get_ncells()')
            pass

    return ncells

# Parameter checkers ---------------------------------------------------------------

def check_celltype(pop, celltype):
    '''
    Checks to make sure the inputted celltype is valid. If the celltype is valid, it is returned.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    celltype : str
        A string representing a possible celltype (e.g. T cell, B-cell, Myeloid).
    '''
    assert celltype is not None, 'A cell type must be specified.'
    assert celltype in pop['gmm_types'], f'{celltype} is not a valid celltype.'
    
    return celltype


def check_gene(pop, gene):
    '''
    Checks to make sure the inputted gene is valid. If the gene is valid, then it is returned.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    gene : str
        A string representing the gene name to be checked.        
    '''
    assert gene is not None, 'A gene name must be specified.'
    assert gene in pop['filtered_genes'], f'{gene} is an invalid gene name.'
    
    return gene


def check_sample(pop, sample):
    '''
    Checks the inputted sample to make sure it's a valid sample name. If the sample is valid, it
    is returned. 

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    sample : str
        A string representing a possible sample name.
    '''
    assert sample is not None, 'A sample name must be specified.'
    assert sample in pop['samples'].keys(), f'{sample} is an invalid sample name.'

    return sample


def check_genes(pop, genes):
    '''
    Checks a list of genes to make sure all are valid, and returns a numpy array of valid genes.
    Invalid genes are removed and printed.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
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


def check_samples(pop, samples, filter_reps=True, filter_ctrls=True):
    '''
    Checks a list of samples of samples to make sure all are valid. If valid, returns
    a numpy array containing the inputted sample names.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    samples : list
        A list of strings representing possible sample names.
    filter_reps : bool
        Whether or not to filter out replicate sample names (i.e. samples ending in '_rep'). 
        True by default.
    filter_ctrls : bool
        Whether or not to filter out samples matching the controlstring (stored in pop['controlstring']). 
        True by default.
    '''
    samples = list(samples) # Make sure samples is a list.
    for sample in samples[:]:
        check_sample(pop, sample)

        if filter_ctrls and re.match(pop['controlstring'], sample) is not None:
            samples.remove(sample)
        elif filter_reps and re.match('.*_rep', sample) is not None:
            samples.remove(sample)
    
    return np.array(samples)
        

# Differentially expressed gene selection ------------------------------------------------------------

def get_diffexp_data(pop, **kwargs):
    '''
    Returns a dictionary storing information on differentially expressed genes. It stores a list of the
    up and down-regulated genes by sample, as well as an nsamples by ngenes 2-D array containing the
    calculated L1 values. If cluster is True, it also stores a list of gene clusters (genes which behave
    similarly across all samples). 

    This function is designed to be used in conjunction with HeatmapPlots. 

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    '''
    genes = np.array(pop['filtered_genes'])
    samples = check_samples(pop, pop['order'], filter_ctrls=True, filter_reps=True)
    ctrls = [s for s in pop['order'] if re.match(pop['controlstring'], s) is not None]
    diffexp_data = {}
    diffexp_data['samples'] = {}
    
    assert len(kwargs) < 2, 'Only one filter can be specified.'
    for key, value in enumerate(kwargs): 
        # Store the filter used in diffexp. If there is no filter applied, there should be no filter field.
        diffexp_data['filter'] = (str(key), str(value))

    print(f'Calculating cutoff...    \r', end='')
    ctrl_l1s = np.array([])
    for ctrl in ctrls:
        # Turn off merge_samples when evaluating the controls. 
        for gene in genes:
            l1 = barplot.calculate_l1(pop, gene, ctrl, **kwargs)
            ctrl_l1s = np.append(ctrl_l1s, l1)
    distribution = scipy.stats.rv_histogram(np.histogram(ctrl_l1s, bins=100))
    cutoff = abs(distribution.ppf(0.001)) # Sometimes this value is negative, so take the absolute value.
    print(f'Cutoff is {cutoff}.    ')
    
    t0 = time.time()
    diff_genes, diff_l1s = np.array([]), np.array([]) 
    all_l1s = np.zeros((len(samples), len(genes))) # A 2-D array to store the L1 values.
    for i in range(len(samples)):
        print(f'Gathering differential expression data for sample {i} of {len(samples)}...    \r', end='')
        sample = samples[i]

        l1s = np.array([])
        for gene in genes:
            l1 = barplot.calculate_l1(pop, gene, sample, **kwargs)
            l1s = np.append(l1s, l1)
        all_l1s[i] = l1s
        
        downidxs = np.where(l1s < -1 * cutoff)[0]
        upidxs = np.where(l1s > cutoff)[0]

        diffexp_data['samples'][sample] = {}
        diffexp_data['samples'][sample]['up'] = genes[upidxs]
        diffexp_data['samples'][sample]['down'] = genes[downidxs]

        # Add the new up and down-regulated genes to array of differentially-expressed genes.
        # Also add the corresponding L1 values for the given sample, making sure the order of the 
        # two arrays corresponds.
        diff_genes = np.concatenate((diff_genes, genes[upidxs], genes[downidxs]))
        diff_l1s = np.concatenate((diff_l1s, l1s[upidxs], l1s[downidxs])) 

    t1 = time.time()
    print(f'All differential expression data gathered: {int(t1 - t0)} seconds.    ')
    
    idxs = np.argsort(diff_l1s) # Get the sorting indices to order by L1 value.
    diff_genes = diff_genes[idxs] # Order diff_genes by L1 value.
    
    # Make sure to preserve the order of diff_genes when removing duplicates (np.unique()) will 
    # automatically sort the array, so account for that).
    _, order = np.unique(diff_genes, return_index=True)
    diff_genes = diff_genes[np.sort(order)]

    diffexp_data['l1s'] = all_l1s # Store the calculated L1 values.
    diffexp_data['all'] = diff_genes
    
    return diffexp_data
    

def save_diffexp_data(pop, diffexp_data, cluster_data=None, dirname='diffexp'):
    '''
    Saves a diffexp_data dictionary, created by the get_diffexp_data function, to a subdirectory of 
    out/plotpop. The diffexp_data is organized into four files: one containing upregulated genes by sample,
    one containing downregulated genes by sample, one storing the matrix of all L1 values, and one storing
    all differentially-expressed genes.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    diffexp_data : dict
        The object containing all diffexp data. 
    cluster_data : list
        A list with nclusters sub-lists, which store the gene names by cluster (given by hmap.clusters).
        If this argument is specified, the cluster indices are included in the 'all.csv' file. 
    dirname : str
        The name under which to store the diffexp_data files; pop['output'/diffexp/[DIRNAME].
    '''
    samples = check_samples(pop, pop['order'], filter_ctrls=True, filter_reps=True)
    genes = pop['filtered_genes']

    # Make the directory in which to store the diffexp data in the location specified
    # by the path argument. This is the home directory by default.

    loc = os.path.join(pop['output'], 'plotpop', 'diffexp', dirname)
    PA.mkdir(loc)
    
    # Save the differentially-expressed genes.
    all_loc = os.path.join(loc, 'all_csv')
    all_df = pd.DataFrame(data={'genes':diffexp_data['all']})
    if cluster_data: # If clustering data is included, add it to the dataframe.
        assert len(cluster_data) == len(diffexp_data['all']), \
                'The cluster data does not align with the number of differentially-expressed genes.'
        all_df['cluster'] = cluster_data
    all_df.to_csv(all_loc) # Save the dataframe.

    # Save the L1 data.
    l1_loc = os.path.join(loc, 'l1s')
    l1_df = pd.DataFrame(data=diffexp_data['l1s'],
                         columns=genes)
    l1_df.insert(1, 'samples', samples) # Insert a samples column.
    l1_df.to_csv(l1_loc) # Save the dataframe.

    ngenes = len(genes)
    down_data, up_data = {}, {}
    # Save the up and down-regulated genes by sample.
    for sample in samples:
        up_loc, down_loc = os.path.join(loc, 'upregulated.csv'), os.path.join(loc, 'downregulated.csv')
        # Pad the ends of the arrays with np.nan so the length is constant.    
        up_arr = diffexp_data['samples'][sample]['up']
        down_arr = diffexp_data['samples'][sample]['down']
        up_arr = np.pad(up_arr, (0, ngenes - len(up_arr)), constant_values=np.nan)
        down_arr = np.pad(down_arr, (0, ngenes - len(down_arr)), constant_values=np.nan)
        
        up_data[sample] = up_arr
        down_data[sample] = down_arr
    
    up_df, down_df = pd.DataFrame(data=up_data), pd.DataFrame(data=down_data)
    up_df.to_csv(up_loc)
    down_df.to_csv(down_loc)

    print(f'Differential expression data saved to {loc}.')


