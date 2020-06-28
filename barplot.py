import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

from plot import plot


class BarPlot(plot.Plot):
    # Class constructor.
    def __init__(self, pop, gene=None, celltype=None, samples=None, nbins=25):
        '''
        Initializes the BarPlot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        gene : str
            The official name of the gene being analyzed.
        samples : str, list
            The sample or list of samples to be analyzed. If None, all samples in the pop
            object are analyzed.
        celltype : str
            The cell type to be plotted.
        grid : bool
            Whether or not the barplot is a grid of subplots. If True, more than one sample must
            be given.
        nbins : int
            The number of bins into which to sort the data. Default is 25.
        '''
        # NOTE: init_samples is not called in the Plot initializer because Plot needs to be initialized with
        # the number of plots. This is the simplest solution I could think of.
        self.samples = plot._init_samples(pop, samples)
        nplots = len(self.samples)
       
        super().__init__(pop, nplots) # Call the parent class initializer.

        # Adjust the filepath.
        self.filepath.append('barplots')
        if nplots > 1:
            self.filename = f'barplot_grid.png'
        else:
            self.filename = f'barplot.png'

        self.gene = gene
        assert gene in pop['filtered_genes'], f'{gene} is an invalid gene name.' 
        self.geneidx = pop['filtered_genes'].index(gene)
        
        self.celltype = celltype
        assert celltype in pop['gmm_types'], 'Invalid cell type.'  # Make sure the cell type name is valid.
       
        # NOTE: len(self.bins) is one greater than the data arrays.
        self.bins = None # This will store the bin values.
        self.nbins = nbins
        self.binmax = 0.0 # This will be the max bin value.

        # Populate the data and bin attributes. Note that self.data is already defined as an attribute in
        # the parent class. 
        self.__get_data() 

    # Private methods --------------------------------------------------------------------

    def __get_data(self):
        '''
        Initializes the data and bin attributes with data from the pop object.
        '''
        self.__binmax() # Get the max bin value and store it in the binmax attribute.
        self.ctrl_data = self.__get_sample_data(self.ctrls[0]) # Get the control data of the first control sample. 
        
        self.data = {}
        for i in range(self.nplots):
            sample = self.samples[i]
            # NOTE: The '\r' sequence tells print to return to the beginning of the line. The 'end'
            # argument tells it not to move to the next line after printing. The print statement outside
            # outside of the for loop prevents the last print from being overwritten.
            # [https://stackoverflow.com/questions/5419389/how-to-overwrite-the-previous-print-to-stdout-in-python]
            # ax = self.axes[i]
            
            self.data[sample] = self.__get_sample_data(sample) # Get experimental data from each sample.

    def __get_sample_data(self, sample):
        '''
        Gets the data used for creating a barplot for a specific sample and a 
        specific gene. It populates the bins attribute and returns a 1-D
        numpy array containing the cell fraction value for each bin.

        Parameters
        ----------
        sample : str
            The name of the sample from which to retrieve data.
        '''
        if self.binmax == 0.0: 
            # If there is no expression of the gene for the given celltype, initialize an array
            # of zeroes where the first element is 1 (100 percent of cells have no expression).
            data = np.zeros(self.nbins, dtype=float)
            data[0] = 1.0
            bins = np.zeros(self.nbins + 1, dtype=float)
        else: 
            celltypes = self.pop['samples'][sample]['cell_type']
            cellidxs = np.array(celltypes == self.celltype) # Get indices of cells with the correct celltype.
            ncells = np.count_nonzero(celltypes == self.celltype) # The number of {celltype} cells.
            
            # NOTE: M_norm is of type scipy.sparse.csc.csc_matrix.
            unfiltered = self.pop['samples'][sample]['M_norm'].toarray()[self.geneidx] # Get gene data for a sample.
            filtered = unfiltered[cellidxs] # Filter by celltype.
            
            # Sort the filtered data into self.nbins evenly-spaced bins.
            data, bins = np.histogram(filtered, bins=self.nbins, range=(0, self.binmax))
            data = data / ncells # Make the data a percentage.

        if self.bins is None: # If the bin attribute has not been filled... 
            self.bins = bins
        
        return data 

    def __binmax(self):
        '''
        Gets the maximum gene expression value across all samples for a particular gene
        and celltype (namely self.gene and self.celltype).
        '''
        binmax = 0.0    
        for sample in self.samples: # Get the max gene expression value across all samples
            celltypes = self.pop['samples'][sample]['cell_type']
            cellidxs = np.where(celltypes == self.celltype) # Get indices of cells with the correct celltype.
            
            # NOTE: M_norm is of type scipy.sparse.csc.csc_matrix.
            unfiltered = self.pop['samples'][sample]['M_norm'].toarray()[self.geneidx] # Get gene data for a sample.
            filtered = unfiltered[cellidxs] # Filter by celltype.
          
            samplemax = filtered.max()
            if samplemax > binmax:
                binmax = samplemax
        
        self.binmax = binmax # Set the max_exp attribute.

    def __plotter(self, axes, index, color):
        '''
        Generate a single barplot for a specified gene using the inputted axes. 
        Transcript counts are plotted on the axis, and percentage of cells which 
        share that level of expression is plotted on the y-axis.
        
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to add the barplot. If None, new axes are created.
        index : str
            The index of the sample to be plotted.
        color: tuple
            Colors of the control and experimental data bars, respectively. See 
            matplotlib.colors module documentation for information on possible colors. 
        '''
        assert isinstance(color, tuple), 'Color must be a tuple for a BarPlot object.'
        
        # NOTE: I am using the index instead of the sample, because this allows the __show() 
        # method in the Plot class to be extended to other Plot subclasses (e.g. if each subplot is for
        # a different gene rather than a different sample). 
        sample = self.samples[index]
        data = self.data[sample]

        barwidth = 1.0 / self.nbins # Width of each bar.
        # NOTE: Remember to remove the last bin element to ensure len(self.bins) is equal to 
        # len(self.data[sample]).
        axes.bar(self.bins[:-1], self.ctrl_data, 
                 color=colors.to_rgba(color[0], alpha=0.5), 
                 width=barwidth,
                 align='edge') # Add control data.
        axes.bar(self.bins[:-1], data, 
                 color=colors.to_rgba(color[1], alpha=0.5), 
                 width=barwidth,
                 align='edge') # Add experimental data.

        # Make the graph prettier!
        axes.set_title(f'{self.gene} in {sample} ({self.celltype})')
        if self.nplots == 1:
            axes.set_xticks(np.round(self.bins, 3))
        else: # Draw fewer tick marks if the graph is a grid (looks much nicer). 
            axes.set_xticks(np.round(self.bins, 3)[::5])
        if self.binmax != 0:
            axes.set_xlim(xmin=0, xmax=self.binmax)
        else:
            axes.set_xlim(xmin=0, xmax=2.0)
            
        axes.set_ylabel('cell fraction')
        axes.set_yticks(np.arange(0, data.max(), 0.1))
        axes.set_ylim(ymin=0, ymax=data.max())
        axes.legend(labels=[f'{sample}', f'{self.ctrls[0]}'])

    # Public methods -------------------------------------------------
    
    def plot(self, color=('lightsalmon', 'turquoise')):
        '''
        Uses the data to generate graphs.
        
        color : tuple
            Colors of the control and experimental data bars, respectively. See 
            matplotlib.colors module documentation for information on possible colors. 
        '''
        # NOTE: After the base class's __init__ ran, the derived object has the attributes set there, as 
        # it's the very same object as the self in the derived class' __init__. You can and should just 
        # use self.some_var everywhere. 
        # [https://stackoverflow.com/questions/6075758/python-super-object-has-no-attribute-attribute-name]
        self._plot(self.__plotter, color)
    
    def calculate_l1(self, sample=None, ref=None):
        '''
        Calculates the L1 error metric and returns it.

        Parameters
        ----------
        ref : BarPlot
            Another BarPlot object, which will serve as the reference barplot.
            If None, the function uses the ctrl data. ref.grid must be False.
        sample : str
            The sample for which to compare the two distributions. If None, all samples stored in 
            self.samples are used. 
        '''
        testdata = self.data
        if ref is None:
            refdata = self.ctrl_data
        else: # If no reference BarPlot is specified, use the control data. 
            assert not ref.grid, 'Reference BarPlot object cannot be a grid.'
            # Make sure the inputted BarPlot is L1-compatible with the test BarPlot.
            assert ref.bins == self.bins and self.celltype == ref.celltype and self.gene == ref.gene, \
                    'Reference BarPlot is not compatible.'
            refsample = ref.samples[0]
            refdata = ref.data[refsample] # ref.data should be a dictionary with only one element.

        if sample is None:
            samples = self.samples
        elif isinstance(sample, str):
            samples = [sample]
    
        l1data = {}
        for s in samples:
            l1 = np.linalg.norm(testdata[s] - refdata, ord=1)
            if testdata[s].mean() > refdata.mean():
                l1 *= -1
            l1data[s] = l1
            
        return l1data


