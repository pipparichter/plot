import numpy as np
from matplotlib import colors

from plotpop import plot

class BarPlot(plot.Plot):
    # Class constructor.
    def __init__(self, pop,
                 gene=None,
                 celltype=None,
                 sample=None,
                 nbins=25,
                 merge_samples=False,
                 is_subplot=False):
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
        nbins : int
            The number of bins into which to sort the data. Default is 25.
        merge_reps : bool
            Whether or not to merge the data for replicate samples.
        '''
        self.merge_samples = merge_samples

        # NOTE: init_samples is not called in the Plot initializer because Plot needs to be initialized with
        # the number of plots. This is the simplest solution I could think of.
        self.sample = sample
        self.gene = gene
        self.geneidx = pop['filtered_genes'].index(gene)
        self.celltype = celltype
        assert celltype in pop['gmm_types'], 'Invalid cell type.'  # Make sure the cell type name is valid.
       
        super().__init__(pop, is_subplot=is_subplot) # Call the parent class initializer.
        
        # Set the plotting function and default colors.
        self.plotter = self._plotter
        self.color = ('lightsalmon', 'turquoise')
        
        # Adjust the filepath.
        self.filepath.append('barplots')
        self.filename = f'barplot.png'
      
        # NOTE: len(self.bins) is one greater than the data arrays.
        self.bins = None # This will store the bin values.
        self.nbins = nbins
        self.binmax = 0.0 # This will be the max bin value.

        # Populate the data and bin attributes. Note that self.data is already defined as an attribute in
        # the parent class. 
        self.data = self.__get_data() 

    # Private methods --------------------------------------------------------------------

    def __get_data(self):
        '''
        Initializes the data and bin attributes with data from the pop object.
        '''
        self.__binmax() # Get the max bin value and store it in the binmax attribute.
       
        ctrl_data, ctrl_ncells = np.zeros(self.nbins), 0
        for ctrl in self.ctrls:
            d, n = self.__get_sample_data(ctrl) # Get the control data of the first control sample. 
            ctrl_data = np.add(ctrl_data, d)
            ctrl_ncells += n
        self.ctrl_data = ctrl_data / ctrl_ncells # Normalize the data by cell number.
        
        sample_data, ncells = self.__get_sample_data(self.sample)
        if self.merge_samples: # If merge_samples, combine the data from the *_rep sample.
            rep = self.sample + '_rep'
            rep_data, rep_ncells = self.__get_sample_data(rep) # Get the replicate data.
            ncells += rep_ncells # Add the number of cells in the replicate sample to the total cell countself.
            data = np.add(sample_data, rep_data) / ncells # Normalize the data and add it to self.data.
        else:
            data = sample_data / ncells

        return data

    def __get_sample_data(self, sample):
        '''
        Gets the data used for creating a barplot for a specific sample and a 
        specific gene. It populates the bins attribute and returns a 1-D
        numpy array containing the cell count in each bin, as well as the total number of cells in the 
        sample of self.celltype.

        Parameters
        ----------
        sample : str
            The name of the sample from which to retrieve data.
        '''
        # NOTE: ncells is initialized as one in case self.binmax == 0.0; then, dividing by ncells in the 
        # __get_data() function won't alter the distribution.
        ncells = 1
        if self.binmax == 0.0: 
            # If there is no expression of the gene for the given celltype, initialize an array
            # of zeroes where the first element is 1 (100 percent of cells have no expression).
            data = np.zeros(self.nbins, dtype=float)
            data[0] = 1.0
            bins = np.zeros(self.nbins + 1, dtype=float)
        else: 
            celltypes = np.array(self.pop['samples'][sample]['cell_type'])
            cellidxs = np.array(celltypes == self.celltype) # Get indices of cells with the correct celltype.
            ncells = np.count_nonzero(celltypes == self.celltype) # The number of self.celltype cells.

            unfiltered = self.pop['samples'][sample]['M_norm'].toarray()[self.geneidx] # Get gene data for a sample.
            filtered = unfiltered[cellidxs] # Filter by celltype.
            # Sort the filtered data into self.nbins evenly-spaced bins.
            data, bins = np.histogram(filtered, bins=self.nbins, range=(0, self.binmax))

        if self.bins is None: # If the bin attribute has not been filled... 
            self.bins = bins
        
        # NOTE: The number of cells is returned (rather than used within this function to modify data)
        # in order to support merge_reps without duplicating too much code.
        return data, ncells 

    def __binmax(self):
        '''
        Gets the maximum gene expression value across all samples for a particular gene
        and celltype (namely self.gene and self.celltype).
        '''
        binmax = 0.0    
        for sample in self.pop['samples'].keys(): # Get the max gene expression value across all samples
            celltypes = np.array(self.pop['samples'][sample]['cell_type'])
            cellidxs = np.where(celltypes == self.celltype) # Get indices of cells with the correct celltype.
            
            # NOTE: M_norm is of type scipy.sparse.csc.csc_matrix.
            unfiltered = self.pop['samples'][sample]['M_norm'].toarray()[self.geneidx] # Get gene data for a sample.
            filtered = unfiltered[cellidxs] # Filter by celltype.
            
            samplemax = filtered.max()
            if samplemax > binmax:
                binmax = samplemax
        
        self.binmax = binmax # Set the max_exp attribute.

    def _plotter(self, axes, color=None, fontsize=None):
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
        color : tuple
            Colors of the control and experimental data bars, respectively. See 
            matplotlib.colors module documentation for information on possible colors. 
        fontsize : int
            The font size for the axes.
        '''
        assert isinstance(color, tuple), 'Color must be a tuple for a BarPlot object.'

        barwidth = 1.0 / self.nbins # Width of each bar.
        # NOTE: Remember to remove the last bin element to ensure len(self.bins) is equal to 
        # len(self.data[sample]).
        axes.bar(self.bins[:-1], self.ctrl_data, 
                 color=colors.to_rgba(color[0], alpha=0.5), 
                 width=barwidth,
                 align='edge') # Add control data.
        axes.bar(self.bins[:-1], self.data, 
                 color=colors.to_rgba(color[1], alpha=0.5), 
                 width=barwidth,
                 align='edge') # Add experimental data.

        # Make the graph prettier!
        axes.set_title(f'{self.gene} in {self.sample} ({self.celltype})')
        axes.set_xticks(np.round(self.bins, 3)[::5])
        if self.binmax != 0:
            axes.set_xlim(xmin=0, xmax=self.binmax)
        else:
            axes.set_xlim(xmin=0, xmax=2.0)    
        axes.set_ylabel('cell fraction', fontdict={'fontsize':fontsize})
        axes.set_xlabel('expression level', fontdict={'fontsize':fontsize})
        axes.set_yticks(np.arange(0, self.data.max(), 0.1))
        axes.set_ylim(ymin=0, ymax=self.data.max())
        axes.legend(labels=[f'{self.sample}', 'CONTROL'])
   
    def calculate_l1(self, ref=None):
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
            assert ref.bins == self.bins and self.celltype == ref.celltype and self.gene == ref.gene, \
                    'Reference BarPlot is not compatible.'
            refdata = ref.data
        
        l1 = np.linalg.norm(testdata - refdata, ord=1)
        if testdata.mean() > refdata.mean():
            l1 *= -1
            
        return l1


