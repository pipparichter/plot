import numpy as np
from matplotlib import colors
import popalign as PA

from plotpop import plot

class BarPlot(plot.Plot):
    '''
    '''
    def __init__(self, pop,
                 type_=None,
                 nbins=25,
                 is_subplot=False, 
                 **kwargs):
        '''
        Initializes the BarPlot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        type_ : str
            The type of BarPlot to be graphed.
        nbins : int
            The number of bins into which to sort the data. Default is 25.
        is_subplot : bool
            Whether or not the plot is a subplot.
        '''
        self.merge_samples = kwargs.get('merge_samples', True)
       
        self.bins = None # This will store the bin values.
        self.nbins = nbins
        self.binmax = 0 # This will be the maximum bin value (i.e. bins[-1])
        self.ncells = 0 # This will be the number of cells represented by the distribution.

        # Parent class inititalization ------------------------------------------------------      
        super().__init__(pop, is_subplot=is_subplot)
        # Set the plotting function and default colors.
        self.plotter = self._plotter
        self.color = ('lightsalmon', 'turquoise')
        
        # Type-specific initialization ------------------------------------------------------
        options = ['g_s_ct', 'g_s_rp']
        assert type_ in options, f'The type_ parameter must be one of: {options}.'
        self.type_ = type_

        if type_ == 'g_s_ct':
            celltype = kwargs.get('celltype', None)
            gene = kwargs.get('gene', None)
            sample = kwargs.get('sample', None)
            self.celltype = plot.check_celltype(pop, celltype)
            self.gene = plot.check_gene(pop, gene)
            self.sample = plot.check_sample(pop, sample)
            self.geneidx = pop['filtered_genes'].index(self.gene)

        elif type_ == 'g_s_rp':
            refpop = kwargs.get('refpop', None)
            gene = kwargs.get('gene', None)
            sample = kwargs.get('sample', None)
            self.refpop = refpop
            self.gene = plot.check_gene(pop, gene)
            self.sample = plot.check_sample(pop, sample)
            self.geneidx = pop['filtered_genes'].index(self.gene)
       
        # Populate the data, bin, and binmax attributes.
        self.init_ctrls = kwargs.get('init_ctrls', True)
        self.data = self.__g_s_get_data(init_ctrls=self.init_ctrls) 

        # Adjust the filepath -------------------------------------------------------------------
        self.filepath.append('barplots')
        self.filename = f'barplot.png'

    # G_S --------------------------------------------------------------------

    def __g_s_get_data(self, init_ctrls=True):
        '''
        Initializes the data and bin attributes with data from the pop object.

        Parameters
        ----------
        init_ctrls : bool
            Whether or not to initialize controls. This option reduces computational cost when using
            the BarPlot class to find differentially expressed genes. 
        '''
        # Assign the function which will be used for gathering the relevant indices. 
        if self.type_ == 'g_s_ct':
            idx_getter = self.__get_ct_idxs
        elif self.type_ == 'g_s_rp':
            idx_getter = self.__get_rp_idxs
        
        binmax = self.__g_s_binmax(idx_getter=idx_getter)
        self.binmax = binmax # Store the binmax in the object.

        if self.binmax == 0.0: 
            # If there is no expression of the gene for the given celltype, initialize an array
            # of zeroes where the first element is 1 (100 percent of cells have no expression).
            data = np.zeros(self.nbins, dtype=float)
            data[0] = 1.0
            self.bins = np.zeros(self.nbins + 1, dtype=float)
            self.ctrl_data = data
    
            return data
 
        ctrl_data, ctrl_ncells = np.zeros(self.nbins), 1 
        if init_ctrls: # If the init_ctrls setting is True...    
            for ctrl in self.ctrls:
                ctrl_idxs = idx_getter(ctrl) # Get the control data of the first control sample. 
                arr = self.pop['samples'][ctrl]['M_norm'].toarray()[self.geneidx][ctrl_idxs]
                d, _ = np.histogram(arr, bins=self.nbins, range=(0, binmax))
                ctrl_data = np.add(ctrl_data, d)
                ctrl_ncells += len(ctrl_idxs)    
        self.ctrl_data = ctrl_data / ctrl_ncells # Normalize the data by cell number.
        
        idxs = idx_getter(self.sample)
        ncells = len(idxs)
        arr = self.pop['samples'][self.sample]['M_norm'].toarray()[self.geneidx][idxs]
        data, bins = np.histogram(arr, bins=self.nbins, range=(0, binmax))

        if self.merge_samples: # If merge_samples, combine the data from the *_rep sample.
            rep = self.sample + '_rep'
            rep_idxs = idx_getter(rep) 
            arr = self.pop['samples'][rep]['M_norm'].toarray()[self.geneidx][rep_idxs]
            rep_data, bins = np.histogram(arr, bins=self.nbins, range=(0, binmax))

            ncells += len(rep_idxs) # Add the number of cells in the replicate sample to the total cell countself.
            data = np.add(data, rep_data) # Normalize the data and add it to self.data.
        
        self.bins = bins # Store the bins in the object.
        self.ncells = ncells # Store the number of cells represented by the BarPlot.
        return data / ncells # Normalize the bin data and return.
    
    def __get_rp_idxs(self, sample):
        '''
        Get the cell indices for cells in the specified sample belonging to the subpopulation which aligns with the reference 
        population stored in self.refpop. 

        Parameters
        ----------
        sample : str
            The name of the sample for which to gather data. 
        '''
        alignments = self.pop['samples'][sample]['alignments'] # Get the alignments information for the specified sample.
        r = np.where(alignments[:, 1] == self.refpop)[0] # Get the row index of the sample population aligned to refpop.
        subpop = alignments[r, 0] # Get the aligned subpopulation. 
        
        # Get the subpopulation assignments of every cell in the test and reference samples.
        t = PA.get_coeff(self.pop, sample)
        assignments = self.pop['samples'][sample]['gmm'].predict(t)
        # Get the indices of the cells which belong to refpop and the aligned test subpopulation.
        subpopidxs = np.where(assignments == subpop)[0]
        return subpopidxs

    def __get_ct_idxs(self, sample):
        '''
        Gets the indices for the cells in the inputted sample corresponding to the celltype stored in 
        self.celltype. 
        
        Parameters
        ----------
        sample : str
            The name of the sample from which to retrieve data.
        '''
        celltypes = np.array(self.pop['samples'][sample]['cell_type'])
        cellidxs = np.where(celltypes == self.celltype)[0] # Get indices of cells with the correct celltype.
        
        return cellidxs

    def __g_s_binmax(self, idx_getter=None):
        '''
        Gets the maximum gene expression value across all samples for a particular gene
        and celltype (namely self.gene and self.celltype).

        Parameters
        ----------
        idx_getter : function
            The function __g_s_binmax() uses to retrieve indices from each sample. The function 
            used depends on the type of BarPlot being generated.
        '''
        binmax = 0.0    
        for sample in self.pop['samples'].keys(): # Get the max gene expression value across all samples
            idxs = idx_getter(sample)  # Get indices of cells with the correct celltype.
            arr = self.pop['samples'][sample]['M_norm'].toarray()[self.geneidx][idxs] # Get gene data for a sample.
            
            if len(arr) == 0:
                # NOTE: In one instance, I ran into a problem where one of the samples was empty (I verified this with
                # pop['samples'][the empty samples]. For whatever reason, the metadata wasn't totally aligned with the
                # actual data, so this addressed that problem.   
                arr = np.zeros(1)
            samplemax = arr.max()
            if samplemax > binmax:
                binmax = samplemax
        
        return binmax # Return the binmax attribute.
    
    # -------------------------------------------------------------------------------------------------------

    def _plotter(self, axes, color=None, fontsize=None):
        '''
        Generate a single barplot for a specified gene using the inputted axes. 
        Transcript counts are plotted on the axis, and percentage of cells which 
        share that level of expression is plotted on the y-axis.
        
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to add the barplot. If None, new axes are created.
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
        if self.type_ == 'g_s_ct':
            title = f'{self.gene} in {self.sample} (CT: {self.celltype})'
        elif self.type_ == 'g_s_rp':
            title = f'{self.gene} in {self.sample} (RP: #{self.refpop})'
        axes.set_title(title)
        axes.set_xticks(np.round(self.bins, 3)[::5])
        if self.binmax != 0:
            axes.set_xlim(xmin=0, xmax=self.binmax)
        else:
            axes.set_xlim(xmin=0, xmax=2.0)    
        axes.set_ylabel('cell fraction', fontdict={'fontsize':fontsize})
        axes.set_xlabel('expression level', fontdict={'fontsize':fontsize})
        axes.set_yticks(np.arange(0, self.data.max(), 0.1))
        axes.set_ylim(ymin=0, ymax=self.data.max())
        axes.legend(labels=['CONTROL', f'{self.sample}'])
   
    def calculate_l1(self, ref=None):
        '''
        Calculates the L1 error metric and returns it.

        Parameters
        ----------
        ref : BarPlot
            Another BarPlot object, which will serve as the reference barplot.
            If None, the function uses the ctrl data.
        '''
        testdata = self.data
        if ref is None: # If no reference BarPlot is specified, use the control data. 
            # Make sure the control data has been inititalized.
            assert self.init_ctrls is True, 'Controls have not been inititalized; a reference BarPlot must be given.'
            refdata = self.ctrl_data
        else: 
            refdata = ref.data

        l1 = np.linalg.norm(testdata - refdata, ord=1) 
        if testdata.mean() < refdata.mean():
            l1 *= -1 
        
        return l1

# Accessory functions -----------------------------------------------------------------------

# NOTE: This function is a little redundant... It's main use is for flexibility, and so that
# L1 values can be calculated with multiprocessing for sp HeatmapPlots.
def calculate_l1(bar, ref=None):
    '''
    Calculates the L1 norm between two BarPlot objects and returns it. 

    Parameters
    ----------
    bar : BarPlot
        A BarPlot object which operates as the 'test' BarPlot.
    ref : BarPlot
        Another BarPlot, which operates as the 'reference' BarPlot. If None, the ctrls
        stored in the BarPlot object will be used. 
    '''
    return bar.calculate_l1(ref=ref)
