import numpy as np
from matplotlib import colors
import re

from plotpop import data
from plotpop import plot

class BarPlot(plot.Plot):
    '''
    '''
    def __init__(self, obj,
                 sample=None,
                 gene=None,
                 nbins=25,
                 binmax=None,
                 is_subplot=False):
        '''
        Initializes the BarPlot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        nbins : int
            The number of bins into which to sort the data. Default is 25.
        is_subplot : bool
            Whether or not the plot is a subplot.
        '''
        assert obj.npops == 1, 'BarPlots can only be initialized with a Data object containing one PopAlign object'
        pop = obj.pops[0] # Avoid storing the entire PopAlign object as an attribute.
        self.celltype = obj.celltype

        self.ctrls = [s for s in pop['samples'].keys() if re.match(pop['controlstring'], s) is not None]
 
        self.merge_samples = obj.merge_samples
        if sample in self.ctrls: # Make sure merge_samples is off if the sample is a control.
            self.merge_samples = False

       # Parent class inititalization ------------------------------------------------------      
        super().__init__(obj, 
                         is_subplot=is_subplot, 
                         filename='barplot',
                         color=('lightsalmon', 'turquoise'), 
                         plotter=self._plotter)
        self.title = f'{gene} in {sample} ({self.celltype})'
        self.xtitle = 'expression level'
        self.ytitle = 'cell fraction'
        
        # Initialization ---------------------------------------------------------------------- 
        self.gene = data.check_gene(pop, gene)
        self.geneidx = pop['filtered_genes'].index(self.gene)           
        # The inputted sample should not be named.
        self.sample = data.check_sample(pop, sample)

        self.bins = None # This will store the bin values.
        self.nbins = nbins
        self.ncells = 0 # This will be the number of cells represented by the distribution.
        self.mean, self.ctrl_mean = 0, 0 # The means of the data and controls.

        if binmax is None:
            self.binmax = self.__get_binmax(pop) # This will be the maximum bin value (i.e. bins[-1])
        else: # Allows the user to specify a binmax.
            self.binmax = binmax

        # Populate the data and bin attributes.
        self.data = self.__get_data(pop) 


    # G_S_* --------------------------------------------------------------------

    def __get_data(self, pop):
        '''
        Initializes the data and bin attributes with data from the pop object.
        '''
        if self.binmax == 0.0: 
            # If there is no expression of the gene for the given celltype, initialize an array
            # of zeroes where the first element is 1 (100 percent of cells have no expression).
            data = np.zeros(self.nbins, dtype=float)
            data[0] = 1.0
            self.bins = np.zeros(self.nbins + 1, dtype=float)
            self.ctrl_data = data
            # NOTE: Because there is no expression, leave self.mean as the default 0.
    
            return data
 
        ctrl_arr, ctrl_ncells = np.array([]), 1 
        for ctrl in self.ctrls:
            ctrl_idxs = self.__get_celltype_idxs(pop, sample=ctrl) # Get the control data of the first control sample. 
            ctrl_arr = np.append(ctrl_arr, pop['samples'][ctrl]['M_norm'].toarray()[self.geneidx][ctrl_idxs])
            ctrl_ncells += len(ctrl_idxs)   

        self.ctrl_mean = np.mean(ctrl_arr) # Store the mean as an attribute.
        ctrl_data, _ = np.histogram(ctrl_arr, bins=self.nbins, range=(0, self.binmax))
        self.ctrl_data = ctrl_data / ctrl_ncells # Normalize the data by cell number.
        
        idxs = self.__get_celltype_idxs(pop, sample=self.sample)
        ncells = len(idxs)
        arr = pop['samples'][self.sample]['M_norm'].toarray()[self.geneidx][idxs]
        
        rep = self.sample + '_rep'
        # If sample merging is turned on AND a replicate is present, merge the *_rep and sample data. 
        if self.merge_samples and rep in pop['order']: 
            rep_idxs = self.__get_celltype_idxs(pop, sample=rep) 
            arr = np.append(arr, pop['samples'][rep]['M_norm'].toarray()[self.geneidx][rep_idxs])
            ncells += len(rep_idxs) # Add the number of cells in the replicate sample to the total cell countself.
        
        self.mean = np.mean(arr) # Store the mean as an attribute.
        data, bins = np.histogram(arr, bins=self.nbins, range=(0, self.binmax))
        self.bins = bins # Store the bins in the object.
        self.ncells = ncells # Store the number of cells represented by the BarPlot.
        
        return data / ncells # Normalize the bin data and return.
    
    def __get_celltype_idxs(self, pop, sample):
        '''
        Gets the indices for the cells in the inputted sample corresponding to the celltype stored in 
        self.celltype. 
        
        Parameters
        ----------
        sample : str
            The name of the sample from which to retrieve data.
        '''
        celltypes = np.array(pop['samples'][sample]['cell_type'])
        celltype_idxs = np.where(celltypes == self.celltype)[0] # Get indices of cells with the correct celltype.
        
        return celltype_idxs

    def __get_binmax(self, pop):
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
        for sample in pop['samples'].keys(): # Get the max gene expression value across all samples
            idxs = self.__get_celltype_idxs(pop, sample=sample)  # Get indices of cells with the correct celltype.
            arr = pop['samples'][sample]['M_norm'].toarray()[self.geneidx][idxs] # Get gene data for a sample.
            
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

    def _plotter(self, axes, color=None, **kwargs):
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
        **kwargs : N/A
            Additional plotting settings. None are currently used in BarPlots.
        '''
        assert isinstance(color, tuple), 'Color must be a tuple for a BarPlot object.'
            
        barwidth = 1.0 / self.nbins # Width of each bar.
        # NOTE: Remember to remove the last bin element to ensure len(self.bins) is equal to 
        # len(self.data[sample]).
        axes.bar(self.bins[:-1], self.ctrl_data, 
                 color=colors.to_rgba(color[0], alpha=0.3), 
                 width=barwidth,
                 align='edge') # Add control data.
        axes.bar(self.bins[:-1], self.data, 
                 color=colors.to_rgba(color[1], alpha=0.3), 
                 width=barwidth,
                 align='edge') # Add experimental data.
        
        # Make the graph prettier!
        axes.set_yticks(np.arange(0, self.data.max(), 0.1))
        axes.set_xticks(np.round(self.bins, 3)[::5])
        axes.set_ylim(ymin=0, ymax=self.data.max())
        axes.legend(labels=['CONTROL', f'{self.sample}'])
        if self.binmax != 0:
            axes.set_xlim(xmin=0, xmax=self.binmax)
        else:
            axes.set_xlim(xmin=0, xmax=2.0)    
        axes.set_xlabel(self.xtitle, fontdict={'fontsize':20})
        axes.set_ylabel(self.ytitle, fontdict={'fontsize':20})

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
        testmean = self.mean
        if ref is None: # If no reference BarPlot is specified, use the control data. 
            refdata = self.ctrl_data
            refmean = self.ctrl_mean
        else: 
            refdata = ref.data
            refmean = ref.mean
        
        l1 = np.linalg.norm(testdata - refdata, ord=1) 
        if testmean < refmean:
            l1 *= -1 
        
        return l1


