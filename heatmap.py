import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import time
import scipy.cluster.hierarchy as sch
# import multiprocessing
# import os
# import re
import scipy.stats

import sys
sys.path.insert(0, './popalign/popalign')
import popalign as PA

from plotpop import plot
from plotpop import barplot

class HeatmapPlot(plot.Plot):
    '''

    '''
    def __init__(self, pop, 
                 type_=None,
                 is_subplot=False,
                 cluster=True, 
                 **kwargs):
        '''
        Initializes a HeatmapPlot object.

        Parameters
        ----------
        pop : dict
            The pop object.
        type_ : str
            One of 's_ct', 'sp_s', or 'sp_rp'. This specifies what will be plotted on the Heatmap.
        is_subplot : bool
            Whether or not the heatmap is a subplot.
        cluster : bool
            Whether or not to cluster the data. Clustering options can be specified in cluster_params.
        kwargs : N/A
            A series of keyword arguments, details of which are specified in the documentation. 
        '''
           
        self.merge_samples = kwargs.get('merge_samples', True)
        
        # Initialize dictionary of genes and their corresponding indices.
        self.genedict = {gene:pop['filtered_genes'].index(gene) for gene in pop['filtered_genes']}
        self.allgenes = np.array(pop['filtered_genes'])
        self.samples = plot.check_samples(pop, 
                                          samples=np.array(pop['order']), 
                                          filter_reps=self.merge_samples, 
                                          filter_ctrls=True)
        
        # Parent class initialization --------------------------------------------------------
        super().__init__(pop, is_subplot=is_subplot) 
        self.plotter = self._plotter
        self.color = 'bwr'
        
        # Clustering -----------------------------------------------------------------------
        self.cluster = cluster 
        self.cluster_plot_dendrograms = kwargs.get('cluster_plot_dendrograms', True)
        self.cluster_metric = kwargs.get('cluster_metric', 'euclidean') 
        self.cluster_linkage = kwargs.get('cluster_linkage', 'complete') 
        self.cluster_nclusters = kwargs.get('cluster_nclusters', 3) 
        self.cluster_axis = kwargs.get('cluster_axis', 'both')

        self.lms = [None, None] # A list which will store the linkage matrices.
        
        # Differential expression ----------------------------------------------------------
        self.diffexp = None
        self.upreg = np.array([])
        self.downreg = np.array([])
        self.genes = plot.check_genes(pop, kwargs.get('genes', None)) 
        
        # Type-specific initialization ------------------------------------------------------
        options = ['s_ct', 's_rp']
        assert type_ in options, f'The type_ parameter must be one of: {options}.'
        self.type_ = type_
        
        if type_ == 's_ct': # Samples filtered by a celltype.
            self.celltype = plot.check_celltype(self.pop, kwargs.get('celltype', None))
        
        elif type_ == 's_rp': # Samples filtered by a reference population.
            self.refpop = kwargs.get('refpop', None)
            self.ref = pop['ref'] # Store the name of the reference sample.
            self.celltype = pop['samples'][self.ref]['gmm_types'][self.refpop] # Get the type of the subpopulation.
       
        if self.genes is None: # If no genes are given, proceed with unsupervised gene selection.
            self.data = self.__s_get_data(unsupervised=True)
        else: # If genes are specified, use those genes. 
            self.data = self.__s_get_data(unsupervised=False)

        self.ylabels = self.samples
        self.xlabels = self.genes
        
        # Adjusting the filepath -------------------------------------------------------------
        self.filepath.append('heatmap') # Assign plot to 'heatmap' directory.
        self.filename = f'heatmap.png'
    
    # S_* ------------------------------------------------------------------------------------------------
    
    def __s_get_data(self, unsupervised=False):
        '''
        Gets the data to be plotted on the heatmap. This function creates and
        returns a 2-D array with the rows corresponding to samples in self.samples and the columns corresponding
        to genes. If the unsupervised option is set, unsupervised analysis is use to calculate all differentially
        expressed genes.
        
        Parameters
        ----------
        celltype : str
            The celltype for which to collect the data, i.e. the label of the subplot.
        unsupervised : bool
            Whether or not genes should be selected by an unsupervised algorithm.
        '''
        if self.type_ == 's_ct':
            params = {'celltype':self.celltype}
        elif self.type_ == 's_rp':
            params = {'refpop':self.refpop} 
 
        if unsupervised: # If analysis is unsupervised, get the cutoff.
            data_getter = self.__get_sample_data_unsupervised
            cutoff = self.__get_cutoff(**params) # Get the L1 cutoff from controls.
            params['cutoff'] = cutoff # Add the cutoff to params.
            genes = self.allgenes
        else:
            data_getter = self.__get_sample_data
            cutoff = None
            genes = self.xlabels
        
        ngenes, nsamples = len(genes), len(self.samples)
        t0 = time.time() # Get the start time for performance info.
        # nans = [] # List to store the indices to be poplulated with NaNs.
        data = np.zeros((nsamples, ngenes)) # Initialize an array where rows are samples and columns are genes.
        for i in range(nsamples):
            sample = self.samples[i]
            # rep = sample + '_rep'
            print(f'Gathering {sample} data...    \r', end='')

            # mincells = 50 # The minimum number of cells needed to make a valid distribution
            # ncells = plot.get_ncells(self.pop, sample=sample, celltype=celltype)
            # if self.merge_samples: 
            #     ncells += plot.get_ncells(self.pop, sample=rep, celltype=celltype)
            # if ncells < mincells:
            #     nans.append(i)
            
            sampledata = data_getter(sample=sample, **params)
            data[i] = sampledata # Add the L1 data to the data matrix.

        t1 = time.time()
        print(f'Sample data gathered: {t1 - t0} seconds    ')
        
        if unsupervised:
            # Remove duplicates from the list of differentially expressed genes, while preserving order. 
            # NOTE: return_index=True makes np.unique return an array of indices which result in the unique array. 
            diffexp = np.append(self.upreg, self.downreg)
            _, order = np.unique(diffexp, return_index=True) 
            diffexp = diffexp[np.sort(order)] # Sort the indices so xlabels is in the original order. 
            # Filter L1 data by differentially expressed genes.
            geneidxs = np.array([self.genedict[x] for x in diffexp.tolist()])
            data = data[:, geneidxs]
            self.diffexp = diffexp # Store differentially expressed genes.
            self.genes = diffexp # Switch the genes variable from allgenes to diffexp.

        if self.cluster: # If clustering is set to True...
            if self.cluster_axis == 'x':
                data, genes = self.__cluster(data, axis='x')
            elif self.cluster_axis == 'y':
                data, samples = self.__cluster(data, axis='y')
            elif self.cluster_axis == 'both':
                data, genes  = self.__cluster(data, axis='x')
                data, samples = self.__cluster(data, axis='y')
            # Change the axes labels stored in the object to match the new ordering.
            self.xlabels = genes
            self.ylabels = samples

        # For the datapoints corresponding to distributions with ncells < mincells, replace the current value with NaN.
        # for i in nans:
            # data[i] = np.array([np.nan] * data.shape[1])

        return data
     
    def __get_sample_data(self, sample=None):
        '''
        Retrieve the L1 norm for a certain sample for all specified genes. As this analysis is 
        supervised, a genes list must be stored in self.xlabels.

        Parameters
        ----------
        sample : str
            The sample from which to collect data.
        **kwargs : str or int
            Either a celltype (str) or reference population number (int), which will be passed into 
        '''
        genes = self.xlabels
        if self.type_ == 's_ct':
            bar_type = 'g_s_ct'
            bar_params = {'celltype':self.celltype}
        elif self.type_ == 's_rp':
            bar_type = 'g_s_rp'
            bar_params = {'refpop':self.refpop}
        
        l1s = np.array([])
        for gene in genes:
            bar_params.update({'gene':gene, 'sample':sample, 'merge_samples':self.merge_samples})
            bar = barplot.BarPlot(self.pop, type_=bar_type, **bar_params)    
            l1 = bar.calculate_l1() # Get the L1 metric for the distribution (reference is control by default).
            l1s = np.append(l1s, l1) # Add the L1 value.

        return l1s
    
    def __get_sample_data_unsupervised(self, sample=None, cutoff=0.5, merge_samples=True, **kwargs):
        '''
        '''
        if self.type_ == 's_ct':
            bartype = 'g_s_ct'
            assert kwargs.get('celltype', None) is not None, 'A celltype must be specified for type_ s_ct.'
        elif self.type_ == 's_rp':
            bartype = 'g_s_rp'
            assert kwargs.get('refpop', None) is not None, 'A reference population must be specified for type_ s_rp.'
        
        l1s = np.array([])
        for gene in self.allgenes:
            params = {'gene':gene, 'sample':sample, 'merge_samples':merge_samples}
            params.update(kwargs)
            bar = barplot.BarPlot(self.pop, type_=bartype, **params) 
            
            l1 = bar.calculate_l1()
            l1s = np.append(l1s, l1)
        
        upidxs = np.where(l1s > cutoff)
        downidxs = np.where(l1s < -1 * cutoff)
        self.upreg = np.append(self.upreg, self.allgenes[upidxs])
        self.downreg = np.append(self.downreg, self.allgenes[downidxs])
 
        return l1s
      
        # pool = multiprocessing.pool.Pool(os.cpu_count()) # Inititialize multiprocessing object.
        # l1s = pool.starmap(barplot.calculate_l1, [(testbars[i], refbars[i]) for i in range(ngenes)])
        # NOTE: starmap() is just like map, but it passes the elements in the iterable as individual arguments
        # rather than as a single argument (i.e. func(*args) rather than func(args))

    def __get_cutoff(self, tail=0.001, **kwargs):
        '''
        Calculate the L1 cutoff using the control samples.

        Parameters
        ----------
        tail : float
            The percentage of all genes which should be included in the tails of the cutoff.
        '''
        print(f'Calculating cutoff for {self.celltype}...    \r', end='')
        ctrl_l1s = np.array([])
        for ctrl in self.ctrls:
            # Turn off merge_samples when evaluating the controls. 
            l1s = self.__get_sample_data_unsupervised(sample=ctrl, merge_samples=False, **kwargs)
            ctrl_l1s = np.append(ctrl_l1s, l1s)
        
        distribution = scipy.stats.rv_histogram(np.histogram(ctrl_l1s, bins=100))
        cutoff = abs(distribution.ppf(tail)) # Sometimes this value is negative, so return the absolute value.
        print(f'Cutoff for {self.celltype} is {cutoff}.    ')

        return cutoff

    # Clustering ------------------------------------------------------------------------------------
    
    def __cluster(self, data, axis=None):
        '''
        Runs an agglomerative hierarchical clustering algorithm on the inputted matrix. It uses
        the cluster settings inputted at Heatmap initialization. It returns the modified matrix, 
        the reordered data labels, and the clustering model generated.

        Parameters
        ----------
        matrix : numpy.ndarray
            A 2-D array which contains Heatmap data.
        labels : numpy.array
            An array which contains the data labels for the axis being sorted. 
        axis : str
            One of 'x' or 'y'. If 'y', the rows of the matrix are clustered; if 'x', 
            the columns of the matrix are clustered.
        '''
        if axis == 'x': # If clustering is to be carried out by column...
            matrix = data.T # Get the transpose of the inputted matrix.
            labels = self.xlabels
        elif axis == 'y':
            matrix = data
            labels = self.ylabels
            
        # NOTE: This function returns a  matrix such that element {i, j} is the distance between 
        # the ith and jth vectors of the given matrix. 
        X = skl.metrics.pairwise_distances(X=matrix, metric=self.cluster_metric) # Get the distance matrix.
        model = skl.cluster.AgglomerativeClustering(n_clusters=self.cluster_nclusters,
                                                    affinity='precomputed', # Distance matrix was precomputed.
                                                    linkage=self.cluster_linkage) # Create the clustering model.
        
        sorter = get_sorter(model, X=X) # Get the indices with which to sort the data and data labels.
        labels = labels[sorter] # Sort the labels to match the data.
        if axis == 'y':
            data = data[sorter, :] # Sort the data.
        elif axis == 'x':
            data = data[:, sorter] # Sort the data.

        # NOTE: Make sure to re-calculate the distance matrix for non-transposed and sorted data so that the 
        # dendrograms are accurate.
        if axis == 'x':
            X = skl.metrics.pairwise_distances(X=data.T, metric=self.cluster_metric) # Re-calculate the distance matrix.
            self.lms[0] = sch.linkage(X, method=self.cluster_linkage) # Get the linkage matrix.
        elif axis == 'y':
            X = skl.metrics.pairwise_distances(X=data, metric=self.cluster_metric) # Re-calculate the distance matrix.
            self.lms[1] = sch.linkage(X, method=self.cluster_linkage) # Get the linkage matrix.

        return data, labels

    def __plot_dendrogram(self, daxes):
        '''
        Plots dendrograms on the x, y, or both axes, depending on which linkage matrices have been
        calculated.
        '''
        xdax, ydax = daxes # Get the axes for the x and y-axis dendrograms.
        ydax.axis('off') # Turn off axes display.
        xdax.axis('off')
        if self.lms[0] is not None: # If the x-axis was clustered...
            lm = self.lms[0] # Retrieve the x-axis linkage matrix.
            sch.dendrogram(lm, ax=xdax, 
                           orientation='top', 
                           color_threshold=0,
                           above_threshold_color='black',
                           no_labels=True)
        if self.lms[1] is not None: # If the y-axis was clustered...
            lm = self.lms[1] # Retrieve the y-axis linkage matrix.
            sch.dendrogram(lm, ax=ydax, 
                           orientation='left',
                           color_threshold=0,
                           above_threshold_color='black',
                           no_labels=True)

    def _plotter(self, axes, color=None, fontsize=None):
        '''
        Plots a heatmap on the inputted axes.        
        
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes of the subplot on which the graph will be plotted.
        index : 
            The name of the subplot to be plotted.
        color: str
            The name of a matplotlib.colors.Colormap to color the heatmap data.
        fontsize : int
            Size of the font on the axes labels (not the title). 
        '''
 
        assert isinstance(color, str), 'Color must be a string for a HeatMap object.'
            
        # Retrieve the data and data labels for the subplot.
        data = self.data
        xlabels, ylabels = self.xlabels, self.ylabels
        
        axes.axis('off') # Turn off the axes frame.
        axes.set_title('Expression by subpopulation', fontdict={'fontsize':30})
        
        # Get information for creating a layout; if the HeatmapPlot is a subplot, the layout
        # will need to be constructed relative the the larger figure.
        if self.is_subplot:
            nplots = len(self.figure.axes)
            i = self.figure.axes.index(axes)
            nr, nc = PA.nr_nc(nplots) # Get the number of rows and columns.
            w, h = 1.0 / nc, 1.0 / nr
            x0, y0 =  ((i // nc) - 1) * h, (i % nc) * w
        else:
            w, h = 1, 1
            x0, y0 = 0, 0
        # Create a layout for all the subcomponents of the heatmap...
        # If cluster=True and plot_dendrograms is set to True (or left blank), plot the dendrograms.
        if self.cluster and self.cluster_plot_dendrograms:
            d = 0.075 * w # The thickness of the dendrogram plots.
            mainw, mainh = 0.7 * w, 0.9 * h # The dimensions of the main axes.
            
            # Create axes for the x-and-y dendrograms.
            ydax = self.figure.add_axes([x0, y0, d, mainh], frame_on=False)
            xdax = self.figure.add_axes([x0 + d, y0 + mainh, mainw, d], frame_on=False)
            self.__plot_dendrogram((xdax, ydax)) # Plot the dendrograms.
        else:
            d = 0
            mainw, mainh = 0.775 * w, 0.975 * h
           
        # Create axes for the main heatmap and the colorbar.
        c = 0.6 * h  # The length of the colorbar.
        mainax = self.figure.add_axes([x0 + d, y0, mainw, mainh], frame_on=False)
        cax = self.figure.add_axes([x0 + 0.95 * w, y0 + mainh / 2 - c / 2, 0.05 * w, c], frame_on=False)
       
        # Plot the heatmap on the main axes.
        cmap = plt.get_cmap(color) # Get the colormap.
        cmap.set_bad(color='gray') # Set NaN values to be gray.
        mainax.imshow(data, cmap=cmap, aspect='auto', vmin=-2.0, vmax=2.0) 
        # Set the axes ticks. 
        mainax.set_xticks(np.arange(0, len(xlabels))) 
        mainax.set_yticks(np.arange(0, len(ylabels))) 
        # Set the axes labels, with the correct orientation and font size.
        mainax.set_yticklabels(ylabels, fontdict={'fontsize':fontsize})
        xlabels = mainax.set_xticklabels(xlabels, fontdict={'fontsize':fontsize})
        for label in xlabels: # Make x-axis labels vertical.
            label.set_rotation('vertical')
        # If dendrograms are plotted, move the y-tick labels to make room for the dendrograms.
        if self.cluster and self.cluster_plot_dendrograms: 
            mainax.yaxis.set_label_position('right')
            mainax.yaxis.tick_right() 
       
        # Add a colorbar to the colorbar axes.
        norm = mpl.colors.Normalize(vmin=-2.0, vmax=2.0)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) # Turn the selected colormap into a ScalarMappable object.
        cbar = plt.colorbar(mappable, cax=cax, ticks=[-2, 0, 2])
        cbar.ax.set_title('L1 norm', fontdict={'fontsize':fontsize})
        cbar.ax.set_yticklabels(['-2', '0', '2'], fontdict={'fontsize':fontsize})


# Accessory functions --------------------------------------------------------------------------------------

def get_sorter(model, X=None):
    '''
    Takes the indices generated by the AgglomerativeClustering model, which assigns each sample element to a 
    cluster, and produces a list of indices to sort each element according to its assigned cluster.
    
    Parameters
    ----------
    model : sklearn.cluster.AgglomerativeClustering
        The agglomerative clustering model generated using the data.
    X : np.array
        A 2-D array representing a distance matrix.
    '''
    clusteridxs = model.fit_predict(X=X) # Return the clustering labels.
    
    nelements = len(clusteridxs) # Get the number of elements to sort.
    idxs = list(range(nelements))
    sorter = [idx for (_, idx) in sorted(zip(clusteridxs, idxs))]
    
    return sorter


# def diffexp_get_cutoff(pop, refpop=None, tail=0.2):
#     '''
#     '''
#     ref = pop['ref']
#     # Get all controls that are not the ref sample.
#     ctrls = [s for s in pop['samples'].keys() if re.match(pop['controlstring'], s) is not None and s != ref]
#     
#     ctrl_l1s = np.array([])
#     for ctrl in ctrls:
#         # Turn off merge_samples when evaluating the controls. 
#         l1s, upreg, downreg = diffexp_get_sample(pop, refpop=refpop, sample=ctrl, merge_samples=False)
#         ctrl_l1s = np.append(ctrl_l1s, l1s)
#     
#     distribution = scipy.stats.rv_histogram(np.histogram(ctrl_l1s, bins=100))
#     return distribution.ppf(tail)
# 
# 
# def diffexp_get_sample(pop, refpop=None, sample=None, cutoff=0.5, merge_samples=True):
#     '''
#     Calculates the L1 norm for each gene. The vectors used to calculate the L1 norm contain the expression data
#     of that gene across each cell in a subpopulation (refpop and the aligned subpopulation in the specified sample).
#     It returns an numpy array of norms, one for each gene in pop['filtered_genes'], as well as which genes were
#     highly up-or-down regulated; these genes are determined by the cutoff. 
# 
#     Parameters
#     ----------
#     refpop : int
#         The index of the subpopulation in the reference sample. This is used to identify the 
#         aligned subpopulations in the inputted sample.
#     sample : str
#         The name of the sample being compared to the reference sample.
#     cutoff : float
#         The L1 norm cutoff which determines whether or not a gene is differentially expressed. If the
#         cutoff is 0.5, then only genes with an L1 value above 0.5 or below -0.5 are returned. 
#     merge_samples : bool
#         Whether or not the samples are being merged.
#     '''
#     # NOTE: The reference sample is one of the controls.
#     assert sample in pop['samples'].keys(), 'Sample name is invalid.'
#     ref = pop['ref']
#     
#     # NOTE: arr is a three-by-three array. I'm guessing the second column contains the indices of the
#     # reference subpopulation to which the subpopulation in the first column aligns. 
#     alignments = pop['samples'][sample]['alignments'] # Get the alignments for the specified sample.
#     testpop = np.where(alignments[:, 1] == refpop) # Get the index of the sample population aligned to refpop.
#     assert len(testpop) > 0, f'No alignments were found for {sample}.'
# 
#     l1s = np.array([])
#     for gene in pop['filtered_genes']:
#         testbar = barplot.BarPlot(pop, 
#                                   type_='g_s_rp', 
#                                   refpop=refpop, 
#                                   sample=sample, 
#                                   merge_samples=merge_samples,
#                                   init_controls=False,
#                                   gene=gene)
#         refbar = barplot.BarPlot(pop, 
#                                  type_='g_s_rp', 
#                                  refpop=refpop, 
#                                  sample=ref, 
#                                  merge_samples=False,
#                                  init_controls=False,
#                                  gene=gene)
#         l1s = np.append(l1s, barplot.calculate_l1(testbar, ref=refbar)) 
#     
#     # pool = multiprocessing.pool.Pool(os.cpu_count()) # Inititialize multiprocessing object.
#     # l1s = pool.starmap(barplot.calculate_l1, [(testbars[i], refbars[i]) for i in range(ngenes)])
#     # NOTE: starmap() is just like map, but it passes the elements in the iterable as individual arguments
#     # rather than as a single argument (i.e. func(*args) rather than func(args))
# 
#     genes = np.array(pop['filtered_genes'])
#     upidxs = np.where(np.array(l1s) > cutoff)
#     downidxs = np.where(np.array(l1s) < -1 * cutoff)
#     print(l1s)    
#     return np.array(l1s), genes[upidxs], genes[downidxs]





