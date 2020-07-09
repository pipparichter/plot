import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import time
import scipy.cluster.hierarchy as sch
import multiprocessing
import os
import re
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
           
        # Initialize dictionary of genes and their corresponding indices.
        self.genedict = {gene:pop['filtered_genes'].index(gene) for gene in pop['filtered_genes']}

        self.merge_samples = kwargs.get('merge_samples', True)
        
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
        
        # Type-specific initialization ------------------------------------------------------
        options = ['s_ct', 'sp', 'sp_rp']
        assert type_ in options, f'The type_ parameter must be one of: {options}.'
        self.type_ = type_

        if type_ == 's_ct':
            celltype = kwargs.get('celltype', None)
            genes = kwargs.get('genes', None)
            samples = kwargs.get('samples', None)
            self.celltype = plot.check_celltype(pop, celltype)
            self.xlabels, self.ylabels = plot.check_genes(pop, genes), plot.check_samples(pop, samples)
            self.data = self.__s_ct_get_data() # Populate the data attribute.
                
        elif type_ == 'sp':
            self.xlabels, self.ylabels = None, None # These attributes will be populated by get_data.
            self.ref = pop['ref'] # Store the name of the reference sample.
            self.allgenes = np.array(pop['filtered_genes']) # Initialize an ordered list of genes.  
            self.samples = plot.check_samples(self.pop, 
                                              samples=self.pop['order'], 
                                              filter_ctrls=True, 
                                              filter_reps=self.merge_samples)
            self.data = self.__sp_get_data()

        elif type_ == 'sp_rp':
            pass
        
        # Adjusting the filepath -------------------------------------------------------------
        self.filepath.append('heatmap') # Assign plot to 'heatmap' directory.
        self.filename = f'heatmap.png'

    
    # S_CT ------------------------------------------------------------------------------------------------
    
    def __s_ct_get_data(self):
        '''
        Gets the data to be plotted on the heatmap for the given celltype. This function creates and
        returns a 2-D array with the rows corresponding to samples in self.samples and the columns corresponding
        to genes in self.genes. 
        
        Parameters
        ----------
        celltype : str
            The celltype for which to collect the data, i.e. the label of the subplot.
        '''
 
        genes, samples, celltype = self.xlabels, self.ylabels, self.celltype
        ngenes, nsamples = len(genes), len(samples)

        t0 = time.time() # Get the start time for performance info.
        print(f'Gathering {celltype} data...    \r', end='')
        
        nans = [] # List to store the indices to be poplulated with NaNs.
        # Initialize a 2-D array, where there is a row for each sample and a column for each gene.
        data = np.zeros((nsamples, ngenes))
        for i in range(nsamples):
            sample = samples[i]
            rep = sample + '_rep'

            mincells = 50 # The minimum number of cells needed to make a valid distribution
            ncells = plot.get_ncells(self.pop, sample=sample, celltype=celltype)
            if self.merge_samples: 
                ncells += plot.get_ncells(self.pop, sample=rep, celltype=celltype)
            if ncells < mincells:
                nans.append(i)
            
            for j in range(ngenes):
                gene = genes[j]

                params = {'gene':gene, 'celltype':celltype, 'sample':sample, 'merge_samples':self.merge_samples}
                distribution = barplot.BarPlot(self.pop, type_='g_s_ct', **params)    
                l1 = distribution.calculate_l1() # Get the L1 metric for the distribution (reference is control by default).
                data[i, j] = l1
        
        t1 = time.time()
        print(f'{celltype} data gathered: {t1 - t0} seconds    ')

        if self.cluster: # If clustering is set to True...
            t0 = time.time()
            print(f'Clustering {celltype} data...    \r', end='')
            
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

            t1 = time.time()
            print(f'{celltype} data clustered: {t1 - t0} seconds    ')

        # For the datapoints corresponding to distributions with ncells < mincells, replace the current value with NaN.
        for i in nans:
            ngenes = len(genes)
            data[i, :] = np.array([np.nan] * ngenes)

        return data

    # SP -------------------------------------------------------------------------------------------------------

    def __sp_get_data(self): # IN PROGRESS
        '''
        This function gets the data for a gene-by-subpopulation heatmap, for all subpopulations across the entire 
        experiment.
        '''
        ngenes = len(self.allgenes)
        xlabels, data = np.array([]), np.empty((0, ngenes))
        
        gmmtypes = self.pop['samples'][self.ref]['gmm_types']
        for refpop in range(len(gmmtypes)):
            celltype = gmmtypes[refpop]
            print(f'Calculating cutoff for {celltype}...    \r', end='')
            cutoff = self.__diffexp_get_cutoff(self.pop) # Get the L1 cutoff from controls.
            print(f'Cutoff for {celltype} is {cutoff}.    ')

            t0 = time.time()
            for sample in self.samples:
                print(f'Gathering data for sample {sample}...    \r', end='')

                l1s, upreg, downreg = self.__diffexp_get_sample(refpop=refpop, sample=sample, cutoff=cutoff)
                    
                xlabels = np.append(np.append(xlabels, upreg), downreg)
                _, order = np.unique(xlabels, return_index=True)
                xlabels = xlabels[np.sort(order)]

                data = np.append(data, l1s)

            t1 = time.time()
            print(f'Sample data gathered: {t1 - t0} seconds    ')

        # Filter L1 data by xlabels.
        geneidxs = np.array([self.genedict[x] for x in np.nditer(xlabels)])
        data = data[:, geneidxs]
        self.xlabels = xlabels # All differentially expressed genes. 
        
        return data    
    
    # Unsupervised gene selection -------------------------------------------------------------------------------

    def __diffexp_get_sample(self, refpop=None, sample=None, cutoff=0.5):
        '''
        Calculates the L1 norm for each gene. The vectors used to calculate the L1 norm contain the expression data
        of that gene across each cell in a subpopulation (refpop and the aligned subpopulation in the specified sample).
        It returns an numpy array of norms, one for each gene in pop['filtered_genes'], as well as which genes were
        highly up-or-down regulated; these genes are determined by the cutoff. 

        Parameters
        ----------
        refpop : int
            The index of the subpopulation in the reference sample. This is used to identify the 
            aligned subpopulations in the inputted sample.
        sample : str
            The name of the sample being compared to the reference sample.
        cutoff : float
            The L1 norm cutoff which determines whether or not a gene is differentially expressed. If the
            cutoff is 0.5, then only genes with an L1 value above 0.5 or below -0.5 are returned. 
        merge_samples : bool
            Whether or not the samples are being merged.
        '''
        # NOTE: arr is a three-by-three array. I'm guessing the second column contains the indices of the
        # reference subpopulation to which the subpopulation in the first column aligns. 
        alignments = self.pop['samples'][sample]['alignments'] # Get the alignments for the specified sample.
        testpop = np.where(alignments[:, 1] == refpop) # Get the index of the sample population aligned to refpop.
        assert len(testpop) > 0, f'No alignments were found for {sample}.'

        l1s = np.array([])
        for gene in self.allgenes:
            testbar = barplot.BarPlot(self.pop, 
                                      type_='g_s_rp', 
                                      refpop=refpop, 
                                      sample=sample, 
                                      merge_samples=self.merge_samples,
                                      init_controls=False,
                                      gene=gene)
            refbar = barplot.BarPlot(self.pop, 
                                     type_='g_s_rp', 
                                     refpop=refpop, 
                                     sample=self.ref, 
                                     merge_samples=False,
                                     init_controls=False,
                                     gene=gene)
            l1s = np.append(l1s, barplot.calculate_l1(testbar, ref=refbar)) 
        
        # pool = multiprocessing.pool.Pool(os.cpu_count()) # Inititialize multiprocessing object.
        # l1s = pool.starmap(barplot.calculate_l1, [(testbars[i], refbars[i]) for i in range(ngenes)])
        # NOTE: starmap() is just like map, but it passes the elements in the iterable as individual arguments
        # rather than as a single argument (i.e. func(*args) rather than func(args))

        upidxs = np.where(np.array(l1s) > cutoff)
        downidxs = np.where(np.array(l1s) < -1 * cutoff)
        
        return np.array(l1s), self.allgenes[upidxs], self.allgenes[downidxs]
 
    def __diffexp_get_cutoff(self, refpop=None, tail=0.01):
        '''
        '''
        ctrls = [ctrl for ctrl in self.ctrls if ctrl != self.ref] # Get a list of controls that are not the ref sample.
        
        ctrl_l1s = np.array([])
        for ctrl in ctrls:
            # Turn off merge_samples when evaluating the controls. 
            l1s, upreg, downreg = self.__diffexp_get_sample(refpop=refpop, sample=ctrl)
            ctrl_l1s = np.append(ctrl_l1s, l1s)
        
        distribution = scipy.stats.rv_histogram(np.histogram(ctrl_l1s, bins=100))
        return distribution.ppf(tail)

  
    # ---------------------------------------------------------------------------------------------------

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
        axes.set_title(f'Expression in {self.celltype}', fontdict={'fontsize':30})
        
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





