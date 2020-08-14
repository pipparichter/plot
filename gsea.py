from plotpop import plot
import popalign as PA
import numpy as np
import os
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

class GSEAPlot(plot.Plot):
    '''
    '''
    def __init__(self, obj, 
                 is_subplot=False,
                 geneset_file='c5bp'):
        '''
        Inititalizes a GSEA plot object.
        
        Parameters
        ----------
        obj : data.Data
            The Data object.
        is_subplot : bool
            Whether or not the Plot is a subplot.
        geneset_file : str
            The file storing the geneset data to be read in.
        '''
        # Parent class initialization ------------------------------
        super().__init__(obj, 
                         is_subplot=is_subplot,
                         color={'heatmap':'Reds', 'barplot':'turquoise'},
                         filename='gsea',
                         plotter=self._plotter)

        # Inititalization -------------------------------------------------------------
        currpath = os.path.abspath(os.path.dirname(__file__)) # Get the filepath of this file.
        self.genesets = PA.load_dict(os.path.join(currpath, f'gsea/{geneset_file}.npy'))
        self.geneset_names = np.array(list(self.genesets.keys())) # A list of geneset names. 
       
        assert obj.diffexp_genes is not None, \
            'Differential expression analysis has not been carried out on the Data object.'
        self.allgenes = obj.genes
        self.genelists = obj.diffexp_by_sample # The dictionary storing the differentially-expressed genes by sample.
        self.samples = obj.samples

        # Get the information for creating a hypergeometric distribution. 
        # NOTE: Variable naming conventions used are consistent with those in scipy.hypergeom.
        self.M = len(self.allgenes)
        
        self.top_genesets = np.array([]) # This list will store the most enriched genesets across ALL samples.
        self.data = self.__get_data()

        self.ylabels = np.array(list(self.data.keys())) # Get the names of the 

    def __get_data(self):
        '''
        Gets the data used to create GSEAPlots. The self.data attribute will be a dictionary with the keys
        being sample names. The values themselves are sub-dictionaries with the keys being geneset names, 
        and the values being the corresponding p-value.
        '''
        data = {}
        for sample in self.samples:
            up_and_down = self.genelists[sample]  # Get the up and down-regulated genes.
            genelist = np.concatenate((up_and_down['upregulated'], up_and_down['downregulated']))
            # The case of len(genelist) == 0 is handled in __get_sample_data(). 
            data[sample] = self.__get_sample_data(genelist)
        print(data) 
        return data

    def __get_sample_data(self, genelist):
        ''' 
        This function returns a dictionary where the keys are the geneset names and the values are the 
        corresponding p-values.

        Parameters
        ----------
        genelist : list
            A list of genes.
        '''
        N = len(genelist) # The number of genes in the genelist. 
        
        if N == 0: # If the genelist is empty, inititalize an array with np.nan.
            p_values = np.full((len(self.genesets)), np.nan)
        else: # If there are genes, calculate the p-value for each geneset and store in an array.
            p_values = np.array([])
            for geneset_name in self.geneset_names:
                geneset_genes = self.genesets[geneset_name] # Get the genes in a geneset.
                genes = genelist[np.in1d(genelist, geneset_genes)] # Get the list of genes in both the genelist and geneset.
                k = len(genes) # The number of genes in both the geneset and genelist.
                n = len(geneset_genes) # The number of genes in the geneset.

                # NOTE: The 'survival function' is the same as the hypergeometric test. It is the same as 1 - cdf, or the 
                # probability that more than k genes in the geneset would be present in the genelist if the distribution
                # was random.
                p_value = scipy.stats.hypergeom.sf(k, self.M, n, N) # Calculate the p-value.
                p_values = np.append(p_values, p_value)
            # Add the top genesets for the sample to the top_genesets list.
            top = self.geneset_names[np.argsort(p_values)][:2]
            self.top_genesets = np.append(self.top_genesets, top)
            self.top_genesets = np.unique(self.top_genesets) # Remove any duplicates. 

        sample_data = {}
        for geneset, p_value in zip(self.geneset_names.tolist(), p_values.tolist()):
            sample_data[geneset] = p_value # Store the geneset name and the corresponding p-value.

        return sample_data          

    def __get_plotter_data(self, style):
        '''
        Converts the information stored in self.data to a format relevant to the style in which the 
        p-values will be displayed. If type='heatmap', data will be a 2-D array. If type='barplot', 
        data will be a 1-D array.

        Parameters
        ----------
        style : str
            The format in which the p-values will be displayed. One of: heatmap, barplot.
        '''
        if style == 'heatmap':
            ylabels = self.top_genesets
            xlabels = self.samples

            data = np.zeros(shape=(len(ylabels), len(xlabels))) # Inititalize a 2-D array for the heatmap.
            for i in range(len(xlabels)):
                sample = xlabels[i]
                sample_data = self.data[sample]

                for j in range(len(ylabels)):
                    geneset = ylabels[j]
                    p_value = sample_data[geneset] # Get the p-value associated with the geneset and sample.
                    
                    data[j, i] = p_value # Assign the p-value to the correct pixel.
            # data = np.log10(data) # Take the base-10 log of every element.

        elif style == 'barplot': # If the selected style is barplot...
            pass

        else:
            raise Exception('The style parameter must be one of: heatmap, barplot.')

        return data 

    def _plotter(self, axes, 
                 color=None, 
                 fontsize={}, 
                 style='heatmap',
                 sample=None):
        '''

        Parameters
        ----------
        style : str

        '''
        # Initialize custom colors and fontsize.
        # NOTE: Color will never be None, as it is passed in in the Plot.plot() method
        if isinstance(color, dict): # If the default is passed into plot...
            color = color[style] # Select the color corresponding to the specified style.
        x_fontsize = fontsize.get('x', 20)
        y_fontsize = fontsize.get('y', 20)
        title_fontsize = fontsize.get('title', 30)
 
        # Get the data for plotting in the correct format for the specified style.
        data = self.__get_plotter_data(style)
        
        if style == 'heatmap':
            axes.set_title('GSEA results', fontdict={'fontsize':title_fontsize})
            axes.axis('off') # Stop borders from being plotted. 
            ylabels = self.top_genesets
            xlabels = self.samples
            
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
            d = 0
            mainw, mainh = 0.775 * w, 0.975 * h
            
            # Create axes for the main heatmap and the colorbar.
            c = 0.6 * h  # The length of the colorbar.
            mainax = self.figure.add_axes([x0 + d, y0, mainw, mainh], frame_on=False)
            cax = self.figure.add_axes([x0 + 0.95 * w, y0 + mainh / 2 - c / 2, 0.05 * w, c], frame_on=False)
            
            # Plot the heatmap on the main axes.
            cmap = plt.get_cmap(color) # Get the colormap.
            data = -1 * data # Smaller p-values are more significant.
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            print(vmin, vmax)
            mainax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

            # Set the axes ticks. 
            mainax.set_xticks(np.arange(0, len(xlabels))) 
            mainax.set_yticks(np.arange(0, len(ylabels))) 
            # Set the axes labels, with the correct orientation and font size.
            mainax.set_yticklabels(ylabels, fontdict={'fontsize':y_fontsize})
            xlabels = mainax.set_xticklabels(xlabels, fontdict={'fontsize':x_fontsize})
            for label in xlabels: # Make x-axis labels vertical.
                label.set_rotation('vertical')
            
            # Add a colorbar to the colorbar axes.
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) # Turn the selected colormap into a ScalarMappable object.
            cbar = plt.colorbar(mappable, cax=cax, ticks=[vmin, vmax])
            cbar.ax.set_title('log10(p-value)', fontdict={'fontsize':20})
            cbar.ax.set_yticklabels([f'{-1 * vmin}', f'{-1 * vmax}']) # When setting the y labels, make sure to flip the signs.
        
        elif style == 'barplot':
            pass
            # axes.set_title(f'GSEA results for sample {sample}')

            # if color is None: # If no color is specified, use a default.
                # color='turquoise'

