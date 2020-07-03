import popalign as PA
from plotpop import barplot
from plotpop import heatmap

class Grid():

    def __init__(self, pop, 
                 plot_type=None,
                 subplots=None,
                 **kwargs):
        '''
        '''

        self.nplots = len(subplots)
        self.subplots = subplots
        self.plot_type = plot_type
    
        self.figure, self.axes = None, None # These attributes will store the figure and axes objects.
        self.nrows, self.ncols = PA.nr_nc(self.nplots)
     
        classes = {'heatmap':heatmap.HeatmapPlot, 'barplot':barplot.BarPlot}
        self.class_ = classes[plot_type]

        for subplot in subplots:
            self.class_(**kwargs)
        
