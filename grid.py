import matplotlib.pyplot as plt

import popalign as PA
from plotpop import bar
from plotpop import heatmap
from plotpop.plot import Plot

class Grid(Plot):

    def __init__(self, pop, 
                 class_=None,
                 params={},
                 var=None):
        '''
        Initializes a Grid object.

        pop : dict
            The pop object.
        class_ : str
            The class of plot to be plotted on the grid.
        params : dict
            The parameters which will be passed into the Plot subclass inititializer
            as keyword arguments.
        var : tuple
            A two-tuple where the first element is a string representing the name of the parameter which varies by 
            subplot, and the second element is a list containing the values the parameter takes on. For example, 
            params=('celltype', ['Myeloid', 'T cell'])
        '''
        self.filepath = [pop['output'], 'plots']
        self.filename = f'{class_}_grid.png'
        self.nplots = len(var[1])
        self.class_ = class_
        self.plotlist = []
    
        self.figure, self.axes = None, None # These attributes will store the figure and axes objects.
     
        classes = {'heatmap':heatmap.HeatmapPlot, 'bar':bar.BarPlot}
        self.class_ = classes[class_]

        for x in var[1]:
            param, value = var[0], x
            params[param] = value
            plot = self.class_(pop, is_subplot=True, **params)
            self.plotlist.append(plot)
        
        self.plotted = False
        self.is_subplot = False

    def __init_figure(self):
        '''
        Initializes the Grid figure and axes, and stores them as attributes. 
        '''
        self.nrows, self.ncols = PA.nr_nc(self.nplots)
        self.figure, self.axes = plt.subplots(self.nrows, self.ncols, figsize=(80, 80))

    def plot(self, color=None, fontsize=20):
        '''
        A function to plot all subplots in self.plotlist in the grid.

        Parameters
        ----------
        color : Dependent on class_
            The color scheme to use when plotting the subplots. For more details on possible inputs
            for this parameter, see the documentation for the specifice class_. If None, the class_'s 
            default color is used. 
        fontsize : int
            The size of the axes font. 
        '''
        self.__init_figure()

        for i in range(self.nplots):
            plot = self.plotlist[i]
            plot.plot(color=color, fontsize=fontsize, axes=self.axes[i])
        
        self.plotted = True
