import matplotlib.pyplot as plt
import os
# import sklearn as skl
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
    def __init__(self, obj,
                 is_subplot=False,
                 filename=None,
                 color=None,
                 plotter=None):
        '''
        Initializes a Plot object.

        Parameters
        ----------
        obj : data.Data
            An instance of the Data class.
        is_subplot : bool
            Whether or not the plot is a subplot.
        '''
       
        self.figure, self.axes = None, None
        self.is_subplot = is_subplot

        self.filepath = [obj.output, 'plots']
        self.filename = filename # This will be set in the derived class initializer. 
        self.color = color

        self.plotted = False # Once self.plot() has been called once, this becomes True.
        self.plotter = plotter # The plotting function, which will be initialized in a derived class initializer.
        
        # Plot features which will be set in the derived-class initializer.
        self.title = ''
        self.ytitle = ''
        self.xtitle = ''
        self.xlabels = []
        self.ylabels = []

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

    def plot(self, 
             color=None, 
             title=None, 
             fontsize={}, 
             axes=None, 
             **kwargs):
        '''
        Graphs the Plot object on the axes.

        Parameters
        ----------
        plotter : function
            The function to use to plot the data.
        color : N/A
            The color data for the plotter function. The format of this data varies by subclass.
        title : str
            The title of the plot. If None, the default title is used.
        axes : matplotlib.axes.Axes
            If plotting a subplot, this is the axes of the subplot. This parameter should not be 
            specified if not plotting a subplot.
        fontsize : dict
            Stores the font information. It allows variable setting of the x and y-axis font sizes,
            as well as the title.
        **kwargs : N/A
            Additional plotting settings specific to the type of plot.
        '''
        assert self.data is not None, 'Data has not been initialized.'
        if axes is not None:
            assert self.is_subplot == True, 'If a plotting axes is specified, the Plot must be a subplot.'

        if self.plotted:
            self.axes.clear()

        self.__init_figure(axes=axes)

        if color is None: # If color is not specified, use the default color. 
            color = self.color

        if title is None:
            title = self.title # If no title is specified, use the default title.
        # Inititalize font sizes.
        title_fontsize = fontsize.get('title', 28)
        self.x_fontsize = fontsize.get('x', 20)
        self.y_fontsize = fontsize.get('y', 20)
        # Set the title.
        self.axes.set_title(title, fontdict={'fontsize':title_fontsize})

        self.plotter(self.axes, color=color, **kwargs)
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
   

