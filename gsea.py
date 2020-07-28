import plot

class GSEAPlot(plot.Plot):
    '''
    '''
    def __init__(self, # Should I include an option to initialize with a pop object?
                 diffexp_data=None,
                 is_subplot=False):
        '''
        Inititalizes a GSEA plot object.
        '''
        # Parent class initialization ------------------------------
        super().__init__(diffexp_data)
