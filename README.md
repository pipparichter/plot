# plotpop

A collection of tools used for visualizing data stored in a PopAlign object. Each tool described below inherits from 
a parent `Plot` class (see [plot.py](./plot.py)).

Plot generation is broken into three steps:

1. **Initialization**, when argument information is stored in attributes, and data is gathered and stored in a `data` attribute.
2. **Plotting**, when visuals are generated and displayed using `matplotlib`. 
3. **Saving**, when the `save()` method, defined in the parent `Plot` class, is called, and the generated plot is saved to an output 
    file.

This modularization allows the efficient use of Plot objects within other Plot objects. For example, 
distributions identical to those generated upon BarPlot initialization are used within the HeatmapPlot class. Separation of 
initialization and plotting allows easy creation of distribution data by initializing a BarPlot within a HeatmapPlot, while
avoiding the computationally-intensive process of producing a visual.


## Derived classes

1. [**BarPlot**](./docs/barplot.md)
2. [**HeatmapPlot**](./docs/heatmap.md)
3. [**ScatterPlot**](./docs/scatterplot.md)

## Coming soon...

- [x] Adding clustering to Heatmap.
- [x] Making Heatmap grids functional. 
- [ ] Adding more heatmap types to the class.
- [ ] A ScatterPlot class. 

