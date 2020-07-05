# The Grid class

The Grid class inherits from the parent Plot class. It allows you to manage a group of subplots, each of which is its
own individual Plot object, and display them in a grid. 

## Initialization

The Grid class initialzer takes the following inputs:

* `pop` **dict** The PopAlign object.
* `class_` **str** The class of Plot the subplots will belong to. One of `'heatmap'`, `'barplot'`.
* `params` **dict** A dictionary containing arguments for the subplots. This will be passed into the subplot class initializer
as keyword arguments. 
* `var` **tuple** A two-tuple where the first element is a string representing the name of the parameter which varies by 
subplot, and the second element is a list containing the values the parameter takes on. For example, params=('celltype', 
['Myeloid', 'T cell'])

## Usage

The Grid class interface is similar to that of a singular Plot object. It is also broken into initialization, plotting, and 
saving steps, and the method names are identical; in fact, the Grid class implements the Plot class `save()` method directly.

```python
from plotpop import grid # Import the grid module.

params = {'type_':'g_s', 'celltype':'T cell', 'gene':'CD3D', 'merge_samples':True} # The BarPlot parameters.
var = ('sample', ['ALL CYTO', 'CCL3_10', 'CCL3_50']) # The variable parameter.
grid = grid.Grid(pop, class_='barplot', params=params, var=var)

grid.plot()
grid.save(filename='example_grid.png')
```  
Note that the variable parameter specified by `var` must be excluded from the `params` dictionary; if it is not, an 
AssertionError will be thrown. Additionally, the `is_subplot` argument and pop arguments are excluded from the `params` dictionary,
as they are assigned by the Grid initializer. 

As was is true for singular plots, a color can be specified in the `plot()` method. The color passed into the Grid class 
`plot()` function is passed into the `plot()` function of the subplot class. The code below keeps the BarPlot default colors, 
and produces the grid below. 
