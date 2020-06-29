# The HeatmapPlot class

## Initialization

Heatmaps can be created using the `HeatmapPlot` class. The Heatmap class supports three different modes, which have slightly 
different initializations. The mode is determined by the `toplot` argument. The `toplot` string is of the format
`{yaxis}_{suplot}`; for example `sp_s` plots a heatmap or a grid of heatmaps with subpopulations (`sp`) on the 
y-axis, where each subplot corresponds to a single sample (`s`).

### `s_ct`

This HeatmapPlot mode plots genes on the x-axis and samples on the y-axis. The cells of the heatmap correspond to the expression
level of a gene in a particular sample; each plot represents data for a single celltype. The expression level is calculated by taking
the L1 distance metric between an expression distribution for the sample and one for the control (see **The BarPlot class** for
information on how these distributions are calculated.).

* `pop` **[dict]** The PopAlign object.
* `samples` **[str, list, None]** The sample or list of samples for which plots will be created. If `None`, all samples
    will be used (see `plot._init_samples()`). 
* `celltypes` **[str, list]** The celltypes for which the plot(s) will be created, either `Myeloid`, `B-cell`, or `T-cell`. If
    multiple celltypes are specified, a grid of heatmaps is created.
* `genes` **[str, list]** The genes for which the plot(s) will be created. The inputted gene name or list of names must be valid, 
    i.e. in the list `pop['filtered_genes']`.
* `clustering` **[bool]**
* `**kwargs` Optional clustering settings. See the **Clustering** subsection for more information. 

### `sp_s`

In progress...

### `sp_rp` 

In progress... 

### Clustering

Specific clustering settings can be specified using the following keyword arguments:

* 
* 
* 


## Usage


