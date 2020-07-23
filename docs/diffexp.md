# Unsupervised gene selection

Analyzing only a curated list of genes can result in the ommisison of several important signals from analysis. The 
`get_diffexp_data()` function, found in the `plot` module, prevents this by carrying out an exhaustive search of the 
PopAlign data in order to find the most differentially expressed (significantly up or down-regulated) genes.

## Arguments

The `plot.get_diffexp_data()` function takes the following arguments as input:

* `pop` **dict** The PopAlign object.
* `**kwargs`` Currently, a `celltype` **str** or `refpop` **int** are accepted as keyword arguments. This
specifies a specific filter through which to analyze the data (i.e. generate L1 values for a specific celltype
or subpopulation). If no keyword argument is specified, no filter is applied. 

## Calculating the L1 cutoff

The first step in unsupervised gene selection is approximating a cutoff. This is done by iterating over each control sample
(i.e. each sample matching the `pop['controlstring']`) For each control sample and each gene in `pop['filtered_genes']`, 
an L1 norm is calculated. A normalized probability
