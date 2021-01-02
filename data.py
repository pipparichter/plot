from plotpop import bar
import popalign as PA
import os

import scipy
import re
import time
import numpy as np
import pandas as pd


class Data():
    '''
    '''
    def __init__(self, pop,
                 output='~/out/',
                 sample_order=None,
                 get_diffexp=False,
                 cutoff=None,
                 get_l1s=False,
                 celltype=None,
                 merge_samples=True):
        '''
        Initializes an instance of a Data object.

        Parameters
        ----------
        pop : dict, list
            A PopAlign object or a list of PopAlign objects.
        output : str
            The output directory associated with the Data object; this is used
            for saving the data, as well as plots initialized using the Data object.
        sample_order : list
            A list of strings which re.match with samples in the pop objects. This specifies an order for the
            samples.
        get_diffexp : bool
            Whether or not to collect differential expression data.
        cutoff : float
            The cutoff for calculating differential expression data.
        get_l1s : bool
            Whether or not to calculate the L1 values.
        celltype : str
            The celltype for which to filter the collected data.
        merge_samples : bool
            Whether or not to merge replicate samples. True by default. 
        '''
        self.merge_samples = merge_samples
        self.output = output

        # Inititalize the PopAlign dictionaries underlying the Data object.
        if isinstance(pop, dict):
            self.pops = [pop]
        elif isinstance(pop, list):
            self.pops = pop
        else:
            raise Exception('Input must be a PopAlign object or a list of PopAlign objects.')
        for pop in self.pops:
            assert 'name' in pop.keys(), 'Each PopAlign object must have a name.'
        self.npops = len(self.pops)

        if sample_order is None:
            self.samples = self.__get_named_samples(order=[])
        else:
            self.samples = self.__get_named_samples(order=sample_order)
        self.samples_with_pop = self.__get_samples_with_pop(self.samples)
        
        # NOTE: A lot of genes seem to be removed in this step... check with Sisi to make sure this is OK.
        self.genes = merge_genes(self.pops) # Get the intersection of the genes in each pop input. This avoids errors later on.
        self.celltype = celltype
        
        # NOTE: Calculating the L1 values requires initializing a Data object (so initialization of a BarPlot
        # is consistent. In this case, get_l1s must be set to False, or an infinite loop starts.
        self.l1s = None
        if get_l1s:
            self.l1s = self.__get_l1s()

        # Differential-expression data inititalization -------------------------------
        self.diffexp_genes = None
        self.diffexp_by_sample = None
        self.upregulated = None
        self.downregulated = None
        
        if get_diffexp: # Check whether or not to run differential expression search.
            if self.l1s is None:
                self.l1s = self.__get_l1s() # If L1 values have not yet been calculated, calculate them.
            self.cutoff = cutoff
            self.__get_diffexp_data(cutoff=cutoff)

        # Retrieve a combined data matrix ---------------------------------------------
        self.mtx = self.__get_mtx()
    
    def __get_mtx(self):
        '''
        Collect data for the specified celltype across all given samples. 
        '''
        # Build a matrix with each row representing a sample, and each column representing data for
        # a particular cell. 
        mtx = []
        sample_labels = []
        
        for pop in self.pops:
            name = pop['name']
            for sample in pop['samples']:
                celltype_idxs = np.where(self.pop['samples'][sample]['cell_type'] == self.celltype)
                # Get the data from the original M matrix, which is normalized but not filtered. 
                # The M matrix has a column corresponding to each cell, and a row for each gene.
                mtx.append(pop['samples'][sample]['M'][:, celltype_idxs])
                sample_labels.append([f'{sample}_{name}'] * len(celltype_idxs))
        
        self.mtx = scipy.sparse.hstack(mtx) # This concatenates the columns of all columns stored in the mtx list.
        self.sample_labels = np.hstack(sample_labels)

    def __get_named_samples(self, order=[]):
        '''
        Takes in a list of pop objects and returns a list of samples with pop['name'] appended.
        If an ordering is specified, the rearranged sample list is returned. 

        Parameters
        ----------
        order : list
            A list of strings which match with samples to specify an order.
        '''
        named_samples = np.array([])
        for i in range(self.npops):
            pop, name = self.pops[i], self.pops[i]['name']
            # Check and filter sample names. 
            samples = check_samples(pop, pop['order'], filter_reps=True, filter_ctrls=True)
            for sample in samples:
                named_sample = f'{sample}_{name}' # Combine the experiment and sample names. 
                named_samples = np.append(named_samples, named_sample)
        
        ordered_named_samples = np.array([])
        for elem in order:
            matches = np.array([s for s in named_samples if re.match(elem, s) is not None]) 
            ordered_named_samples = np.append(ordered_named_samples, matches) # Add the matches to named_samples.
            named_samples = np.delete(named_samples, np.in1d(named_samples, matches)) # Remove all matches in named_samples.
        ordered_named_samples = np.append(ordered_named_samples, named_samples) # Add any remaining named samples.
        
        return ordered_named_samples

    def __get_samples_with_pop(self, samples):
        '''
        Generates a list of tuples where the first tuple element is an integer corresponding to a pop object,
        and the second tuple element is a string corresponding to the name of a sample in that pop object. 
        (i.e. not including the experiment name). This is just a way to associate samples with a specific 
        pop object to clean up the get_diffexp_data() function.

        Parameters
        ----------
        samples : list
            A list of strings specifying sample names. If a sample order is specified, the 
            sample names must be in the format {SAMPLE NAME}_{EXPERIMENT NAME}, e.g. 'ALL CYTO_COVID1'.
        '''
        samples_with_pop = [] # This will store a list of (pop, sample) tuples. 
        pop_names = [pop['name'] for pop in self.pops]

        for named_sample in samples:
            # NOTE: This does not guarantee the naming scheme is correct, but does catch some cases. 
            assert '_' in named_sample, 'The sample naming scheme is incorrect.'
            s = named_sample.split('_')
            name = s[-1] # Get the experiment name.
            try:
                idx = pop_names.index(name)
                sample = '_'.join(s[:-1]) # Recreate the sample name without the experiment name. 
                samples_with_pop.append((idx, sample)) # Store the sample name with the corresponding pop object in a list. 
            except ValueError:
                raise Exception(f'No experiment matching {name} was found in the list of pop objects.')

        return samples_with_pop


    def __get_cutoff(self, pop):
        '''
        Calculates the L1 cutoff for a specified PopAlign object from the controls. This is for use within the 
        get_diffexp_data() function.

        Parameters
        ----------
        pop : dict
            The PopAlign object.
        '''
        ctrls = [s for s in pop['order'] if re.match(pop['controlstring'], s) is not None]
        
        print(f'Calculating cutoff...    \r', end='')
        ctrl_l1s = []
        for ctrl in ctrls:
            for gene in self.genes:
                obj = Data(pop) # merge_samples will be automatically disabled for controls.
                l1 = bar.calculate_l1(obj, gene=gene, sample=ctrl, celltype=self.celltype)
                ctrl_l1s.append(l1)
        distribution = scipy.stats.rv_histogram(np.histogram(ctrl_l1s, bins=100))
        cutoff = abs(distribution.ppf(0.001)) # Sometimes this value is negative, so take the absolute value.
        print(f'Cutoff is {cutoff}.    ')
        
        return cutoff

    def __calculate_l1(self, pop, gene, sample):
        '''
        Calculates the L1 norm for a particular gene in a particular sample relative to controls,
        and returns it.

        Parameters
        ----------
        pop : dict
            The PopAlign object.
        gene : str 
            A valid gene name.
        sample : str
            A valid sample name. 
        '''
        obj = Data(pop, 
                   merge_samples=self.merge_samples, 
                   celltype=self.celltype,
                   get_l1s=False, # Make sure this is False to avoid an infinite loop.
                   get_diffexp=False) # Make sure this is False to avoid an infinite loop.
        barplot = bar.BarPlot(obj, sample=sample, gene=gene) 
        l1 = barplot.calculate_l1()
        
        return l1

    def __get_l1s(self):
        '''
        Returns a dictionary which contains a 2-D samples-by-genes numpy array storing the L1 data for a pop
        object or a collection of pop objects.
        '''

        # Inititalize arrays to store data.
        all_l1s = []
        t0 = time.time()
        count = 1 # Count for keeping track of what sample we're on.
        for idx, sample in self.samples_with_pop:
            print(f'Gathering L1 data for sample {count} of {len(self.samples_with_pop)}...    \r', end='')
            
            pop = self.pops[idx] # Get the pop object corresponding to the index.    
            sample_l1s = []
            for gene in self.genes: # Store the up and down-regulated genes and their corresponding L1 values. 
                l1 = self.__calculate_l1(pop, sample=sample, gene=gene)
                sample_l1s.append(l1) # Add the L1 value to the list of sample L1s. 
            all_l1s.append(sample_l1s) # Add the sample L1 values to the matrix. 
            count += 1 # Increment the count by one.
        
        t1 = time.time()
        print(f'All L1 data gathered: {int(t1 - t0)} seconds.                           ')
        
        return np.array(all_l1s) # Store the calculated L1 values after converting to an array.
        
    # NOTE: This function probably will not work with a refpop filter, as it assumes reference populations
    # are aligned across experiments. Later, I will fix this (ask Sisi how to address this issue!). 
    def __get_diffexp_data(self, cutoff=None):
        '''
        It stores the following data as attributes in the object: a list of the
        up and down-regulated genes by sample, as well as an nsamples by ngenes 2-D array containing the
        calculated L1 values. If cluster is True, it also stores a list of gene clusters (genes which behave
        similarly across all samples). 
        '''
        diffexp_by_sample = dict({}) 

        # Inititalize arrays to store data.
        all_upreg, all_downreg = np.array([]), np.array([])
        t0 = time.time()
        count = 1 # Count for keeping track of what sample we're on.
        for idx, sample in self.samples_with_pop:
            print(f"Gathering differential expression data for sample {count} of {len(self.samples)}...    \r", end='')
            
            pop = self.pops[idx] # Get the pop object corresponding to the index.
            named_sample = f"{sample}_{pop['name']}"
            sample_idx = np.where(self.samples == named_sample)[0] # Get the index of the sample.
            sample_l1s = self.l1s[sample_idx].flatten() # The rows are samples, and the columns are genes. 
            
            if cutoff is None: # If no cutoff is specified, calculate it from controls. 
                cutoff = self.__get_cutoff(pop)
               
            up_idxs = np.where(sample_l1s >= self.cutoff)
            down_idxs = np.where(sample_l1s <= -1 * self.cutoff)
            # Get the indices with which to sort the genes by order of increasing L1 value.
            up_l1s, down_l1s = sample_l1s[up_idxs], sample_l1s[down_idxs] # Get L1 arrays.
            up_genes, down_genes = self.genes[up_idxs], self.genes[down_idxs] # Get corresponding gene names.
            up_sort, down_sort = np.argsort(up_l1s), np.argsort(down_l1s) # Get sorting indices.
           
            for gene in up_genes[up_sort].tolist():
                if np.count_nonzero(all_upreg == gene) == 0:
                    all_upreg = np.append(all_upreg, gene)
            for gene in down_genes[down_sort].tolist():
                if np.count_nonzero(all_downreg == gene) == 0:
                    all_downreg = np.append(all_downreg, gene)
            
            # Add the data to the diffexp_data dictionary under the relevant sample. 
            # Make sure to use the named sample!
            diffexp_by_sample[named_sample] = {}
            diffexp_by_sample[named_sample]['upregulated'] = up_genes
            diffexp_by_sample[named_sample]['downregulated'] = down_genes
            
            count += 1 # Increment the count by one.

        diffexp_genes = np.array([])
        for gene in np.concatenate((all_upreg, all_downreg)).tolist():
            if np.count_nonzero(diffexp_genes == gene) == 0:
                diffexp_genes = np.append(diffexp_genes, gene)
     
        t1 = time.time()
        print(f'All differential expression data gathered: {int(t1 - t0)} seconds.                      ')
        
        # Store the collected data in attributes.
        self.diffexp_genes = diffexp_genes
        self.diffexp_upreg = all_upreg
        self.diffexp_downreg = all_downreg
        self.diffexp_by_sample = diffexp_by_sample
   

    def save(self, dirname='data'):
        '''
        Saves a diffexp_data dictionary, created by the get_diffexp_data function, to a subdirectory of 
        out/. The diffexp_data is organized into four files: one containing upregulated genes by sample,
        one containing downregulated genes by sample, one storing the matrix of all L1 values, and one storing
        all differentially-expressed genes.

        Parameters
        ----------
        diffexp_data : dict
            The object containing all diffexp data. 
        dirname : str
            The name under which to store the diffexp_data files; pop['output'/diffexp/[DIRNAME].
        '''

        # Make the directory in which to store the diffexp data in the location specified
        # by the path argument. This is the home directory by default.
        loc = os.path.join(self.output, 'data', dirname)
        PA.mkdir(loc)
        # Save the L1 data.
        l1s_loc = os.path.join(loc, 'l1s')
        l1s_df = pd.DataFrame(data=self.l1s,
                             columns=self.genes)
        l1s_df.insert(1, 'samples', self.samples) # Insert a samples column.
        l1s_df.to_csv(l1s_loc) # Save the dataframe.
        
        if self.diffexp_genes is not None:
            # Save the differentially-expressed genes.
            diffexp_genes_loc = os.path.join(loc, 'diffexp_genes')
            diffexp_genes_df = pd.DataFrame(data={'genes':self.genes})
            diffexp_genes_df.to_csv(diffexp_genes_loc) # Save the dataframe.

            ngenes = len(self.genes)
            down_data, up_data = {}, {}
            # Save the up and down-regulated genes by sample.
            for sample in self.samples:
                up_loc, down_loc = os.path.join(loc, 'upregulated.csv'), os.path.join(loc, 'downregulated.csv')
                # Pad the ends of the arrays with np.nan so the length is constant.    
                up_arr = self.diffexp_by_sample[sample]['upregulated']
                down_arr = self.diffexp_by_sample[sample]['downregulated']
                up_arr = np.pad(up_arr, (0, ngenes - len(up_arr)), constant_values=np.nan)
                down_arr = np.pad(down_arr, (0, ngenes - len(down_arr)), constant_values=np.nan)
                
                up_data[sample] = up_arr
                down_data[sample] = down_arr
            
            up_df, down_df = pd.DataFrame(data=up_data), pd.DataFrame(data=down_data)
            up_df.to_csv(up_loc)
            down_df.to_csv(down_loc)

            print(f'Data object was saved to {loc}.')


# Auxiliary functions --------------------------------------------------------------------------------------

def merge_genes(pops):
    '''
    Gets the intersection of the pop['filtered_genes'] lists in each inputted pop object. This is used 
    when comparing two separate experiments.

    Parameters
    ----------
    pops : list
        A list of PopAlign objects. 
    '''
    merged = np.array(pops[0]['filtered_genes']) # Get the filtered genes list from the first pop object. 
    removed = []
    for pop in pops:
        genes = np.array(pop['filtered_genes'])
        idxs = np.in1d(genes, merged)
        
        removed_genes = np.delete(genes, idxs) # Create an array with all elements except the specified indices removed.
        removed.extend(removed_genes.tolist()) # Add all removed elements to the removed list. 

        merged = merged[np.in1d(merged, genes)] # Filter merged by the indices of the elements also in genes. 
        merged = check_genes(pop, merged.tolist()) # Check the merged list against the pop object.
    
    # print('The following genes were removed in the merge: ' + ', '.join(removed))

    return merged


def get_ncells(pop, sample=None, celltype=None):
    '''
    Returns the number of cells in the pop object.

    Parameters
    ----------
    sample : str
        The sample for which to retrieve the number of cells. 
    celltype : str
        The celltype for which to retrieve the number of cells. If None, the total number of cells in 
        the sample is returned.
    '''

    ncells = 0
    if sample is None:
        if celltype is None:
            ncells = pop['ncells']
        else:
            for sample in pop['order']:
                celltypes = np.array(pop['samples'][sample]['cell_type'])
                ncells += np.count_nonzero(celltypes == celltype)
    
    else: # If a sample is specified...
        if celltype is None:
            ncells = pop['samples'][sample]['M_norm'].shape[1]
        else:
            celltypes = np.array(pop['samples'][sample]['cell_type'])
            ncells = np.count_nonzero(celltypes == celltype)

    return ncells


def check_celltype(pop, celltype):
    '''
    Checks to make sure the inputted celltype is valid. If the celltype is valid, it is returned.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    celltype : str
        A string representing a possible celltype (e.g. T cell, B-cell, Myeloid).
    '''
    assert celltype is not None, 'A cell type must be specified.'
    assert celltype in pop['gmm_types'], f'{celltype} is not a valid celltype.'
    
    return celltype


def check_gene(pop, gene):
    '''
    Checks to make sure the inputted gene is valid. If the gene is valid, then it is returned.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    gene : str
        A string representing the gene name to be checked.        
    '''
    assert gene is not None, 'A gene name must be specified.'
    assert gene in pop['filtered_genes'], f'{gene} is an invalid gene name.'
    
    return gene


def check_sample(pop, sample):
    '''
    Checks the inputted sample to make sure it's a valid sample name. If the sample is valid, it
    is returned. 

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    sample : str
        A string representing a possible sample name.
    '''
    assert sample is not None, 'A sample name must be specified.'
    assert sample in pop['samples'].keys(), f'{sample} is an invalid sample name.'

    return sample


def check_genes(pop, genes):
    '''
    Checks a list of genes to make sure all are valid, and returns a numpy array of valid genes.
    Invalid genes are removed and printed.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    genes : list
        A list of genes to check.
    '''
    assert genes is not None, 'A gene list must be specified.'
    invalid = []
    for gene in genes[:]:
        if gene not in pop['filtered_genes']:
            genes.remove(gene)
            invalid.append(gene)
    
    if len(invalid) > 1:
        print('The following genes are invalid and were removed: ' + ', '.join(invalid))    
    assert len(genes) > 1, 'At least one valid gene name must be given.'

    return np.array(genes)


def check_samples(pop, samples, filter_reps=True, filter_ctrls=True):
    '''
    Checks a list of samples of samples to make sure all are valid. If valid, returns
    a numpy array containing the inputted sample names.

    Parameters
    ----------
    pop : dict
        The PopAlign object.
    samples : list
        A list of strings representing possible sample names.
    filter_reps : bool
        Whether or not to filter out replicate sample names (i.e. samples ending in '_rep'). 
        True by default.
    filter_ctrls : bool
        Whether or not to filter out samples matching the controlstring (stored in pop['controlstring']). 
        True by default.
    '''
    samples = list(samples) # Make sure samples is a list.
    for sample in samples[:]:
        check_sample(pop, sample)

        if filter_ctrls and re.match(pop['controlstring'], sample) is not None:
            samples.remove(sample)
        elif filter_reps and re.match('.*_rep', sample) is not None:
            samples.remove(sample)
    
    return np.array(samples)
        






