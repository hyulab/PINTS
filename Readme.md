# PINTS: Peak Identifier for Nascent Transcripts Sequencing
![](https://img.shields.io/badge/platform-linux%20%7C%20osx-lightgrey.svg)
![](https://img.shields.io/badge/python-3.x-blue.svg)

## Installation
PINTS is available on PyPI, which means you can install it with the following command:
```shell
pip install pyPINTS
```
Alternatively, you can clone this repo to a local directory, then in the directory, run the following command:
```shell
python setup.py install
```

## Prerequisite
Python packages
* matplotlib
* numpy
* pandas
* pysam
* pybedtools
* pyBigWig
* scipy
* statsmodels

## Get started
PINTS can call peaks directly from BAM files. To call peaks from BAM files, 
you need to provide the tool a path to the bam file and what kind of experiment it was from.
If it's from a standard protocol, like PROcap, then you can set `--exp-type PROcap`.  
Other supported experiments including PROseq/PROcap/GROseq/GROcap/CoPRO. If the data was generated
by other methods, then you need to tell the tool where it can find ends of RNAs which you are interested in.
For example, `--exp-type R_5` tells the tool that:
   1. this is a single-end library; 
   2. the tool should look at 5' of reads. Other supported values are `R_3`, `R1_5`, `R1_3`, `R2_5`, `R2_3`.

If reads represent the reverse complement of original RNAs, like PROseq, then you need to use `--reverse-complement` 
(not necessary for standard protocols).

One example for calling peaks from BAM file:
```shell
pints_caller --bam-file input.bam --save-to output_dir --file-prefix output_prefix --thread 16 --exp-type PROcap
```
Or you can call peaks from BigWig files:
```shell
pints_caller --save-to output_dir --file-prefix output_prefix --bw-pl path_to_pl.bw --bw-mn path_to_mn.bw --thread 16
```
If you want to call peaks from experiments with replicates:
```shell
pints_caller --bam-file input1.bam input2.bam --save-to output_dir --file-prefix output_prefix --thread 16 --exp-type PROcap
```

## Outputs
* prefix+`_{SID}_divergent_peaks.bed`: Divergent TREs;
* prefix+`_{SID}_bidirectional_peaks.bed`: Bidirectional TREs (divergent + convergent);
* prefix+`_{SID}_unidirectional_peaks.bed`: Unidirectional TREs, maybe lncRNAs transcribed from enhancers (e-lncRNAs) as suggested [here](http://www.nature.com/articles/s41576-019-0184-5).

`{SID}` will be replaced with the number of samples that peaks are called from,
  if you only provide PINTS with one sample, then `{SID}` will be replaced with **1**,
  if you try to use PINTS with three replicates (`--bam-file A.bam B.bam C.bam`), then `{SID}` for peaks identified from `A.bam` will be replaced with 1.

For divergent or bidirectional TREs, there will be 6 columns in the outputs:
1. Chromosome
2. Start site: 0-based
3. End site: 0-based 
4. Confidence about the peak pair. Can be: 
    * `Stringent(qval)`, which means the two peaks on both forward and reverse strands are significant based-on their q-values; 
    * `Stringent(pval)`, which means one peak is significant according to q-value while the other one is significant according to p-value; 
    * `Relaxed`, which means only one peak is significant in the pair.
    * A combination of the three types above, because of overlap for nearby elements.
5. Major TSSs on the forward strand, if there are multiple major TSSs, they will be separated by comma `,`
6. Major TSSs on the reverse strand, if there are multiple major TSSs, they will be separated by comma `,`


For single TREs, there will be 6 columns in the output:
1. Chromosome
2. Start
3. End
4. Peak ID
5. Q-value
6. Strand

## Parameters
### Input & Output
* If you want to use BAM files as inputs:
   * `--bam-file`: input bam file(s);
   * `--exp-type`: Type of experiment, acceptable values are: `CoPRO`/`GROcap`/`GROseq`/`PROcap`/`PROseq`, or if you know the position of RNA ends which you're interested on the reads, you can specify `R_5`, `R_3`, `R1_5`, `R1_3`, `R2_5` or `R2_3`;
   * `--reverse-complement`: Set this switch if 1) `exp-type` is `Rx_x` and 2) reads in this library represent the reverse complement of RNAs, like PROseq;
   * `--ct-bam`: Bam file for input/control (optional);
* If you want to use bigwig files as inputs:
  * `--bw-pl`: Bigwig for signals on the forward strand;
  * `--bw-mn`: Bigwig for signals on the reverse strand;
  * `--ct-bw-pl`: Bigwig for input/control signals on the forward strand (optional);
  * `--ct-bw-mn`: Bigwig for input/control signals on the reverse strand (optional);
* `--save-to`: save peaks to this path (a folder), by default, current folder
* `--file-prefix`: prefix to all outputs

### Optional parameters
* `--mapq-threshold <min mapq>`: Minimum mapping quality, by default: 30 or `None`;
* `--close-threshold <close distance>`: Distance threshold for two peaks (on opposite strands) to be merged, by default: 300;
* `--fdr-target <fdr>`: FDR target for multiple testing, by default: 0.1;
* `--chromosome-start-with <chromosome prefix>`: Only keep reads mapped to chromosomes with this prefix, if it's set to `None`, then all reads will be analyzed;
* `--thread <n thread>`: Max number of threads the tool can create;
* `--borrow-info-reps`: Borrow information from reps to refine calling of divergent elements;
* `--output-diagnostic-plot`: Save diagnostic plots (independent filtering and pval dist) to local folder

More parameters can be seen by running `pints_caller -h`.

## Other tools
* `pints_boundary_extender`: Extend peaks from summits.
* `pints_visualizer`: Generate bigwig files for the inputs.
* `pints_normalizery`: Normalize inputs.

## Tips
1. Be cautious to reads mapped to scaffolds instead of main chromosome (for example the notorious `chrUn_gl000220` in `hg19`, they maybe rRNA contamination)!

## Contact
Please submit an issue with any questions or if you experience any issues/bugs.
