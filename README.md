# Calcium Spike Analyzer

Python modules for detection and basic analysis of spikes in intraneuronal calcium (represented by fluorescence signal) resulting from electrical stimulation. Analysis can be done on individual data files (x = time, y = signal) or a directory of such files.

![alt text](https://github.com/crotty-d/calcium-spike-analyzer/blob/master/calspikes/results/568region_B_wholeplot.png)

![alt text](https://github.com/crotty-d/calcium-spike-analyzer/blob/master/calspikes/results/568region_B_waveforms_plot.png?raw=true)

The project is at a very early stage, and currently focused on particular experiments I carried out during my PhD: periodic electrical stimulation of cultured neurons stained with a fluorescent dye for intracellular calcium. However, relatively minor adjustments to parameters could allow for application to other related signals/experiments.

There are some example input data files (from fluorescence imaging videos of periodically stimulated neurons) in the 'data' directory and some example output in the 'results' directory. The 'reference_files' directory holds validated spike-timing files for performance testing. These directories are used throughout the analysis code, but others can be used by editing directories.py (global changes) or via various individual function parameters (local changes).

For analysis of an individual example data file, just run spikes_analysis.py as is. Along with numerical output to the terminal, the data is plotted in a native window via matplotlib (ipython console only) and in a browser via Bokeh (html file saved to 'results' directory). To analyze your own file, place it in the 'data' directory and modify the filename parameter in the main function of spikes_analysis.py; then run it. To analyze all the files in the data directory, run batch_analysis.py, which outputs aggregate statisitics for the spikes of each input data file. These gathered into a data frame and saved to a csv file in the 'results' folder.

The Numpy, Pandas, Matplotlib and Bokeh packages are required. I would recommend using Anaconda or similar scientific/analytical Python distribution.
