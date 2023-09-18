Hillforts Primer

An Analysis of the Atlas of Hillforts of Britain and Ireland
https://www.dairsieonline.co.uk/hillforts/hillforts_primer_part_01.html

Mike Middleton, March 2022
https://orcid.org/0000-0001-5813-6347

This Hillforts Primer is a research tool that analyses, maps, plots, transforms and adds to the data published in The Atlas of Hillforts of Britain and Ireland (Lock & Ralston, 2017).
The atlas contains 4147 records, with each record having 244 columns of associated information.
The size and shape of the data can make it difficult to manipulate in traditional software so, the aim is to analyse and process the data programmatically to facilitate interpretation and reuse. 
The following csv files were created from data made available via the Atlas of Hillforts of Britain and Ireland Rest Service:

The original data is available here:

Lock, G. and Ralston, I. 2017. Atlas of Hillforts of Britain and Ireland. [ONLINE] Avalable at: https://hillforts.arch.ox.ac.uk
Rest services: https://maps.arch.ox.ac.uk/server/rest/services/hillforts/Atlas_of_Hillforts/MapServer
Licence: https://creativecommons.org/licenses/by-sa/4.0/ 

Python files

The python files in this folder are plain text archive copies of the notebooks. 
These are more archivally stable but they will not run in Pyhon without reformatting. For instance, notebooks are designed to output data using inline as in:

pd.info()  will output information about a dataframe in the notebook. To achieve the same result in Python, this function would need to be wrapped in a print statement:
print(pd.info()) to enable the information to be output to the terminal.

These files are included in the archive as they are more stable than the notebooks and they can be converted to .txt for archiving in a trusted digital repository. 
This should ensure the logic is retained even if the functionality is not.   

