This code is a reimagining of 

Robert Staadt

Development of a system for high-volume multi-channel brain imaging of fluorescent voltage signals

Dissertation

Ruhr-Universität Bochum, Universitätsbibliothek

08.02.2024

[https://doi.org/10.13154/294-11032](https://doi.org/10.13154/294-11032)

-----------------------------------------------------------------------------------------------------

Updated: 19.03.2025

Files are now organized in subdirectories to distinguish better between code for GEVI or GECI analysis.

gevi-geci/
    stage_1*, stage_2*, stage_3*, stage_4*, stage_5*
	-> main stages for data preprocessing
	-> use e.g.: python stage_1_get_ref_image.py -c config_example_GEVI.json
    functions/
	-> functions used by the main stages
	
gevi-geci/gevi/
    config_example_GEVI.json
    	-> typical config file for GEVI (compare to gevi-geci/geci/config_example_GECI.json)
    config_M0134M*, config_M3905F*
	-> config files for a few recordings (adjust directory names, if necessary!)
    example_load_gevi.py
	-> simple script demonstrating how to load data

gevi-geci/geci/
    config_example_GECI.json
	-> typical config file for GECI (compare to gevi-geci/gevi/config_example_GEVI.json)
    config_M_Sert_Cre_4*
	-> config files for a few recordings (adjust directory names, if necessary!)
    stage_6_convert_roi.py
	-> additional stage for the analysis of Hendrik's recordings
	-> use e.g.: python stage_6_convert_roi.py -f config_M_Sert_Cre_41.json 
    geci_loader.py, geci_plot.py
	-> additional code for summarizing the results and plotting with the ROIs
	-> use e.g. python geci_loader.py --filename config_M_Sert_Cre_41.json
	
gevi-geci/other/
    stage_4b_inspect.py, stage_4c_viewer.py
	-> temporary code for assisting search for implantation electrode
	

    






