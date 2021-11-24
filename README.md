# VSANS-Pol-Reduction
VSANS (and NG7 SANS) Reduction for Polarized and Unpolarized Small-Angle Neutron Scattering. 
Author K.L. Krycka can be reached at klkrycket@gmail.com.

# Installation using Jupyter notebook:
* Copy AllSANS.ipynb and requirements.txt to a directory on your computer
* 'pip install -r requirements.txt' 	(one-time operation)
* 'pip install notebook'		(one-time operation)
* 'jupyer notebook'
* select AllSANS.ipnb and run all cells (shift + enter)
* user input choices and data directory are selected in top cell (parameters discussed at the end of this document)
* example data set can be found in the directory Example_Fe3O4Nanoparticles_VSANS26903
# Installation using Python code:
* copy AllSANS_ReductionHighRes.py, requirements.txt, and AllSANS_Usernput.py to your computer
* 'pip install -r requirements.txt' 	(one-time operation)
* 'python AllSANS_ReductionHighRes.py'
* user input choices are defined in AllSANS_Userinput.py (parameters discussed at the end of this document), which must be located in same folder as AllSANS_ReductionHighRes.py 
* example data set can be found in the directory Example_Fe3O4Nanoparticles_VSANS26903

# Usage notes:

This program will automatically sort and classify VSANS or NG7 SANS data taken at the NIST Center for Neutron Research. It will sort by unpolarized, half-polarized, and fully-polarized data sets, further sorting by intent and sample name (as determined from the description, minus configuration, temperature, and voltage data). The data will be put in absolute scale if the corresponding transmission files are present. Empties, open beam, and blocked beams are all utilized if present at the matching configurations. For VSANS, detector shadowing from nearby detector banks if fully accounted for (both in 1D data and 2D ASCII data). Attenuation is accounted for (although for He3 OUT and IN measurements, it should be held constant). Slices are automatically generated for each polarization type, per sample and condition (where ‘condition’ = unique combination of temperature, voltage, and instrument configuration).

The program will further reduce the data into structural and magnetic data, assuming that the field is oriented along the X-direction (to match with the Titan magnet). It automatically corrects for the angular dependencies associated with the finite sector cuts width (see J. Appl. Cryst. 2012, 45, 554-565 for details). Please note that M_Parl can be extracted from the difference of horizontal and vertical slices (often noisier) or from the division method for half-pol and full-pol (less noisy, but only gets the NET M_Parl contribution). Also note, M_Perp (full-pol) is set to be 3*M_Y in order to easily compare M_perp at remanence with M_Parl at saturation. For modeling purposes (see SasView.org), add a factor of 3 to the scale factor when comparing M_Perp to M_Parl or Structural Full-Pol scattering.

The polarization reduction is automatically taken care of for fully-polarized scattering, where empty needs only to have one non spin-flip and one spin-flip scattering file. See J. Appl. Cryst. 2012, 45, 546-553 (different from the previous reference) for details, with an updated version to be coming soon. You will need three He-3 measurements before a decay curve can be calculated – so take one as soon as possible when a fresh He3 cell is installed at the beamline and registered using NICE.

# Parameters (commonly modified):

* input_path -- select where your data is located
* save_path -- select where your reduced data should be stored (note if the listed folder doesn't exist, this program will make it for you).
* Instrument -- select 'VSANS' or 'NG7SANS' (exactly as written here)
* He3Only_Check -- a zero will run the full reduction, while 1 (or larger) will run only the abbreviated helium-3 reduction plots (to check on how cells are performing)
* SectorCutAngles -- determines half-width of secot cuts in degrees (i.e. 15 translated to +/- 15 degrees)
* Absolute_Q_min -- given in inverse angstroms, the program will take the maximum of Q_min_Calc (determined from all detectors) and this value
* Absolute_Q_max -- given in inverse angstroms, the program Will take the minimum of Q_max_Calc (from all detectors) and this value
* StrucutrallyIsotropic -- 0 means no and 1 means yes. This will determine if M_parallel from the division method should be calculated from both horizontal and vertical sector cuts (option 1) or if it should be restricted to only vertical cuts (option 0)
* AutoSubtractEmpty = -- 1 will automatically subtract any appropriate empty, 0 will not (and empty will be treated as another sample). Selecting 1 won’t cause any issues even if no empties are available.
* YesNoRenameEmpties -- 0 will not rename the empties, 1 = will simply rename them to Empty
* UseMTCirc – 0 will subtract empties pixel-by-pixel, 1 will instead subtracts circularly averaged empties
* TempDiffAllowedForSharingTrans -- this specifies the maximum temperature difference in Kelvin that can be used to fill in any missing transmission files from other transmission measurements that contain the same sample name
* Excluded_Filenumbers -- list any five-digit file numbers to be excluded from analysis, given as [] list and separated by commas for multiple entries 
* ReAssignBlockBeam -- list any five-digit file numbers to be re-classified as blocked beam measurements (if incorrectly labeled during file collection), given as [] list and separated by commas for multiple entries
* ReAssignEmpty -- list any five-digit file numbers to be re-classified as empty measurements (if incorrectly labeled during file collection), given as [] list and separated by commas for multiple entries
* ReAssignOpen -- list any five-digit file numbers to be re-classified as open beam measurements (if incorrectly labeled during file collection), given as [] list and separated by commas for multiple entries
* ReAssignSample -- list any five-digit file numbers to be re-classified as sample measurements (if incorrectly labeled during file collection), given as [] list and separated by commas for multiple entries
* Min_Filenumber – the minimum file number you wish to consider for analysis (default = 0)
* Max_Filenumber -- the maximum file number you wish to consider for analysis (default = 1000000)
* Min_Scatt_Filenumber – similar to Min_Filenumber, but restricted to scattering files only
* Max_Scatt_Filenumber – similar to Max_Filenumber, but restricted to scattering files only
* Min_Trans_Filenumber – similar to Min_Filenumber, but restricted to transmission files only
* Max_Trans_Filenumber – similar to Min_Filenumber, but restricted to transmission files only
* SampleDescriptionKeywordsToExclude – given as comma-separated strings in a [] list, this denotes any keywords you wish to have removed from your sample names
* YesNoSetPlotXRange – 0 means any generated plots will be automatically scaled, 1 means the user-selected minimum and maximum values for X will be used
* YesNoSetPlotYRange – 0 means any generated plots will be automatically scaled, 1 means the user-selected minimum and maximum values for Y will be used
* PlotXmin – minimum Qx value in inverse angstroms, only used if YesNoSetPlotXRange = 1
* PlotXmax – maximum Qx value in inverse angstroms, only used if YesNoSetPlotXRange = 1
* PlotYmin – minimum Qy value in inverse angstroms, only used if YesNoSetPlotYRange = 1
* PlotYmax – maximum Qy value in inverse angstroms, only used if YesNoSetPlotYRange = 1

# Parameters (rarely modified):

* SampleApertureInMM – if the sample aperture was conventionally entered in cm choose False; if the sample aperture were incorrectly entered into the file collection system in mm, choose True to correct
* PreSebtractOpen – Choice of 1 subtracts trans-scaled open (if available) from pol-full scattering files in attempt to remove main beam spillover before applying the polarization correction. Choice of 0 (the conventional method, default) does not do any pre-subtraction.
* Calc_Q_From_Trans – 0 means the beam centers recorded at the time of data collection will be maintained in the calculation of Q per pixel. 1 means that if an appropriate sample transmission taken at the same instrument condition, temperature, pol-state, and sample name as the scattering file exists, then the transmission beam center will be recalculated from the transmission files. The benefit of selecting 1 is that if the sample position drifts over time (or with temperature), the program can automatically make the appropriate correction.
* AverageQRanges – 0 means front and middle detector overlap on VSANS will not be averaged (low-Q front-detector will overwrite any overlapping high-Q middle-detector values). 1 means that this overlap region would be averaged on VSANS.
* YesNoShowPlots -- 0 means that the plethora of automatically generated plots will simply be save, but not shown to the user upon running the code (default). 1 means all the plots will be shown to the user before saving.
* CompareUnpolCirc – 1 allows all unpolarized circular averages from similarly-names samples to be shown together on a comparison plot. 0 means this plot will not be generated.
* CompareHalfPolSumCirc – 1 allows all half-pol circular averages from similarly-names samples to be shown together on a comparison plot. 0 means this plot will not be generated.
* CompareFullPolSumCirc – 1 allows all full-pol circular averages from similarly-names samples to be shown together on a comparison plot. 0 means this plot will not be generated.
* CompareFullPolStruc – 1 allows all full-pol structural-only scattering from similarly-names samples to be shown together on a comparison plot. 0 means this plot will not be generated.
* CompareFullPolMagnetism -- 1 allows all full-pol magnetic-only (M_Parl and M_Perp) scattering from similarly-names samples to be shown together on a comparison plot. 0 means this plot will not be generated.

* YesNo_2DCombinedFiles -- 1 produces 2D ASCII files which can be read by SasView (works for NG7 SANS and 2019 VSANS, newer VSANS seem to have an issue that needs to be tracked down!). Detector shadowing is automatically taken care of, with shadowed parts excluded. 0 will not produce these 2D ASCII files.
* YesNo_2DFilesPerDetector -- 1 allows individual VSANS detector to be inspected in 2D with SasView. 0 (Default) does not produce these 2D files.
* TransPanel – This is the panel where the VSANS transmissions are collected, typically 'MR'.
* all_detectors = Set to ["B", "MT", "MB", "MR", "ML", "FT", "FB", "FR", "FL"] (do not modify)
* nonhighres_detectors  -- Set to ["MT", "MB", "MR", "ML", "FT", "FB", "FR", "FL"] (do not modify)
* MidddlePixelBorderHorizontal – This selects the number of horizontal border pixels to remove for VSANS, 3 or 4 work well (default 4).
* MidddlePixelBorderVertical -- This selects the number of vertical border pixels to remove for VSANS, 4 (default) works well.
* Slices – selects slices to view: Default: ["Vert", "Horz", "Diag", "Circ"]

* High Res Detector is linked to then Converging Beam option (at 6.7 angstroms). These will change when new high-resolution detector is installed!
* HighResMinX = 240 #Default 240
* HighResMaxX = 474 #Default 474
* HighResMinY = 667 #Default 667
* HighResMaxY = 917 #Default 917
* ConvertHighResToSubset = 1 #Default = 1 for yes (uses only a small subset of the million plus pixels for approximately an 18 x's savings in computing power).
* HighResGain = 100.0

* UsePolCorr = 1 pol-corrects full-pol data, 0 means no and will only correct for 3He transmission as a function of time.
* He3CorrectionType -- Select 1. 0 for chi, 1 for chi = upsilon, 2 for upsilon where these correspond to where the depolarization happens w.r.t the sample.
* YesNoBypassBestGuessPSM  -- The supermirror polarization has been measured at  0.985. Selecting 1 will bypass to higher (or the highest) PSM value if one (or more) is/are measured; 0 will not the the measured polarization go higher than 0.985.
* PSM_Guess -- 0.9985 is good for 4 guides, 5.5 angstroms VSANS (i.e. works for 5.5 angstroms and higher). We should remeasure this for NG7 SANS.
* Minimum_PSM – sets the minimum polarization that will be used for polarization correction.
* YesNoManualHe3Entry – This asks whether you need to manually enter in the He3 polarization information given below. 0 for no (default), 1 for yes. This option should not be needed for data taken after July 2019 if He3 cells are properly registered
* New_HE3_Files -- This is only used if YesNoManualHe3Entry = 1. These would be the starting He3 files for each new cell IF YesNoManualHe3Entry = 1
* MuValues – This is only used if YesNoManualHe3Entry = 1. These are He3 polarization values, dependent upon each cell (check with He3 team if in doubt). This should not typically be needed after July 2019.
* TeValues -- This is only used if YesNoManualHe3Entry = 1. These are He3 glass transmission values, dependent upon each cell (check with He3 team if in doubt). This should not typically be needed after July 2019.
