#This is set to for summer school 2020 VSANS data.
input_path = r'C:/Users/klkry/Desktop/VSANS27633_CoFe2O4NP_Data'
save_path = input_path +'/ReducedData/'

He3Only_Check = 0 #Default 0 = No (runs full reduction), 1 = Yes (for helium team's use)
SectorCutAngles = 20.0 #Default is typically 10.0 to 20.0 (degrees)
Absolute_Q_min = 0.0035 #Default 0; Will take the maximum of Q_min_Calc from all detectors and this value
Absolute_Q_max = 0.12 #Default 0.6; Will take the minimum of Q_max_Calc from all detectors and this value

AutoSubtractEmpty = 1 #Default is 1 for yes; 0 for no. Selecting 1 doesn't cause any issues even if no empties are available.
YesNoRenameEmpties = 1 #0 = No; 1 = Yes and will simply rename to Empty
UseMTCirc = 1 #Default is 1 for yes, 0 for no (which instead subtracts sector-by-sector MT from data)
TempDiffAllowedForSharingTrans = 51.0 #Max temperature difference in K to fill in for missing transmission files

Excluded_Filenumbers = [] #[Use filenumers separated by commas as needed]
ReAssignBlockBeam = []
ReAssignEmpty = []
ReAssignOpen = []
ReAssignSample = []
Min_Filenumber = 64954 #Default 0
Max_Filenumber =  1000000 #Default 1000000
Min_Scatt_Filenumber = Min_Filenumber
Max_Scatt_Filenumber = Max_Filenumber
Min_Trans_Filenumber = Min_Scatt_Filenumber
Max_Trans_Filenumber = Max_Scatt_Filenumber
SampleDescriptionKeywordsToExclude = ['Align', 'Check']

YesNoSetPlotXRange = 0 #Default is 0 (no), 1 = yes
YesNoSetPlotYRange = 0 #Default is 0 (no), 1 = yes
PlotXmin = 0.00032 #Only used if YesNoSetPlotXRange = 1
PlotXmax = 0.12 #Only used if YesNoSetPlotXRange = 1
PlotYmin = 1E-6 #Only used if YesNoSetPlotYRange = 1
PlotYmax = 5E3 #Only used if YesNoSetPlotYRange = 1

#********************************************************************
#**** Here on should rarely need to be adjusted *********************
#********************************************************************
PreSebtractOpen = 0 #Default is 0 for no; 1 for yes. Subtracts trans-scaled open (if available) from pol-full in attempt to remove main beam spillover.
Calc_Q_From_Trans = 1 #Default is 1 for yes; 0 for no
AverageQRanges = 1 #0 for no; 1 for yes

YesNoShowPlots = 0 #0 = No and simply saves plots; 1 = yes and displays plots when code is run
CompareUnpolCirc = 1
CompareHalfPolSumCirc = 1
CompareFullPolSumCirc = 1
CompareFullPolStruc = 1
CompareFullPolMagnetism = 1
YesNo_2DCombinedFiles = 0 #Default is 0 (no), 1 = yes which can be read using SasView
YesNo_2DFilesPerDetector = 0 #Default is 0 (no), 1 = yes; Note all detectors will be summed after beamline masking applied and can be read by SasView 4.2.2 (and higher?)

TransPanel = 'MR' #Default is 'MR'
all_detectors = ["B", "MT", "MB", "MR", "ML", "FT", "FB", "FR", "FL"]
nonhighres_detectors = ["MT", "MB", "MR", "ML", "FT", "FB", "FR", "FL"]
MidddlePixelBorderHorizontal = 4 #Default = 3 or 4
MidddlePixelBorderVertical = 4 #Default = 4
Slices = ["Vert", "Horz", "Diag", "Circ"] #Default: ["Vert", "Horz", "Diag", "Circ"]

#High Res Detector is linked to then Converging Beam option (at 6.7 angstroms)
HighResMinX = 240 #Default 240
HighResMaxX = 474 #Default 474
HighResMinY = 667 #Default 667
HighResMaxY = 917 #Default 917
ConvertHighResToSubset = 1 #Default = 1 for yes (uses only a small subset of the million plus pixels for approximately an 18 x's savings in computing power).
HighResGain = 100.0

UsePolCorr = 1 #Default is 1 to pol-correct full-pol data, 0 means no and will only correct for 3He transmission as a function of time.
He3CorrectionType = 1 #0 for chi, 1 for chi = upsilon (only active if YesNoManualHe3Entry = 1), 2 for upsilon
YesNoBypassBestGuessPSM = 0 #Default is 1, will bypass to higher (or the highest) PSM value if one (or more) is/are measured
PSM_Guess = 0.9985 #0.9985 is good for 4 guides, 5.5 angstroms
Minimum_PSM = 0.01
YesNoManualHe3Entry = 0 #0 for no (default), 1 for yes; should not be needed for data taken after July 2019 if He3 cells are properly registered
New_HE3_Files = [] #Default is []; These would be the starting files for each new cell IF YesNoManualHe3Entry = 1
MuValues = [] #Default is []; Values only used IF YesNoManualHe3Entry = 1; example [3.374, 3.105]=[Fras, Bur]; should not be needed after July 2019
TeValues = [] #Default is []; Values only used IF YesNoManualHe3Entry = 1; example [0.86, 0.86]=[Fras, Bur]; should not be needed after July 2019
