#This files refers to VSANS experiment 27633, July 2020 taken on summer school data.
input_path = r'C:/Users/klkry/Desktop/SumSch_27633_July2020'
save_path = r'C:/Users/klkry/Desktop/SumSch_27633_July2020/Results/'

Excluded_Filenumbers = [] #Default is []; Be sure to exclude any ConvergingBeam / HighResolutionDetector scans which are not run for the ful default amount of time.#Default is []; Be sure to exclude any ConvergingBeam / HighResolutionDetector scans which are not run for the ful default amount of time.
ReAssignBlockBeam = [] #Default is []
ReAssignEmpty = [] #Default is []
ReAssignOpen = [] #Default is []
YesNoRenameEmpties = 1 #0 = no (default); 1 = yes and will simply rename to Empty; use with caution
Min_Filenumber = 64954 #Defaukt is 0
Max_Filenumber = 1000000 #Default is 1000000
Min_Scatt_Filenumber = Min_Filenumber
Max_Scatt_Filenumber = Max_Filenumber
Min_Trans_Filenumber = Min_Filenumber 
Max_Trans_Filenumber = Max_Filenumber

TransPanel = 'MR' #Default is 'MR'
SectorCutAngles = 20.0 #Default is typically 10.0 to 20.0 (degrees)
Slices = ["Vert", "Horz", "Diag", "Circ"] #Default: ["Vert", "Horz", "Diag", "Circ"]
AutoSubtractEmpty = 1 #Default is 1 for yes; 0 for no. Selectiong 1 doesn't cause any issues even if no empties are available.
UseMTCirc = 0 #Default is 1 for yes, 0 for no (which subtracts sector-by-sector MT from data)
Calc_Q_From_Trans = 0
TempDiffAllowedForSharingTrans = 51.0
AverageQRanges = 1 #Default is 1; 0 for no

Absolute_Q_min = 0.003 #Default 0; Will take the maximum of Q_min_Calc from all detectors and this value
Absolute_Q_max = 0.11 #Default 0.6; Will take the minimum of Q_max_Calc from all detectors and this value
YesNoShowPlots = 0 #0 = No and simply saves plots; 1 = yes and displays plots when code is run
YesNoSetPlotXRange = 0 #Default is 0 (no), 1 = yes
YesNoSetPlotYRange = 0 #Default is 0 (no), 1 = yes
PlotXmin = 0.00023 #Only used if YesNoSetPlotXRange = 1
PlotXmax = 0.12 #Only used if YesNoSetPlotXRange = 1
PlotYmin = 1E-6 #Only used if YesNoSetPlotYRange = 1
PlotYmax = 10E4 #Only used if YesNoSetPlotYRange = 1
CompareFullPolSumCirc = 1
CompareHalfPolSumCirc = 1
CompareUnpolCirc = 1
CompareFullPolTypes = 1

YesNo_2DCombinedFiles = 0 #Default is 0 (no), 1 = yes which can be read using SasView
YesNo_2DFilesPerDetector = 0 #Default is 0 (no), 1 = yes; Note all detectors will be summed after beamline masking applied and can be read by SasView 4.2.2 (and higher?)

#High Res Detector is linked to then Converging Beam option (at 6.7 angstroms)
#UseHighResDetWhenAvailable = 0 #0 = No, 1 = Yes
HighResMinX = 240 #Default 240
HighResMaxX = 474 #Default 474
HighResMinY = 667 #Default 667
HighResMaxY = 917 #Default 917
ConvertHighResToSubset = 1 #Default = 1 for yes (uses only a small subset of the million plus pixels for approximately an 18 x's savings in computing power).
HighResGain = 100.0 #320 for EarlyJanuary2020; 100 for LateJanuary2020

UsePolCorr = 1 #Default is 1 to pol-correct full-pol data, 0 means no and will only correct for 3He transmission as a function of time.
He3CorrectionType = 1 #0 for chi, 1 for chi = upsilon (only active if YesNoManualHe3Entry = 1), 2 for upsilon
YesNoBypassBestGuessPSM = 0 #Default is 1, will bypass to higher (or the highest) PSM value if one (or more) is/are measured
PSM_Guess = 0.9985 #0.9985 is good for 4 guides, 5.5 angstroms
Minimum_PSM = 0.01
YesNoManualHe3Entry = 0 #0 for no (default), 1 for yes; should not be needed for data taken after July 2019 if He3 cells are properly registered
New_HE3_Files = [38139] #Default is []; These would be the starting files for each new cell IF YesNoManualHe3Entry = 1
MuValues = [3.105] #Default is []; Values only used IF YesNoManualHe3Entry = 1; example [3.374, 3.105]=[Fras, Bur]; should not be needed after July 2019
TeValues = [0.86] #Default is []; Values only used IF YesNoManualHe3Entry = 1; example [0.86, 0.86]=[Fras, Bur]; should not be needed after July 2019
