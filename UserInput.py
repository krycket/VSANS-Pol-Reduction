input_path = r'C:/Users/klkry/Desktop/ExchangeConst_27762_July2020/'
save_path = r'C:/Users/klkry/Desktop/ExchangeConst_27762_July2020/Results/'

Excluded_Filenumbers =[65208, 65212] #Default is []; Be sure to exclude any ConvergingBeam / HighResolutionDetector scans which are not run for the ful default amount of time.#Default is []; Be sure to exclude any ConvergingBeam / HighResolutionDetector scans which are not run for the ful default amount of time.
ReAssignBlockBeam = [] #Default is []
ReAssignEmpty = [] #Default is []
ReAssignOpen = [] #Default is []
YesNoRenameEmpties = 1 #0 = no (default); 1 = yes and will simply rename to Empty; use with caution

TransPanel = 'MR' #Default is 'MR'
SectorCutAngles = 20.0 #Default is typically 10.0 to 20.0 (degrees)
Slices = ["Vert", "Horz", "Circ"] #Default: ["Vert", "Horz", "Diag", "Circ"]
AutoSubtractEmpty = 0 #Default is 1 for yes; 0 for no. Selectiong 1 doesn't cause any issues even if no empties are available.
UseMTCirc = 0 #Default is 1 for yes, 0 for no (which subtracts sector-by-sector MT from data)

Absolute_Q_min = 0.001 #Default 0; Will take the maximum of Q_min_Calc from all detectors and this value
Absolute_Q_max = 0.13 #Default 0.6; Will take the minimum of Q_max_Calc from all detectors and this value
YesNoShowPlots = 0 #0 = No and simply saves plots; 1 = yes and displays plots when code is run
YesNoSetPlotXRange = 0 #Default is 0 (no), 1 = yes
YesNoSetPlotYRange = 0 #Default is 0 (no), 1 = yes
PlotXmin = 0.00023 #Only used if YesNoSetPlotXRange = 1
PlotXmax = 0.12 #Only used if YesNoSetPlotXRange = 1
PlotYmin = 0.001 #Only used if YesNoSetPlotYRange = 1
PlotYmax = 1000000 #Only used if YesNoSetPlotYRange = 1
ComparePolUnpolCircSums = 1
CompareCircSums = 1
CompareMPerpSF = 1
CompareMParlNSF = 1

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
He3CorrectionType = 1 #0 for chi, 1 for chi = upsilon (only active if YesNoManualHe3Entry = 1)
YesNoBypassBestGuessPSM = 1 #Default is 1, will bypass to higher (or the highest) PSM value if one (or more) is/are measured
PSM_Guess = 0.9985 #0.9985 is good for 4 guides, 5.5 angstroms
YesNoBypassBestGuessPSM = 1 #Default is 1, will bypass to higher (or the highest) PSM value if one (or more) is/are measured
Minimum_PSM = 0.50
YesNoManualHe3Entry = 0 #0 for no (default), 1 for yes; should not be needed for data taken after July 2019 if He3 cells are properly registered
New_HE3_Files = [28422, 28498, 28577, 28673, 28755, 28869] #Default is []; These would be the starting files for each new cell IF YesNoManualHe3Entry = 1
MuValues = [3.105, 3.374, 3.105, 3.374, 3.105, 3.374] #Default is []; Values only used IF YesNoManualHe3Entry = 1; example [3.374, 3.105]=[Fras, Bur]; should not be needed after July 2019
TeValues = [0.86, 0.86, 0.86, 0.86, 0.86, 0.86] #Default is []; Values only used IF YesNoManualHe3Entry = 1; example [0.86, 0.86]=[Fras, Bur]; should not be needed after July 2019
