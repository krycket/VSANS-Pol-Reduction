import numpy as np
import h5py
import scipy as sp
from scipy.optimize.minpack import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import dateutil
import datetime
from numpy.linalg import inv
from uncertainties import unumpy
import os

Scatt_filenumber = 95171
Trans_filenumber = 95022

Excluded_Filenumbers = [] #Default is []; Be sure to exclude any ConvergingBeam / HighResolutionDetector scans which are not run for the ful default amount of time.
ReAssignBlockBeam = [] #Default is []
ReAssignEmpty = [] #Default is []

YesNoManualHe3Entry = 0 #0 for no (default), 1 for yes; should not be needed for data taken after July 2019 if He3 cells are properly registered
New_HE3_Files = [28422, 28498, 28577, 28673, 28755, 28869] #Default is []; These would be the starting files for each new cell IF YesNoManualHe3Entry = 1
MuValues = [3.105, 3.374, 3.105, 3.374, 3.105, 3.374] #Default is []; Values only used IF YesNoManualHe3Entry = 1; example [3.374, 3.105]=[Fras, Bur]; should not be needed after July 2019
TeValues = [0.86, 0.86, 0.86, 0.86, 0.86, 0.86] #Default is []; Values only used IF YesNoManualHe3Entry = 1; example [0.86, 0.86]=[Fras, Bur]; should not be needed after July 2019

path = ''

def NG7SANS_Config_ID(filenumber):

    Configuration_ID = "Unknown"
    filename = path + "sans" + str(filenumber) + ".nxs.ng7"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        Desired_Distance = int(f['entry/DAS_logs/detectorPosition/desiredSoftPosition'][0]) #in cm
        WV = str(f['entry/DAS_logs/wavelength/wavelength'][0])
        Wavelength = WV[:3]
        GuideHolder = f['entry/DAS_logs/guide/guide'][0]
        if str(GuideHolder).find("CONV") != -1:
            Guides =  "CvB"
        else:
            GuideNum = int(f['entry/DAS_logs/guide/guide'][0])
            Guides = str(GuideNum)
            
        Configuration_ID = str(Guides) + "Gd" + str(Desired_Distance) + "cm" + str(Wavelength) + "Ang"
        
    return Configuration_ID

def NG7SANS_SortData(YesNoManualHe3Entry, New_HE3_Files, MuValues, TeValues):
    BlockBeam = {}
    Configs = {}
    Sample_Names = {}
    Scatt = {}
    Trans = {}
    Pol_Trans = {}
    HE3_Trans = {}
    FileNumberList = [0]
    UU_filenumber = -10
    DU_filenumber = -10
    DD_filenumber = -10
    UD_filenumber = -10
    filenames = '0'
    record_adam4021 = 0
    record_temp = 0
    CellIdentifier = 0
    HE3OUT_filenumber = -10
    start_number = 0
    filelist = [fn for fn in os.listdir("./") if fn.endswith(".nxs.ng7")] #or filenames = [fn for fn in os.listdir("./") if os.path.isfile(fn)]
    if len(filelist) >= 1:
        for name in filelist:
            filename = str(name)
            filenumber = int(filename[4:9])
            if start_number == 0:
                start_number = filenumber
            config = Path(filename)
            if config.is_file():
                f = h5py.File(filename)
                Count_time = f['entry/collection_time'][0]
                Descrip = str(f['entry/sample/description'][0])
                Descrip = Descrip[2:]
                Descrip = Descrip[:-1]
                if Count_time > 29 and str(Descrip).find("Align") == -1 and filenumber not in Excluded_Filenumbers:
                    FileNumberList.append(filenumber)
                    print('Reading:', filenumber, ' ', Descrip)
                    Listed_Config = str(f['entry/DAS_logs/configuration/key'][0])
                    Listed_Config = Listed_Config[2:]
                    Listed_Config = Listed_Config[:-1]
                    Sample_Name = Descrip.replace(Listed_Config, '')
                    Not_Sample = ['T_UU', 'T_DU', 'T_DD', 'T_UD', 'T_SM', 'T_NP', 'HeIN', 'HeOUT', 'S_UU', 'S_DU', 'S_DD', 'S_UD', 'S_NP', 'S_HeU', 'S_HeD', 'S_SMU', 'S_SMD']
                    for i in Not_Sample:
                        Sample_Name = Sample_Name.replace(i, '')
                    Desired_Temp = 'na'
                    if "temp" in f['entry/DAS_logs/']:
                        Desired_Temp = str(f['entry/DAS_logs/temp/desiredPrimaryNode'][(0)])
                        record_temp = 1    
                    Voltage = 'na'
                    if "adam4021" in f['entry/DAS_logs/']:
                        Voltage = str(f['entry/DAS_logs/adam4021/voltage'][(0)])
                        record_adam4021 = 1
                    DT5 = Desired_Temp + " K,"
                    DT4 = Desired_Temp + " K"
                    DT3 = Desired_Temp + "K,"
                    DT2 = Desired_Temp + "K"
                    DT1 = Desired_Temp
                    V5 = Voltage + " V,"
                    V4 = Voltage + " V"
                    V3 = Voltage + "V,"
                    V2 = Voltage + "V"
                    V1 = Voltage
                    Not_Sample = [DT5, DT4, DT3, DT2, DT1, V5, V4, V3, V2, V1]
                    for i in Not_Sample:
                        Sample_Name = Sample_Name.replace(i, '')
                    Sample_Name = Sample_Name.replace(' ', '')
                    Sample_Base = Sample_Name
                    Sample_Name = Sample_Name + '_' + str(Voltage) + 'V_' + str(Desired_Temp) + 'K'
                    #Purpose = f['entry/reduction/file_purpose'][()] #SCATT, TRANS, HE3 on VSANS
                    Purpose = 'SCATT'
                    if str(Listed_Config).find("TRANS") != -1 or  str(Listed_Config).find("Trans") != -1:
                        Purpose = 'TRANS'
                    if str(Listed_Config).find("HE3") != -1:
                        Purpose = 'HE3'
                    #Intent = f['entry/reduction/intent'][()] #Sample, Empty, Blocked Beam, Open Beam on VSANS
                    Intent = 'Sample'
                    if str(Descrip).find("Empty") != -1 or str(Descrip).find("EMPTY") != -1:
                        Intent = 'Empty'
                    if str(Descrip).find("Open") != -1 or str(Descrip).find("OPEN") != -1:
                        Intent = 'Open'
                    if str(Descrip).find("HE3") != -1:
                        Purpose = 'HE3'
                    if str(Descrip).find("Block") != -1 or str(Descrip).find("BLOCK") != -1:
                        Intent = 'Blocked Beam'
                    Descrip = Descrip[:-1]
                    if filenumber in ReAssignBlockBeam:
                        Intent = 'Blocked Beam'
                    if filenumber in ReAssignEmpty:
                        Intent = 'Empty'
                    Type = str(f['entry/sample/description'][()])
                    End_time = dateutil.parser.parse(f['entry/end_time'][0])
                    TimeOfMeasurement = (End_time.timestamp() - Count_time/2)/3600.0 #in hours
                    data = np.array(f['entry/instrument/detector/data'])
                    Trans_Counts = np.sum(data)
                    MonCounts = f['entry/control/monitor_counts'][0]
                    Trans_Distance = f['entry/instrument/detector/distance'][0]
                    #Attenuation = f['entry/DAS_logs/attenuator/attenuator'][0] (VSANS)
                    Attenuation = f['entry/DAS_logs/attenuator/key'][0]
                    Wavelength = f['entry/DAS_logs/wavelength/wavelength'][0]
                    Config = NG7SANS_Config_ID(filenumber)
                    if "frontPolarization" in f['entry/DAS_logs/']:
                        FrontPolDirection = f['entry/DAS_logs/frontPolarization/direction'][()]
                    else:
                        FrontPolDirection = [b'UNPOLARIZED']
                    if "backPolarization" in f['entry/DAS_logs/']:
                        BackPolDirection = f['entry/DAS_logs/backPolarization/direction'][()]
                    else:
                        BackPolDirection = [b'UNPOLARIZED']

                    GuideHolder = f['entry/DAS_logs/guide/guide'][0]
                    if str(GuideHolder).find("CONV") == -1:
                        if int(GuideHolder) == 0:
                            FrontPolDirection = [b'UNPOLARIZED']
                    '''Want to populate Config representative filenumbers on scattering filenumber'''
                    config_filenumber = 0
                    if str(Purpose).find("SCATT") != -1:
                        config_filenumber = filenumber
                    if Config not in Configs:
                        Configs[Config] = config_filenumber
                    if Configs[Config] == 0 and config_filenumber != 0:
                        Configs[Config] = config_filenumber
                    if Config not in BlockBeam:
                        BlockBeam[Config] = {'Scatt':{'File' : 'NA'}, 'Trans':{'File' : 'NA', 'CountsPerSecond' : 'NA'}}
                    if str(Intent).find("Blocked") != -1:
                        if Config not in BlockBeam:
                             BlockBeam[Config] = {'Scatt':{'File' : 'NA'}, 'Trans':{'File' : 'NA', 'CountsPerSecond' : 'NA'}}
                        if str(Purpose).find("TRANS") != -1 or str(Purpose).find("HE3") != -1:
                            if 'NA' in BlockBeam[Config]['Trans']['File']:
                                BlockBeam[Config]['Trans']['File'] = [filenumber]
                                BlockBeam[Config]['Trans']['CountsPerSecond'] = [Trans_Counts/Count_time]
                            else:
                                BlockBeam[Config]['Trans']['File'].append(filenumber)
                                BlockBeam[Config]['Trans']['CountsPerSecond'].append(Trans_Counts/Count_time)
                        elif str(Purpose).find("SCATT") != -1:
                            if 'NA' in BlockBeam[Config]['Scatt']['File']:
                                BlockBeam[Config]['Scatt']['File'] = [filenumber]
                            else:
                                BlockBeam[Config]['Scatt']['File'].append(filenumber)
                    elif str(Intent).find("Sample") != -1 or str(Intent).find("Empty") != -1 or str(Intent).find("Open") != -1:
                        if len(Sample_Names) < 1:
                            Sample_Names = [Sample_Name]
                        else:
                            if Sample_Name not in Sample_Names:
                                Sample_Names.append(Sample_Name)
                        Intent_short = str(Intent)
                        #Intent_short = Intent_short[3:-2] (VSANS)
                        Intent_short = Intent_short.replace(' Cell', '')
                        Intent_short = Intent_short.replace(' Beam', '')
                        if str(Purpose).find("SCATT") != -1:
                            if Sample_Name not in Scatt:
                                Scatt[Sample_Name] = {'Intent': Intent_short, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'Unpol': 'NA', 'U' : 'NA', 'D' : 'NA','UU' : 'NA', 'DU' : 'NA', 'DD' : 'NA', 'UD' : 'NA', 'UU_Time' : 'NA', 'DU_Time' : 'NA', 'DD_Time' : 'NA', 'UD_Time' : 'NA'}}}
                            if Config not in Scatt[Sample_Name]['Config(s)']:
                                Scatt[Sample_Name]['Config(s)'][Config] = {'Unpol': 'NA', 'U' : 'NA', 'D' : 'NA','UU' : 'NA', 'DU' : 'NA', 'DD' : 'NA', 'UD' : 'NA', 'UU_Time' : 'NA', 'DU_Time' : 'NA', 'DD_Time' : 'NA', 'UD_Time' : 'NA'}
                            if str(FrontPolDirection).find("UNPOLARIZED") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['Unpol']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['Unpol'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['Unpol'].append(filenumber)
                            if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['U']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['U'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['U'].append(filenumber)
                            if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['D']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['D'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['D'].append(filenumber)
                            if YesNoManualHe3Entry != 1:        
                                if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UP") != -1:
                                    if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['UU']:
                                        Scatt[Sample_Name]['Config(s)'][Config]['UU'] = [filenumber]
                                        Scatt[Sample_Name]['Config(s)'][Config]['UU_Time'] = [TimeOfMeasurement]
                                    else:
                                        Scatt[Sample_Name]['Config(s)'][Config]['UU'].append(filenumber)
                                        Scatt[Sample_Name]['Config(s)'][Config]['UU_Time'].append(TimeOfMeasurement)                                       
                                if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UP") != -1:
                                    if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['DU']:
                                        Scatt[Sample_Name]['Config(s)'][Config]['DU'] = [filenumber]
                                        Scatt[Sample_Name]['Config(s)'][Config]['DU_Time'] = [TimeOfMeasurement]
                                    else:
                                        Scatt[Sample_Name]['Config(s)'][Config]['DU'].append(filenumber)
                                        Scatt[Sample_Name]['Config(s)'][Config]['DU_Time'].append(TimeOfMeasurement)

                                if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                    if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['DD']:
                                        Scatt[Sample_Name]['Config(s)'][Config]['DD'] = [filenumber]
                                        Scatt[Sample_Name]['Config(s)'][Config]['DD_Time'] = [TimeOfMeasurement]
                                    else:
                                        Scatt[Sample_Name]['Config(s)'][Config]['DD'].append(filenumber)
                                        Scatt[Sample_Name]['Config(s)'][Config]['DD_Time'].append(TimeOfMeasurement)
                                        
                                if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                    if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['UD']:
                                        Scatt[Sample_Name]['Config(s)'][Config]['UD'] = [filenumber]
                                        Scatt[Sample_Name]['Config(s)'][Config]['UD_Time'] = [TimeOfMeasurement]
                                    else:
                                        Scatt[Sample_Name]['Config(s)'][Config]['UD'].append(filenumber)
                                        Scatt[Sample_Name]['Config(s)'][Config]['UD_Time'].append(TimeOfMeasurement)
                            else:
                                if Type[-6:-2] == 'S_UU': #str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UP") != -1:
                                    if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['UU']:
                                        Scatt[Sample_Name]['Config(s)'][Config]['UU'] = [filenumber]
                                        Scatt[Sample_Name]['Config(s)'][Config]['UU_Time'] = [TimeOfMeasurement]
                                    else:
                                        Scatt[Sample_Name]['Config(s)'][Config]['UU'].append(filenumber)
                                        Scatt[Sample_Name]['Config(s)'][Config]['UU_Time'].append(TimeOfMeasurement)
                                        
                                if Type[-6:-2] == 'S_DU': #str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UP") != -1:
                                    if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['DU']:
                                        Scatt[Sample_Name]['Config(s)'][Config]['DU'] = [filenumber]
                                        Scatt[Sample_Name]['Config(s)'][Config]['DU_Time'] = [TimeOfMeasurement]
                                    else:
                                        Scatt[Sample_Name]['Config(s)'][Config]['DU'].append(filenumber)
                                        Scatt[Sample_Name]['Config(s)'][Config]['DU_Time'].append(TimeOfMeasurement)

                                if Type[-6:-2] == 'S_DD': #str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                    if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['DD']:
                                        Scatt[Sample_Name]['Config(s)'][Config]['DD'] = [filenumber]
                                        Scatt[Sample_Name]['Config(s)'][Config]['DD_Time'] = [TimeOfMeasurement]
                                    else:
                                        Scatt[Sample_Name]['Config(s)'][Config]['DD'].append(filenumber)
                                        Scatt[Sample_Name]['Config(s)'][Config]['DD_Time'].append(TimeOfMeasurement)
                                        
                                if Type[-6:-2] == 'S_UD': #str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                    if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['UD']:
                                        Scatt[Sample_Name]['Config(s)'][Config]['UD'] = [filenumber]
                                        Scatt[Sample_Name]['Config(s)'][Config]['UD_Time'] = [TimeOfMeasurement]
                                    else:
                                        Scatt[Sample_Name]['Config(s)'][Config]['UD'].append(filenumber)
                                        Scatt[Sample_Name]['Config(s)'][Config]['UD_Time'].append(TimeOfMeasurement)     
                        if str(Purpose).find("TRANS") != -1:
                            if Sample_Name not in Trans:
                                Trans[Sample_Name] = {'Intent': Intent_short, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files' : 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}}}
                            if Config not in Trans[Sample_Name]['Config(s)']:
                                Trans[Sample_Name]['Config(s)'][Config] = {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files': 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}
                            if Sample_Name not in Pol_Trans:
                                Pol_Trans[Sample_Name] = {'T_UU' : {'File' : 'NA'},
                                                          'T_DU' : {'File' : 'NA'},
                                                          'T_DD' : {'File' : 'NA'},
                                                          'T_UD' : {'File' : 'NA'},
                                                          'T_SM' : {'File' : 'NA'},
                                                          'Config' : 'NA'
                                                          }
                            if str(FrontPolDirection).find("UNPOLARIZED") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files']:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files'].append(filenumber)
                            if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['U_Files']:
                                    Trans[Sample_Name]['Config(s)'][Config]['U_Files'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['U_Files'].append(filenumber)
                            if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['D_Files']:
                                    Trans[Sample_Name]['Config(s)'][Config]['D_Files'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['D_Files'].append(filenumber)
                            if YesNoManualHe3Entry != 1:
                                if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UP") != -1:
                                    UU_filenumber = filenumber
                                    UU_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UP") != -1:
                                    DU_filenumber = filenumber
                                    DU_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                    DD_filenumber = filenumber
                                    DD_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                    UD_filenumber = filenumber
                                    UD_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                    SM_filenumber = filenumber
                                    if SM_filenumber - UU_filenumber == 4:
                                        if 'NA' in Pol_Trans[Sample_Name]['T_UU']['File']:
                                            Pol_Trans[Sample_Name]['T_UU']['File'] = [UU_filenumber]
                                            Pol_Trans[Sample_Name]['T_UU']['Meas_Time'] = [UU_Time]
                                        else:
                                            Pol_Trans[Sample_Name]['T_UU']['File'].append(UU_filenumber)
                                            Pol_Trans[Sample_Name]['T_UU']['Meas_Time'].append(UU_Time)
                                        if 'NA' in Pol_Trans[Sample_Name]['T_DU']['File']:
                                            Pol_Trans[Sample_Name]['T_DU']['File'] = [DU_filenumber]
                                            Pol_Trans[Sample_Name]['T_DU']['Meas_Time'] = [DU_Time]
                                        else:
                                            Pol_Trans[Sample_Name]['T_DU']['File'].append(DU_filenumber)
                                            Pol_Trans[Sample_Name]['T_DU']['Meas_Time'].append(DU_Time)
                                        if 'NA' in Pol_Trans[Sample_Name]['T_DD']['File']:
                                            Pol_Trans[Sample_Name]['T_DD']['File'] = [DD_filenumber]
                                            Pol_Trans[Sample_Name]['T_DD']['Meas_Time'] = [DD_Time]
                                        else:
                                            Pol_Trans[Sample_Name]['T_DD']['File'].append(DD_filenumber)
                                            Pol_Trans[Sample_Name]['T_DD']['Meas_Time'].append(DD_Time)
                                        if 'NA' in Pol_Trans[Sample_Name]['T_UD']['File']:
                                            Pol_Trans[Sample_Name]['T_UD']['File'] = [UD_filenumber]
                                            Pol_Trans[Sample_Name]['T_UD']['Meas_Time'] = [UD_Time]
                                        else:
                                            Pol_Trans[Sample_Name]['T_UD']['File'].append(UD_filenumber)
                                            Pol_Trans[Sample_Name]['T_UD']['Meas_Time'].append(UD_Time)
                                        if 'NA' in Pol_Trans[Sample_Name]['T_SM']['File']:
                                            Pol_Trans[Sample_Name]['T_SM']['File'] = [SM_filenumber]
                                        else:
                                            Pol_Trans[Sample_Name]['T_SM']['File'].append(SM_filenumber)
                                        if 'NA' in Pol_Trans[Sample_Name]['Config']:
                                            Pol_Trans[Sample_Name]['Config'] = [Config]
                                        else:
                                            Pol_Trans[Sample_Name]['Config'].append(Config)
                            else:
                                if Type[-6:-2] == 'T_UU': #str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UP") != -1:
                                    UU_filenumber = filenumber
                                    UU_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                if Type[-6:-2] == 'T_DU': #str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UP") != -1:
                                    DU_filenumber = filenumber
                                    DU_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                if Type[-6:-2] == 'T_DD': #str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                    DD_filenumber = filenumber
                                    DD_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                if Type[-6:-2] == 'T_UD': #str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                    UD_filenumber = filenumber
                                    UD_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                if Type[-6:-2] == 'T_SM': #str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                    SM_filenumber = filenumber
                                    if SM_filenumber - UU_filenumber == 4:
                                        if 'NA' in Pol_Trans[Sample_Name]['T_UU']['File']:
                                            Pol_Trans[Sample_Name]['T_UU']['File'] = [UU_filenumber]
                                            Pol_Trans[Sample_Name]['T_UU']['Meas_Time'] = [UU_Time]
                                        else:
                                            Pol_Trans[Sample_Name]['T_UU']['File'].append(UU_filenumber)
                                            Pol_Trans[Sample_Name]['T_UU']['Meas_Time'].append(UU_Time)
                                        if 'NA' in Pol_Trans[Sample_Name]['T_DU']['File']:
                                            Pol_Trans[Sample_Name]['T_DU']['File'] = [DU_filenumber]
                                            Pol_Trans[Sample_Name]['T_DU']['Meas_Time'] = [DU_Time]
                                        else:
                                            Pol_Trans[Sample_Name]['T_DU']['File'].append(DU_filenumber)
                                            Pol_Trans[Sample_Name]['T_DU']['Meas_Time'].append(DU_Time)
                                        if 'NA' in Pol_Trans[Sample_Name]['T_DD']['File']:
                                            Pol_Trans[Sample_Name]['T_DD']['File'] = [DD_filenumber]
                                            Pol_Trans[Sample_Name]['T_DD']['Meas_Time'] = [DD_Time]
                                        else:
                                            Pol_Trans[Sample_Name]['T_DD']['File'].append(DD_filenumber)
                                            Pol_Trans[Sample_Name]['T_DD']['Meas_Time'].append(DD_Time)
                                        if 'NA' in Pol_Trans[Sample_Name]['T_UD']['File']:
                                            Pol_Trans[Sample_Name]['T_UD']['File'] = [UD_filenumber]
                                            Pol_Trans[Sample_Name]['T_UD']['Meas_Time'] = [UD_Time]
                                        else:
                                            Pol_Trans[Sample_Name]['T_UD']['File'].append(UD_filenumber)
                                            Pol_Trans[Sample_Name]['T_UD']['Meas_Time'].append(UD_Time)
                                        if 'NA' in Pol_Trans[Sample_Name]['T_SM']['File']:
                                            Pol_Trans[Sample_Name]['T_SM']['File'] = [SM_filenumber]
                                        else:
                                            Pol_Trans[Sample_Name]['T_SM']['File'].append(SM_filenumber)
                                        if 'NA' in Pol_Trans[Sample_Name]['Config']:
                                            Pol_Trans[Sample_Name]['Config'] = [Config]
                                        else:
                                            Pol_Trans[Sample_Name]['Config'].append(Config)
                        if str(Purpose).find("HE3") != -1:
                            HE3Type = str(f['entry/sample/description'][()])
                            if HE3Type[-7:-2] == 'HeOUT':
                                if Sample_Name not in Trans:
                                    Trans[Sample_Name] = {'Intent': Intent_short, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files' : 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}}}
                                if Config not in Trans[Sample_Name]['Config(s)']:
                                    Trans[Sample_Name]['Config(s)'][Config] = {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files': 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files']:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files'].append(filenumber)
                            if YesNoManualHe3Entry == 1:
                                if filenumber in New_HE3_Files:
                                    ScaledOpacity = MuValues[CellIdentifier]
                                    TE = TeValues[CellIdentifier]
                                    CellTimeIdentifier = (End_time.timestamp() - Count_time)/3600.0
                                    HE3Insert_Time = (End_time.timestamp() - Count_time)/3600.0
                                    CellIdentifier += 1    
                            else:
                                CellTimeIdentifier = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                                CellName = str(f['entry/DAS_logs/backPolarization/name'][0])
                                CellName = CellName[2:]
                                CellName = CellName[:-1]
                                CellName = CellName + str(CellTimeIdentifier)
                                CellTimeIdentifier = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                                if CellTimeIdentifier not in HE3_Trans:
                                    HE3Insert_Time = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                                    Opacity = f['/entry/DAS_logs/backPolarization/opacityAt1Ang'][0]
                                    Wavelength = f['/entry/DAS_logs/wavelength/wavelength'][0]
                                    ScaledOpacity = Opacity*Wavelength
                                    TE = f['/entry/DAS_logs/backPolarization/glassTransmission'][0]
                            if HE3Type[-7:-2] == 'HeOUT':
                                HE3OUT_filenumber = filenumber
                                HE3OUT_config = Config
                                HE3OUT_sample = Sample_Name
                                HE3OUT_attenuators = int(f['entry/instrument/attenuator/num_atten_dropped'][0])
                            elif HE3Type[-7:-2] == ' HeIN':
                                HE3IN_filenumber = filenumber
                                HE3IN_config = Config
                                HE3IN_sample = Sample_Name
                                HE3IN_attenuators = int(f['entry/instrument/attenuator/num_atten_dropped'][0])
                                HE3IN_StartTime = (End_time.timestamp() - Count_time/2)/3600.0
                                if HE3OUT_filenumber > 0:
                                    if HE3OUT_config == HE3IN_config and HE3OUT_attenuators == HE3IN_attenuators and HE3OUT_sample == HE3IN_sample: #This implies that you must have a 3He out before 3He in of same config and atten
                                        if HE3Insert_Time not in HE3_Trans:
                                            HE3_Trans[CellTimeIdentifier] = {'Te' : TE,
                                                                         'Mu' : ScaledOpacity,
                                                                         'Insert_time' : HE3Insert_Time}
                                        Elasped_time = HE3IN_StartTime - HE3Insert_Time
                                        if "Elasped_time" not in HE3_Trans[CellTimeIdentifier]:
                                            HE3_Trans[CellTimeIdentifier]['Config'] = [HE3IN_config]
                                            HE3_Trans[CellTimeIdentifier]['HE3_OUT_file'] = [HE3OUT_filenumber]
                                            HE3_Trans[CellTimeIdentifier]['HE3_IN_file'] = [HE3IN_filenumber]
                                            HE3_Trans[CellTimeIdentifier]['Elasped_time'] = [Elasped_time]
                                            HE3_Trans[CellTimeIdentifier]['Cell_name'] = [CellName]
                                        else:
                                            HE3_Trans[CellTimeIdentifier]['Config'].append(HE3IN_config)
                                            HE3_Trans[CellTimeIdentifier]['HE3_OUT_file'].append(HE3OUT_filenumber)
                                            HE3_Trans[CellTimeIdentifier]['HE3_IN_file'].append(HE3IN_filenumber)
                                            HE3_Trans[CellTimeIdentifier]['Elasped_time'].append(Elasped_time)
                                            HE3_Trans[CellTimeIdentifier]['Cell_name'].append(CellName)
    return Sample_Names, Configs, BlockBeam, Scatt, Trans, Pol_Trans, HE3_Trans, start_number, FileNumberList

def NG7SANS_AttenuatorTable(wavelength, attenuation):

    if attenuation <= 0:
        attn_index = 0
    elif attenuation >= 10:
        attn_index = 10
    else:
        attn_index = int(attenuation)

    if wavelength < 5.0:
        wavelength = 5.0
    if wavelength > 17.0:
        wavelength = 17.0
            
    Attn_Table = {}
    Attn_Table[5.0] = [1.0, 0.418, 0.189, 0.0784, 0.0328, 0.0139, 5.90E-3, 1.04E-3, 1.90E-4, 3.58E-5, 7.76E-6]
    Attn_Table[6.0] = [1.0, 0.393, 0.167, 0.0651, 0.0256, 0.0101, 4.07E-3, 6.37E-4, 1.03E-4, 1.87E-5, 4.56E-6]
    Attn_Table[7.0] = [1.0, 0.369, 0.148, 0.0541, 0.0200, 7.43E-3, 2.79E-3, 3.85E-4, 5.71E-5, 1.05E-5, 3.25E-6]
    Attn_Table[8.0] = [1.0, 0.347, 0.132, 0.0456, 0.0159, 5.58E-3, 1.99E-3, 2.46E-4, 3.44E-5, 7.00E-6, 7.00E-6]
    Attn_Table[10.0] = [1.0, 0.313, 0.109, 0.0340, 0.0107, 3.42E-3, 1.11E-3, 1.16E-4, 1.65E-5, 1.65E-5, 1.65E-5]    
    Attn_Table[12.0] = [1.0, 0.291, 0.0945, 0.0273, 7.98E-3, 2.36E-3, 7.13E-4, 6.86E-5, 6.86E-5, 6.86E-5, 6.86E-5]    
    Attn_Table[14.0] = [1.0, 0.271, 0.0830, 0.0223, 6.14E-3, 1.70E-3, 4.91E-4, 4.91E-4, 4.91E-4, 4.91E-4, 4.91E-4]
    Attn_Table[17.0] = [1.0, 0.244, 0.0681, 0.0164, 4.09E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3]
    Attn_Table[17.001] = [1.0, 0.244, 0.0681, 0.0164, 4.09E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3]

    Wavelength_Min = 5.0
    Wavelength_Max = 5.0
    Max_trip = 0
    for i in Attn_Table:
        if wavelength >= i:
            Wavelength_Min = i
        if wavelength <= i and Max_trip == 0:
            Wavelength_Max = i
            Max_trip = 1
            
    Trans_MinWave = Attn_Table[Wavelength_Min][attn_index]
    Trans_MaxWave = Attn_Table[Wavelength_Max][attn_index]

    if Wavelength_Max > Wavelength_Min:
        Trans = Trans_MinWave + (wavelength - Wavelength_Min)*(Trans_MaxWave - Trans_MinWave)/(Wavelength_Max - Wavelength_Min)
    else:
        Trans = Trans_MinWave

        
    return Trans


def NG7SANS_SolidAngle(filenumber):
    
    filename = path + "sans" + str(filenumber) + ".nxs.ng7"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        Desired_Distance = int(f['entry/DAS_logs/detectorPosition/desiredSoftPosition'][0]) #in cm

        x_pixel_size = f['entry/instrument/detector/x_pixel_size'][0]/10.0  #in cm
        y_pixel_size = f['entry/instrument/detector/y_pixel_size'][0]/10.0  #in cm
                
        realDistZ = Desired_Distance
        theta_x_step = x_pixel_size / realDistZ
        theta_y_step = y_pixel_size / realDistZ
        Solid_Angle = theta_x_step * theta_y_step
        
    return Solid_Angle

def NG7SANS_TransCountsPer1E8MonCounts(filenumber):
    #Uses function NG7SANS_AttenuatorTable
    
    filename = path + "sans" + str(filenumber) + ".nxs.ng7"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        data = np.array(f['entry/instrument/detector/data'])
        monitor_counts = f['entry/control/monitor_counts'][0]
        count_time = f['entry/collection_time'][0]
        abs_trans = np.sum(data)*1E8/monitor_counts
        wavelength = f['entry/DAS_logs/wavelength/wavelength'][0]
        attenuation = f['entry/DAS_logs/attenuator/key'][0]
        attn_trans = NG7SANS_AttenuatorTable(wavelength, attenuation)
        abs_trans = abs_trans/attn_trans
        
    return abs_trans

def NG7SANS_AbsScaleScattData(filenumber, Abs_Trans, Sample_Trans):
    #Uses functions NG7SANS_SolidAngle and NG7SANS_AttenuatorTable
    
    filename = path + "sans" + str(filenumber) + ".nxs.ng7"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        data = np.array(f['entry/instrument/detector/data'])
        data_unc = np.sqrt(data)
        monitor_counts = f['entry/control/monitor_counts'][0]
        count_time = f['entry/collection_time'][0]
        attenuation = f['entry/DAS_logs/attenuator/key'][0]
        attn_trans = 1.0
        if attenuation > 0:
            attn_trans = NG7SANS_AttenuatorTable(wavelength, attenuation)
            print('NOTE: Scatt file', filenumber, 'has', attenuation, 'attenuators')
        solid_angle = NG7SANS_SolidAngle(filenumber)
        sample_thickness = f['entry/sample/thickness'][0]/10.0 #in mm -> cm
        data = data*(1E8/monitor_counts) / (solid_angle*Abs_Trans*sample_thickness*Sample_Trans)
        data_unc = data_unc*(1E8/monitor_counts) / (solid_angle*Abs_Trans*sample_thickness*Sample_Trans)
        
    return data, data_unc

def NG7SANS_QCalculation(filenumber):
    #Need to check that lateral offset works correctly when not set to zero.

    filename = path + "sans" + str(filenumber) + ".nxs.ng7"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        data = np.array(f['entry/instrument/detector/data'])
        monitor_counts = f['entry/control/monitor_counts'][0]
        data_unc = np.sqrt(data)*1E8/monitor_counts
        data = data*1E8/monitor_counts
        dimX = 128
        dimY = 128
        detector_distance = f['entry/instrument/detector/distance'][0] #in cm
        #detector_distance = f['entry/instrument/detectorPosition/softPosition'][0] #in cm
        x_pixel_size = f['entry/instrument/detector/x_pixel_size'][0]/10.0
        y_pixel_size = f['entry/instrument/detector/y_pixel_size'][0]/10.0
        beam_center_x = f['entry/instrument/detector/beam_center_x'][0]
        beam_center_y = f['entry/instrument/detector/beam_center_y'][0]
        beamstop_diameter = f['/entry/DAS_logs/beamStop/size'][0]/10.0
        lateral_offset = f['entry/DAS_logs/areaDetector/offset'][0] #in cm?
        SampleToSourceAp = f['/entry/DAS_logs/geometry/sourceApertureToSample'][0] #"Calculated distance between sample and source aperture" in cm
        SampleToDetector = f['/entry/DAS_logs/geometry/sampleToAreaDetector'][0]
        SourceAp_Descrip = str(f['/entry/DAS_logs/geometry/sourceAperture'][0]) #source aperture in mm - > cm; convert to RADIUS?
        SourceAp_Descrip = SourceAp_Descrip[2:]
        SourceAp_Descrip = SourceAp_Descrip[:-3]
        SourceAp =  float(SourceAp_Descrip)
        SourceAp = SourceAp/20.0 #external sample aperture in mm -> cm (radius)
        SampleApExternal = f['/entry/DAS_logs/geometry/externalSampleAperture'][0]/20.0 #external sample aperture in mm -> cm (radius)
        Wavelength = f['entry/instrument/monochromator/wavelength'][0]
        Wavelength_spread = f['entry/instrument/monochromator/wavelength_error'][0]

        realDistX = x_pixel_size*(1.0) + lateral_offset
        realDistY = y_pixel_size*(1.0)
        realDistZ = detector_distance
        X, Y = np.indices(data.shape)
        x0_pos =  realDistX - beam_center_x*x_pixel_size + (X)*x_pixel_size 
        y0_pos =  realDistY - beam_center_y*y_pixel_size + (Y)*y_pixel_size

        InPlane0_pos = np.sqrt(x0_pos**2 + y0_pos**2)
        twotheta = np.arctan2(InPlane0_pos,realDistZ)
        phi = np.arctan2(y0_pos,x0_pos)
        twotheta_x = np.arctan2(x0_pos,realDistZ)
        twotheta_y = np.arctan2(y0_pos,realDistZ)

        L1 = SampleToSourceAp
        L2 = SampleToDetector
        R1 = SourceAp #source aperture radius in cm
        R2 = SampleApExternal #sample aperture radius in cm
        Inv_LPrime = 1.0/L1 + 1.0/L2
        k = 2*np.pi/Wavelength
        Sigma_D_Perp = np.sin(phi)*x_pixel_size + np.cos(phi)*y_pixel_size
        Sigma_D_Parl = np.cos(phi)*x_pixel_size + np.sin(phi)*y_pixel_size
        SigmaQPerpSqr = (k*k/12.0)*(3*np.power(R1/L1,2) + 3.0*np.power(R2*Inv_LPrime,2)+ np.power(Sigma_D_Perp/L2,2))
        SigmaQParlSqr = (k*k/12.0)*(3*np.power(R1/L1,2) + 3.0*np.power(R2*Inv_LPrime,2)+ np.power(Sigma_D_Parl/L2,2))
        R = np.sqrt(np.power(x0_pos,2)+np.power(y0_pos,2))
        Q0 = k*R/L2

        g = 981 #in cm/s^2
        m_div_h = 252.77 #in s cm^-2
        A = -0.5*981*L2*(L1+L2)*np.power(m_div_h , 2)
        WL = Wavelength*1E-8
        SigmaQParlSqr = SigmaQParlSqr + np.power(Wavelength_spread*k/(L2),2)*(R*R -4*A*np.sin(phi)*WL*WL + 4*A*A*np.power(WL,4))/6.0 #gravity correction makes vary little difference for wavelength spread < 20%
            
        Q_total = (4.0*np.pi/Wavelength)*np.sin(twotheta/2.0)
        Qx = Q_total*np.cos(twotheta/2.0)*np.cos(phi)
        Qy = Q_total*np.cos(twotheta/2.0)*np.sin(phi)
        Qz = Q_total*np.sin(twotheta/2.0)     
        Q_perp_unc = np.ones_like(Q_total)*np.sqrt(SigmaQPerpSqr)
        Q_parl_unc = np.sqrt(SigmaQParlSqr)
        InPlaneAngleMap = phi*180.0/np.pi
        #TwoTheta_deg = twotheta*180.0/np.pi

    return Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimX, dimY

def NG7SANS_QxQyASCII(Data, Data_Unc, QXMap, QYMap, QZMap, QPerpUncMap, QParlUncMap):

    QQX = QXMap.T
    QXData = QQX.flatten()
    
    QQY = QYMap.T
    QYData = QQY.flatten()
    
    QQZ = QZMap.T
    QZData = QQZ.flatten()

    QPP = QPerpUncMap.T
    QPerpUnc = QPP.flatten()

    QPR = QParlUncMap.T
    QParlUnc = QPR.flatten()

    Shadow = np.ones_like(QXMap)
    Shadow = Shadow.T
    ShadowHolder = Shadow.flatten()

    Int = Data.T
    Intensity = Int.flatten()

    Unc = Data_Unc.T
    IntensityUnc = Unc.flatten()
    
    print('Outputting 2D data')        
    ASCII = np.array([QXData, QYData, Intensity, IntensityUnc, QZData, QParlUnc, QPerpUnc, ShadowHolder])
    ASCII = ASCII.T
    np.savetxt('Test_2D_NG7SANS.DAT', ASCII, delimiter = ' ', comments = ' ', header = 'ASCII data created Mon, Jan 13, 2020 2:39:54 PM')

    return

def NG7SANS_SectorMask(InPlaneAngleMap, PrimaryAngle, AngleWidth, BothSides):

    Angles = InPlaneAngleMap
    SM = np.zeros_like(InPlaneAngleMap)
    SM[np.absolute(Angles - PrimaryAngle) <= AngleWidth] = 1.0
    SM[np.absolute(Angles + 360 - PrimaryAngle) <= AngleWidth] = 1.0
    SM[np.absolute(Angles - 360 - PrimaryAngle) <= AngleWidth] = 1.0

    if BothSides > 0:
        SecondaryAngle = PrimaryAngle + 180
    else:
        SecondaryAngle = PrimaryAngle
        if SecondaryAngle > 360:
            SecondaryAngle = SecondaryAngle - 360
    SM[np.absolute(Angles - SecondaryAngle) <= AngleWidth] = 1.0
    SM[np.absolute(Angles + 360 - SecondaryAngle) <= AngleWidth] = 1.0
    SM[np.absolute(Angles - 360 - SecondaryAngle) <= AngleWidth] = 1.0

    return SM

def NG7SANS_TwoDimToOneDim(Q_min, Q_max, Q_bins, Q_Total, Q_Unc, Mask, Data, Data_Unc):

    Q_Values = np.linspace(Q_min, Q_max, Q_bins, endpoint=True)
    Q_step = (Q_max - Q_min) / Q_bins

    Exp_bins = np.linspace(Q_min, Q_max + Q_step, Q_bins + 1, endpoint=True)
    Counts, _ = np.histogram(Q_Total[Mask > 0], bins=Exp_bins, weights=Data[Mask > 0])
    UncCounts, _ = np.histogram(Q_Total[Mask > 0], bins=Exp_bins, weights=np.power(Data_Unc[Mask > 0],2))    
    MeanQSum, _ = np.histogram(Q_Total[Mask > 0], bins=Exp_bins, weights=Q_Total[Mask > 0])
    MeanQUnc, _ = np.histogram(Q_Total[Mask > 0], bins=Exp_bins, weights=np.power(Q_Unc[Mask > 0],2)) 
    Pixels, _ = np.histogram(Q_Total[Mask > 0], bins=Exp_bins, weights=np.ones_like(Data)[Mask > 0])

    nonzero_mask = (Pixels > 0) #True False map

    Intensity = Counts[nonzero_mask] / Pixels[nonzero_mask]
    Sigma_Intensity = np.sqrt(UncCounts[nonzero_mask]) / Pixels[nonzero_mask]
    Q = Q_Values[nonzero_mask]
    MeanQ =  MeanQSum[nonzero_mask] / Pixels[nonzero_mask]
    Sigma_MeanQ =  np.sqrt(MeanQUnc[nonzero_mask]) / Pixels[nonzero_mask]

    Shadow = np.ones_like(Q)

    ErrorBarsYesNo = 0
    PlotYesNo = 1
    if PlotYesNo == 1:
        fig = plt.figure()
        if ErrorBarsYesNo == 1:
            ax = plt.axes()
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.errorbar(Q, Intensity, yerr=Sigma_Intensity, fmt = 'b*', label='Counts')
        else:
            plt.loglog(Q, Intensity, 'b*', label='Counts')
                
        plt.xlabel('Q')
        plt.ylabel('Intensity')
        plt.title('Plot')
        plt.legend()
        fig.savefig('Test.png')
        plt.pause(2)
        plt.close()

        Output = {}
        Output['Q'] = Q
        Output['Q_Mean'] = MeanQ
        Output['I'] = Intensity
        Output['I_Unc'] = Sigma_Intensity
        Output['Q_Uncertainty'] = Sigma_MeanQ
        Output['Shadow'] = Shadow

    return Output

def SaveTextData(SliceType, Sample, Config, DataMatrix):

    Q = DataMatrix['Q']
    Int = DataMatrix['I']
    IntUnc = DataMatrix['I_Unc']
    Q_mean = DataMatrix['Q_Mean']
    Q_Unc = DataMatrix['Q_Uncertainty']
    Shadow = np.ones_like(Q)
    text_output = np.array([Q, Int, IntUnc, Q_mean, Q_Unc, Shadow])
    text_output = text_output.T
    np.savetxt('SixCol_{samp},{cf}_{cut}.txt'.format(samp=Sample, cf = Config, cut = SliceType), text_output, delimiter = ' ', comments = ' ', header= 'Q, I, DelI, Q_mean, Q_Unc, Shadow', fmt='%1.4e')
  
    return

#*************************************************
#***        Start of 'The Program'             ***
#*************************************************
Sample_Names, Configs, BlockBeam, Scatt, Trans, Pol_Trans, HE3_Trans, start_number, filenumberlisting = NG7SANS_SortData(YesNoManualHe3Entry, New_HE3_Files, MuValues, TeValues)
print(Scatt)
print('  ')
print(Trans)

Config = NG7SANS_Config_ID(Scatt_filenumber)

Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimX, dimY  = NG7SANS_QCalculation(Scatt_filenumber)

Abs_Trans = NG7SANS_TransCountsPer1E8MonCounts(Trans_filenumber)

Sample_Trans = 1.0
Data, DataUnc = NG7SANS_AbsScaleScattData(Scatt_filenumber, Abs_Trans, Sample_Trans)

PrimaryAngle = 42
AngleWidth = 45
BothSides = 1
SectorMask = NG7SANS_SectorMask(InPlaneAngleMap, PrimaryAngle, AngleWidth, BothSides)

Q_min = 0.001
Q_max = 0.03
Q_bins = 100
TwoDimData = NG7SANS_TwoDimToOneDim(Q_min, Q_max, Q_bins, Q_total, Q_total, SectorMask, Data, DataUnc)

Sample = 'Samp' + str(Scatt_filenumber)
SliceType = 'Sec42,45deg'
SaveTextData(SliceType, Sample, Config, TwoDimData)

#NG7SANS_QxQyASCII(Data, DataUnc, Qx, Qy, Qz, Q_perp_unc, Q_parl_unc)
