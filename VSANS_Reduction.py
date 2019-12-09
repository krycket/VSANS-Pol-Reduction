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
#from ConfinedSkyrmions_6June2019 import *
#from MnFe2O4NPs_March2019 import *
#from Fe3O4Cal_Nov2019 import *
#The following is from Fe3O4Cal_Nov2019.py:

UncertaintyMode = 1 #1 = Sigma of summed counts; 0 = Varience of pixels (to match IGOR)
UsePolCorr = 1 #1 = usual; 0 means to turn the supermirror and flipper corrections off, but retain the 3He absorption correction
#path = 'D:\Projects_Current\MnFe2O4PureNPs\Ijiri_March2019_VSANSdata/'
path = ''
TransMask = 'PolTrans_MASK.h5'
start_number = 51277 #51276
end_number = 51360
Excluded_Files = [51279, 51289]
Select_GroupIDs = 0 #0 = No (use all IDs, except those listed as undesired), 1 = Yes (process selected IDs only)
Desired_GroupIDs = []
Undesired_GroupIDs = ['29.0']
BlockedBeamFiles = [51279] #can be empty (no correction) or can list in multiple configurations if desired; automaticaly matched to files of interest based on detector distance
Mask_Files = [51282]
Mask_Thresholds_Front = [4]
Mask_Thresholds_Middle = [1]
Q_min = 0.001
Q_max = 0.13
Q_bins = 120
TransPanel = 'MR' #Default is 'MR'
SectorCutAngles = 10.0
Use_Adam4021 = 1 #0 = no; 1 = yes
Use_Lakeshore340 = 1 #0 = no; 1 = yes
Unpol_Key_list = ['Vert']#['Horz','Vert','Diag','Circ']
FullPol_Key_list = ['Horz','Vert']#['Horz','Vert','Diag','Circ']
FullPolYeseNo = 1
UnpolYesNo = 0
LowQ_Offset_Unpol =  0.0 #will subtract this amount; might be required if forgot to get a good blocked beam
LowQ_Offset_SF = 0.0 #2.3
LowQ_Offset_NSF = 0.0 #3
Print_ASCII = 0 #1 = yes; otherwise no
YesNoManualHe3Entry = 0 #0 for no (default), 1 for yes 
New_HE3_Files = [28422, 28498, 28577, 28673, 28755, 28869] #For manual declaration of 3He cell and parameters (should not be needed from July 2019 onward)
MuValues = [3.374, 3.105, 3.374, 3.105, 3.374, 3.105] #[3.374, 3.105]=[Fras, Bur]; For manual declaration of 3He cell and parameters (should not be needed from July 2019 onward)
TeValues = [0.86, 0.86, 0.86, 0.86, 0.86, 0.86] #[0.86, 0.86]=[Fras, Bur]; For manual declaration of 3He cell and parameters (should not be needed from July 2019 onward)

#This program is set to reduce VSANS data using middle and front detectors - umnpol, fullpol available.
#To do: BS shadow, deadtime corr., check abs. scaling, uncertainty propagation through , choice of 2 cross-section empty files, half-pol

short_detectors = ["MT", "MB", "ML", "MR", "FT", "FB", "FL", "FR"]

def Unique_Config_ID(filenumber):
    
    filename = path + "sans" + str(filenumber) + ".nxs.ngv"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        Desired_FrontCarriage_Distance = int(f['entry/DAS_logs/carriage1Trans/desiredSoftPosition'][0]) #in cm
        Desired_MiddleCarriage_Distance = int(f['entry/DAS_logs/carriage2Trans/desiredSoftPosition'][0]) #in cm
        Wavelength = f['entry/DAS_logs/wavelength/wavelength'][0]
        Guides = int(f['entry/DAS_logs/guide/guide'][0])
        Configuration_ID = str(Guides) + "Gd" + str(Desired_FrontCarriage_Distance) + "cmF" + str(Desired_MiddleCarriage_Distance) + "cmM" + str(Wavelength) + "Ang"
        
    return Configuration_ID

def File_Type(filenumber):

    Type = 'UNKNOWN'
    SolenoidPosition = 'UNKNOWN'
    
    filename = path + "sans" + str(filenumber) + ".nxs.ngv"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        Purpose = f['entry/reduction/file_purpose'][()] #SCATT, TRANS, HE3
        if str(Purpose).find("SCATT") != -1:
            Type = 'SCATT'
        else:
            Type = 'TRANS'
        if "backPolarization" in f['entry/DAS_logs/']:
            BackPolDirection = f['entry/DAS_logs/backPolarization/direction'][()]
        else:
            BackPolDirection = [b'UNPOLARIZED']
        if str(BackPolDirection).find("UP") != -1 or str(BackPolDirection).find("DOWN") != -1:
            SolenoidPosition = 'IN'
        else:
            SolenoidPosition = 'OUT'

    return Type, SolenoidPosition

def SortDataAutomatic(YesNoManualHe3Entry, New_HE3_Files, MuValues, TeValues):

    BlockBeam = {}
    Configs = {}
    Sample_Names = {}
    Scatt = {}
    Trans = {}
    Pol_Trans = {}
    HE3_Trans = {}

    UU_filenumber = -10
    DU_filenumber = -10
    DD_filenumber = -10
    UD_filenumber = -10
    filenames = '0'
    record_adam4021 = 0
    record_temp = 0
    CellIdentifier = 0
    HE3OUT_filenumber = -10
    
    filelist = [fn for fn in os.listdir("./") if fn.endswith(".nxs.ngv")] #or filenames = [fn for fn in os.listdir("./") if os.path.isfile(fn)]
    if len(filelist) >= 1:
        for name in filelist:
            filename = str(name)
            filenumber = int(filename[4:9])
            config = Path(filename)
            if config.is_file():
                f = h5py.File(filename)
                Count_time = f['entry/collection_time'][0]
                Descrip = str(f['entry/sample/description'][0])
                Descrip = Descrip[2:]
                Descrip = Descrip[:-1]
                #print('Reading:', filenumber, ' ', Descrip)
                if Count_time > 59 and str(Descrip).find("Align") == -1:

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

                    Purpose = f['entry/reduction/file_purpose'][()] #SCATT, TRANS, HE3
                    Intent = f['entry/reduction/intent'][()] #Sample, Empty, Blocked Beam, Open Beam
                    #ID = str(f['entry/sample/group_id'][0])
                    End_time = dateutil.parser.parse(f['entry/end_time'][0])
                    Trans_Counts = f['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                    #trans_mask = Trans_masks['MR']
                    #trans_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                    #trans_data = trans_data*trans_mask
                    #Trans_Counts = trans_data.sum()
                    MonCounts = f['entry/control/monitor_counts'][0]
                    Trans_Distance = f['entry/instrument/detector_{ds}/distance'.format(ds=TransPanel)][0]
                    Attenuation = f['entry/DAS_logs/attenuator/attenuator'][0]
                    Wavelength = f['entry/DAS_logs/wavelength/wavelength'][0]
                    Config = Unique_Config_ID(filenumber)
                    FrontPolDirection = f['entry/DAS_logs/frontPolarization/direction'][()]
                    if "backPolarization" in f['entry/DAS_logs/']:
                        BackPolDirection = f['entry/DAS_logs/backPolarization/direction'][()]
                    else:
                        BackPolDirection = [b'UNPOLARIZED']

                    if len(Configs) < 1:
                        Configs = [Config]
                    else:
                        if Config not in Configs:
                            Configs.append(Config)

                    if str(Intent).find("Blocked") != -1: #i.e. a blocked beam
                        if str(Purpose).find("TRANS") != -1 or str(Purpose).find("HE3") != -1:
                            BlockBeam[Config] = {'Trans':{'File' : filenumber, 'CountsPerSecond' : Trans_Counts/Count_time}}
                        elif str(Purpose).find("SCATT") != -1:
                            BlockBeam[Config] = {'Scatt':{'File' : filenumber}}
                            
                    elif str(Intent).find("Sample") != -1 or str(Intent).find("Empty") != -1 or str(Intent).find("Open") != -1:
                        if len(Sample_Names) < 1:
                            Sample_Names = [Sample_Name]
                        else:
                            if Sample_Name not in Sample_Names:
                                Sample_Names.append(Sample_Name)
                                
                        Intent_short = str(Intent)
                        Intent_short = Intent_short[3:-2]
                        Intent_short = Intent_short.replace(' Cell', '')
                        Intent_short = Intent_short.replace(' Beam', '')
                            
                        if str(Purpose).find("SCATT") != -1:
                            if Sample_Name not in Scatt:
                                Scatt[Sample_Name] = {'Intent': Intent_short, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'Unpol': 'NA', 'U' : 'NA', 'D' : 'NA','UU' : 'NA', 'DU' : 'NA', 'DD' : 'NA', 'UD' : 'NA'}}}
                            if Config not in Scatt[Sample_Name]['Config(s)']:
                                #Check that multipe configurations per sample are processed properly; Scatt[Sample_Name]['Config(s)'].append(Config)
                                Scatt[Sample_Name]['Config(s)'][Config] = {'Unpol': 'NA', 'U' : 'NA', 'D' : 'NA','UU' : 'NA', 'DU' : 'NA', 'DD' : 'NA', 'UD' : 'NA'}
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
                            if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UP") != -1:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['UU']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['UU'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['UU'].append(filenumber)
                            if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UP") != -1:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['UD']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['UD'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['UD'].append(filenumber)

                            if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['DD']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['DD'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['DD'].append(filenumber)
                            if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("DOWN") != -1:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['DU']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['DU'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['DU'].append(filenumber)  
                                
                            
                        if str(Purpose).find("TRANS") != -1:
                            if Sample_Name not in Trans:
                                Trans[Sample_Name] = {'Intent': Intent_short, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'Unpol': 'NA', 'U' : 'NA', 'D' : 'NA'}}}
                            if Config not in Trans[Sample_Name]['Config(s)']:
                                Trans[Sample_Name]['Config(s)'][Config] = {'Unpol': 'NA', 'U' : 'NA', 'D': 'NA'}
                            if Sample_Name not in Pol_Trans:
                                Pol_Trans[Sample_Name] = {'T_UU' : {'File' : 'NA'},
                                                          'T_DU' : {'File' : 'NA'},
                                                          'T_DD' : {'File' : 'NA'},
                                                          'T_UD' : {'File' : 'NA'},
                                                          'T_SM' : {'File' : 'NA'},
                                                          'Config' : 'NA'
                                                          }
                            if str(FrontPolDirection).find("UNPOLARIZED") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['Unpol']:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol'].append(filenumber)
                            if str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['U']:
                                    Trans[Sample_Name]['Config(s)'][Config]['U'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['U'].append(filenumber)
                            if str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['D']:
                                    Trans[Sample_Name]['Config(s)'][Config]['D'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['D'].append(filenumber)
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

                            
                        if str(Purpose).find("HE3") != -1:
                            
                            HE3Type = str(f['entry/sample/description'][()])
                            if HE3Type[-7:-2] == 'HeOUT':
                                if Sample_Name not in Trans:
                                    Trans[Sample_Name] = {'Intent': Intent_short, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'Unpol': 'NA', 'U' : 'NA', 'D' : 'NA'}}}
                                if Config not in Trans[Sample_Name]['Config(s)']:
                                    Trans[Sample_Name]['Config(s)'][Config] = {'Unpol': 'NA', 'U' : 'NA', 'D': 'NA'}
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['Unpol']:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol'].append(filenumber)
                            
                            if YesNoManualHe3Entry == 1:
                                if filenumber in New_HE3_Files:
                                    ScaledOpacity = MuValues[CellIdentifier]
                                    TE = TeValues[CellIdentifier]
                                    #HE3Insert_Time = (End_time.timestamp() - Count_time)/3600.0
                                    CellTimeIdentifier = (End_time.timestamp() - Count_time)/3600.0
                                    #f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                                    HE3Insert_Time = (End_time.timestamp() - Count_time)/3600.0
                                    CellIdentifier += 1    
                            else: #i.e. automatic entry
                                CellTimeIdentifier = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                                if CellTimeIdentifier not in HE3_Trans:
                                    HE3Insert_Time = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                                    Opacity = f['/entry/DAS_logs/backPolarization/opacityAt1Ang'][0]
                                    Wavelength = f['/entry/DAS_logs/wavelength/wavelength'][0]
                                    ScaledOpacity = Opacity*Wavelength
                                    TE = f['/entry/DAS_logs/backPolarization/glassTransmission'][0]
                            if HE3Type[-7:-2] == 'HeOUT':
                            #He3Type = Descrip[-5:]
                            #if str(He3Type).find("HEOUT") != -1:
                                HE3OUT_filenumber = filenumber
                                HE3OUT_config = Config
                                HE3OUT_sample = Sample_Name
                                HE3OUT_attenuators = int(f['entry/instrument/attenuator/num_atten_dropped'][0])
                            #elif str(He3Type).find("HEIN") != -1:
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
                                        else:
                                            HE3_Trans[CellTimeIdentifier]['Config'].append(HE3IN_config)
                                            HE3_Trans[CellTimeIdentifier]['HE3_OUT_file'].append(HE3OUT_filenumber)
                                            HE3_Trans[CellTimeIdentifier]['HE3_IN_file'].append(HE3IN_filenumber)
                                            HE3_Trans[CellTimeIdentifier]['Elasped_time'].append(Elasped_time)
                                            



    return Sample_Names, Configs, BlockBeam, Scatt, Trans, Pol_Trans, HE3_Trans

def ReadIn_Masks():

    Masks = {}
    single_mask = {}

    filename = '0'
    Mask_files = [fn for fn in os.listdir("./") if fn.endswith("MASK.h5")]
    if len(Mask_files) >= 1:
        for name in Mask_files:
            filename = str(name)
            associated_filenumber = filename[:5]
            if associated_filenumber.isdigit() == True:
                ConfigID = Unique_Config_ID(associated_filenumber)
                if ConfigID not in Masks:
                    Masks[ConfigID] = {'Trans' : 'NA', 'Scatt_Standard' : 'NA', 'Scatt_WithSolenoid' : 'NA'}
                Type, SolenoidPosition = File_Type(associated_filenumber)
                config = Path(filename)
                if config.is_file():
                    f = h5py.File(filename)
                    for dshort in short_detectors:
                        mask_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                        #need to reverse zeros and ones (assuming IGOR-made masks)
                        single_mask[dshort] = np.zeros_like(mask_data)
                        single_mask[dshort][mask_data == 0] = 1.0
                        
                    if str(Type).find("TRANS") != -1:
                        Masks[ConfigID]['Trans'] = single_mask.copy()
                        print('Saved', filename, ' as Trans Mask for', ConfigID)
                        
                    if str(Type).find("SCATT") != -1 and str(SolenoidPosition).find("OUT") != -1:
                        Masks[ConfigID]['Scatt_Standard'] = single_mask.copy()
                        print('Saved', filename, ' as Standard Scatt Mask for', ConfigID)
                        
                    if str(Type).find("SCATT") != -1 and str(SolenoidPosition).find("IN") != -1:
                        Masks[ConfigID]['Scatt_WithSolenoid'] = single_mask.copy()
                        print('Saved', filename, ' as Scatt Mask With Solenoid for', ConfigID)
                        
                
    return Masks

def Process_Transmissions(BlockBeam, Masks, HE3_Trans, Pol_Trans, Trans):

    #HE3_Trans[CellTimeIdentifier]['Te', 'Mu', 'Insert_time', 'Config', 'HE3_OUT_file', 'HE3_IN_file', 'Elasped_time']
    #BlockBeam[Config]['Trans':{'File','CountsPerSecond'} or 'Scatt':{'File'}]
    #Masks[Config]['Trans' or 'Scatt_Standard' or 'Scatt_WithSolenoid' -> 'NA' or dshorts]

    for Cell in HE3_Trans:
        counter = 0
        for InFile in HE3_Trans[Cell]['HE3_IN_file']:
            OutFile = HE3_Trans[Cell]['HE3_OUT_file'][counter]
            Config = HE3_Trans[Cell]['Config'][counter]
            if Config in BlockBeam:
                if 'Trans' in BlockBeam[Config]:
                    BBFile = BlockBeam[Config]['Trans']['File']
                elif 'Scatt' in BlockBeam[Config]:
                    BBFile = BlockBeam[Config]['Scatt']['File']
                else:
                    BBFile = 0
            if Config in Masks and 'NA' not in Masks[Config]['Trans']:
                mask_it = np.array(Masks[Config]['Trans'][TransPanel])
                IN = path + "sans" + str(InFile) + ".nxs.ngv"
                OUT = path + "sans" + str(OutFile) + ".nxs.ngv"
                f = h5py.File(IN)
                INMon = f['entry/control/monitor_counts'][0]
                IN_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                INCount_time = f['entry/collection_time'][0]
                g = h5py.File(OUT)
                OUTMon = g['entry/control/monitor_counts'][0]
                OUT_data = np.array(g['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                OUTCount_time = g['entry/collection_time'][0]
                if BBFile == 0:
                    BB_Data = np.zeros_like(IN_data)
                else:
                    BB = path + "sans" + str(BBFile) + ".nxs.ngv"
                    h = h5py.File(BB)
                    BBMon = h['entry/control/monitor_counts'][0]
                    BBCount_time = h['entry/collection_time'][0]
                    BB_data = np.array(h['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                trans_num = (IN_data - BB_data*(INCount_time/BBCount_time))*mask_it
                trans_denom = (OUT_data - BB_data*(OUTCount_time/BBCount_time))*mask_it
                trans = (np.sum(trans_num)/np.sum(trans_denom))*(OUTMon / INMon)
                if 'Transmission' not in HE3_Trans[Cell]:
                    HE3_Trans[Cell]['Transmission'] = [trans]
                else:
                    HE3_Trans[Cell]['Transmission'].append(trans)
            else:
                IN = path + "sans" + str(InFile) + ".nxs.ngv"
                OUT = path + "sans" + str(OutFile) + ".nxs.ngv"
                f = h5py.File(IN)
                INMon = f['entry/control/monitor_counts'][0]
                IN_counts = f['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                INCount_time = f['entry/collection_time'][0]
                g = h5py.File(OUT)
                OUTMon = g['entry/control/monitor_counts'][0]
                OUT_counts = g['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                OUTCount_time = g['entry/collection_time'][0]
                if BBFile == 0:
                    BB_counts = 0.0
                else:
                    BB = path + "sans" + str(BBFile) + ".nxs.ngv"
                    h = h5py.File(BB)
                    BBMon = h['entry/control/monitor_counts'][0]
                    BBCount_time = h['entry/collection_time'][0]
                    BB_counts = h['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                trans_num = (IN_counts - BB_counts*(INCount_time/BBCount_time))
                trans_denom = (OUT_counts - BB_counts*(OUTCount_time/BBCount_time))
                trans = (trans_num/trans_denom)*(OUTMon / INMon)
                if 'Transmission' not in HE3_Trans[Cell]:
                    HE3_Trans[Cell]['Transmission'] = [trans]
                else:
                    HE3_Trans[Cell]['Transmission'].append(trans)                
            counter += 1

    for Samp in Pol_Trans:
        counter = 0
        for UUFile in Pol_Trans[Samp]['T_UU']['File']:
            DUFile = Pol_Trans[Samp]['T_DU']['File'][counter]
            DDFile = Pol_Trans[Samp]['T_DD']['File'][counter]
            UDFile = Pol_Trans[Samp]['T_UD']['File'][counter]
            SMFile = Pol_Trans[Samp]['T_SM']['File'][counter]
            Config = Pol_Trans[Samp]['Config'][counter]
            if Config in BlockBeam:
                if 'Trans' in BlockBeam[Config]:
                    BBFile = BlockBeam[Config]['Trans']['File']
                elif 'Scatt' in BlockBeam[Config]:
                    BBFile = BlockBeam[Config]['Scatt']['File']
                else:
                    BBFile = 0
            if Config in Masks and 'NA' not in Masks[Config]['Trans']:
                mask_it = np.array(Masks[Config]['Trans'][TransPanel])
                UU = path + "sans" + str(UUFile) + ".nxs.ngv"
                DU = path + "sans" + str(DUFile) + ".nxs.ngv"
                DD = path + "sans" + str(DDFile) + ".nxs.ngv"
                UD = path + "sans" + str(UDFile) + ".nxs.ngv"
                SM = path + "sans" + str(SMFile) + ".nxs.ngv"
                f = h5py.File(UU)
                UUMon = f['entry/control/monitor_counts'][0]
                UU_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                UUCount_time = f['entry/collection_time'][0]

                g = h5py.File(DU)
                DUMon = g['entry/control/monitor_counts'][0]
                DU_data = np.array(g['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                DUCount_time = g['entry/collection_time'][0]

                h = h5py.File(DD)
                DDMon = h['entry/control/monitor_counts'][0]
                DD_data = np.array(h['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                DDCount_time = h['entry/collection_time'][0]

                j = h5py.File(UD)
                UDMon = j['entry/control/monitor_counts'][0]
                UD_data = np.array(j['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                UDCount_time = j['entry/collection_time'][0]

                k = h5py.File(SM)
                SMMon = k['entry/control/monitor_counts'][0]
                SM_data = np.array(k['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                SMCount_time = k['entry/collection_time'][0]

                if BBFile == 0:
                    BB_Data = np.zeros_like(UU_data)
                else:
                    BB = path + "sans" + str(BBFile) + ".nxs.ngv"
                    l = h5py.File(BB)
                    BBMon = l['entry/control/monitor_counts'][0]
                    BBCount_time = l['entry/collection_time'][0]
                    BB_data = np.array(l['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                    
                trans_UU = (UU_data - BB_data*(UUCount_time/BBCount_time))*mask_it
                trans_DU = (DU_data - BB_data*(DUCount_time/BBCount_time))*mask_it
                trans_DD = (DD_data - BB_data*(DDCount_time/BBCount_time))*mask_it
                trans_UD = (UD_data - BB_data*(UDCount_time/BBCount_time))*mask_it
                trans_SM = (SM_data - BB_data*(SMCount_time/SMCount_time))*mask_it
                
                UU_value = (np.sum(trans_UU)/np.sum(trans_SM))*(SMMon / UUMon)
                UD_value = (np.sum(trans_UD)/np.sum(trans_SM))*(SMMon / UDMon)
                DD_value = (np.sum(trans_DD)/np.sum(trans_SM))*(SMMon / DDMon)
                DU_value = (np.sum(trans_DU)/np.sum(trans_SM))*(SMMon / DUMon)
                SM_value = np.sum(trans_SM)
                
                if 'Trans' not in Pol_Trans[Samp]['T_UU']:
                    Pol_Trans[Samp]['T_UU']['Trans'] = [UU_value]
                    Pol_Trans[Samp]['T_DU']['Trans'] = [DU_value]
                    Pol_Trans[Samp]['T_DD']['Trans'] = [DD_value]
                    Pol_Trans[Samp]['T_UD']['Trans'] = [UD_value]
                    Pol_Trans[Samp]['T_SM']['Abs_Trans'] = [SM_value]
                else:
                    Pol_Trans[Samp]['T_UU']['Trans'].append(UU_value)
                    Pol_Trans[Samp]['T_DU']['Trans'].append(DU_value)
                    Pol_Trans[Samp]['T_DD']['Trans'].append(DD_value)
                    Pol_Trans[Samp]['T_UD']['Trans'].append(UD_value)
                    Pol_Trans[Samp]['T_SM']['Abs_Trans'].append(SM_value)
            else:
                UU = path + "sans" + str(UUFile) + ".nxs.ngv"
                DU = path + "sans" + str(DUFile) + ".nxs.ngv"
                DD = path + "sans" + str(DDFile) + ".nxs.ngv"
                UD = path + "sans" + str(UDFile) + ".nxs.ngv"
                SM = path + "sans" + str(SMFile) + ".nxs.ngv"
                f = h5py.File(UU)
                UUMon = f['entry/control/monitor_counts'][0]
                UU_counts = f['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                UUCount_time = f['entry/collection_time'][0]

                g = h5py.File(DU)
                DUMon = g['entry/control/monitor_counts'][0]
                DU_counts = g['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                DUCount_time = g['entry/collection_time'][0]

                h = h5py.File(DD)
                DDMon = h['entry/control/monitor_counts'][0]
                DD_counts = h['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                DDCount_time = h['entry/collection_time'][0]

                j = h5py.File(UD)
                UDMon = j['entry/control/monitor_counts'][0]
                UD_counts = j['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                UDCount_time = j['entry/collection_time'][0]

                k = h5py.File(SM)
                SMMon = k['entry/control/monitor_counts'][0]
                SM_counts = k['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                SMCount_time = k['entry/collection_time'][0]
                
                if BBFile == 0:
                    BB_counts = 0.0
                else:
                    BB = path + "sans" + str(BBFile) + ".nxs.ngv"
                    l = h5py.File(BB)
                    BBMon = l['entry/control/monitor_counts'][0]
                    BBCount_time = l['entry/collection_time'][0]
                    BB_counts = l['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]

                trans_UU = (UU_counts - BB_counts*(UUCount_time/BBCount_time))
                trans_DU = (DU_counts - BB_counts*(DUCount_time/BBCount_time))
                trans_DD = (DD_counts - BB_counts*(DDCount_time/BBCount_time))
                trans_UD = (UD_counts - BB_counts*(UDCount_time/BBCount_time))
                trans_SM = (SM_counts - BB_counts*(SMCount_time/SMCount_time))
                
                UU_value = (trans_UU/trans_SM)*(SMMon / UUMon)
                UD_value = (trans_DU/trans_SM)*(SMMon / UDMon)
                DD_value = (trans_DD/trans_SM)*(SMMon / DDMon)
                DU_value = (trans_UD/trans_SM)*(SMMon / DUMon)
                SM_value = trans_SM

                if 'Trans' not in Pol_Trans[Samp]['T_UU']:
                    Pol_Trans[Samp]['T_UU']['Trans'] = [UU_value]
                    Pol_Trans[Samp]['T_DU']['Trans'] = [DU_value]
                    Pol_Trans[Samp]['T_DD']['Trans'] = [DD_value]
                    Pol_Trans[Samp]['T_UD']['Trans'] = [UD_value]
                    Pol_Trans[Samp]['T_SM']['Abs_Trans'] = [SM_value]
                else:
                    Pol_Trans[Samp]['T_UU']['Trans'].append(UU_value)
                    Pol_Trans[Samp]['T_DU']['Trans'].append(DU_value)
                    Pol_Trans[Samp]['T_DD']['Trans'].append(DD_value)
                    Pol_Trans[Samp]['T_UD']['Trans'].append(UD_value)
                    Pol_Trans[Samp]['T_SM']['Abs_Trans'].append(SM_value)
                    

            counter += 1
            
    return


def Plex_File():

    PlexData = {}

    filename = '0'
    Plex_file = [fn for fn in os.listdir("./") if fn.startswith("PLEX")]
    if len(Plex_file) >= 1:
        filename = str(Plex_file[0])
    #filename = path + "PLEX_" + str(filenumber) + "_VSANS_DIV.h5"
    config = Path(filename)
    if config.is_file():
        print('Reading in ', filename)
        f = h5py.File(filename)
        for dshort in short_detectors:
            data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
            PlexData[dshort] = data.flatten()
    else:
        filenumber = start_number
        while filenumber < end_number + 1:
            filename = path + "sans" + str(filenumber) + ".nxs.ngv"
            config = Path(filename)
            if config.is_file():
                f = h5py.File(filename)
                for dshort in short_detectors:
                    data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                    PlexData[dshort] = np.ones_like(data).flatten()
                filenumber = end_number + 1
            filenumber += 1
        print('Plex file not found; populated with ones instead')   
            
    return PlexData

def Make_Mask_From_File(mask_number, Front_threshold, Middle_threshold):
    #Returns measured_masks[dshort], usually based on a glassy carbon file
    measured_masks = {}

    
    long_name = path + "sans" + str(filenumber) + ".nxs.ngv"
    config = Path(long_name)
    if config.is_file():
        print('Reading in mask file number:', filenumber)
        f = h5py.File(long_name)
        for dshort in short_detectors:
            measured_mask_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
            measured_masks[dshort] = np.zeros_like(measured_mask_data)
            position_key = dshort[0]
            if position_key == 'F':
                measured_masks[dshort][measured_mask_data >= Front_threshold] = 1.0
            if position_key == 'M':
                measured_masks[dshort][measured_mask_data >= Middle_threshold] = 1.0
    
    '''
    filename = path + "VSANS_SOLENOID_MASK.h5"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        for dshort in short_detectors:
            mask_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
            #need to reverse zeros and ones
            measured_masks[dshort] = np.zeros_like(mask_data)
            measured_masks[dshort][mask_data == 0] = 1.0
    '''
                
    return measured_masks

def Trans_Mask():

    trans_masks = {}

    #filename = path + "PolTrans_MASK.h5"
    filename = TransMask
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        for dshort in short_detectors:
            trans_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
            #need to reverse zeros and ones
            trans_masks[dshort] = np.zeros_like(trans_data)
            trans_masks[dshort][trans_data == 0] = 1.0
                
    return trans_masks

def BlockedBeam_Averaged(BlockedBeamFiles, MeasMasks, Trans_masks):

    BlockBeam_Trans = {}
    BlockBeam_ScattPerPixel = {}
    masks = {}

    for filenumber in BlockedBeamFiles:
        filename = path + "sans" + str(filenumber) + ".nxs.ngv"
        config = Path(filename)
        if config.is_file():
            print('Reading in block beam file number:', filenumber)
            f = h5py.File(filename)
            Config_ID = Unique_Config_ID(filenumber) #int(f['entry/DAS_logs/carriage2Trans/desiredSoftPosition'][0]) #int(f['entry/instrument/detector_{ds}/distance'.format(ds=TransPanel)][0])

            Purpose = f['entry/reduction/file_purpose'][()]
            Count_time = f['entry/collection_time'][0]
            if str(Purpose).find("TRANS") != -1 or str(Purpose).find("HE3") != -1:
                
                #Trans_Counts = f['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                
                trans_mask = Trans_masks['MR']
                trans_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
                trans_data = trans_data*trans_mask
                Trans_Counts = trans_data.sum()
                
                BlockBeam_Trans[Config_ID] = {'File' : filenumber,
                                                             'CountsPerSecond' : Trans_Counts/Count_time}
            #if str(Purpose).find("SCATT") != -1:
            if str(Purpose).find("TRANS") != -1 or str(Purpose).find("HE3") != -1:
                print('BB scattering number is ', filenumber)
                BlockBeam_ScattPerPixel[Config_ID] = {'File' : filenumber}
                for dshort in short_detectors:
                    Holder = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                    if Config_ID in MeasMasks:
                        masks = MeasMasks[Config_ID]
                    else:
                        masks[dshort] = np.ones_like(Holder)
                    Sum = np.sum(Holder[masks[dshort] > 0])
                    Pixels = np.sum(masks[dshort])
                    Unc = np.sqrt(Sum)/Pixels
                    Ave = np.average(Holder[masks[dshort] > 0])
                    
                    BlockBeam_ScattPerPixel[Config_ID][dshort] = {'AvePerSec' : Ave/Count_time, 'Unc' : Unc/Count_time}
    '''
    for CF_ID in BlockBeam_Trans:
        if CF_ID not in BlockBeam_ScattPerPixel:
            filenumber = BlockBeam_Trans[CF_ID]['File']
            filename = path + "sans" + str(filenumber) + ".nxs.ngv"
            config = Path(filename)
            if config.is_file():
                f = h5py.File(filename)
                Count_time = f['entry/collection_time'][0]
                BlockBeam_ScattPerPixel[CF_ID] = {'AltFile' : filenumber}
                for dshort in short_detectors:
                    BlockBeam_ScattPerPixel[Config_ID][dshort] = {'AvePerSec' : Ave/Count_time, 'Unc' : Unc/Count_time}
    '''
                
    return BlockBeam_Trans, BlockBeam_ScattPerPixel

def SortData(YesNoManualHe3Entry, New_HE3_Files, MuValues, TeValues, start_number, end_number):

    Unpol_Trans = {}
    Unpol_Scatt = {}
    HalfpolSM_Trans = {}
    HalfpolSM_Scatt = {}
    HE3_Trans = {}
    Pol_Trans = {}
    Pol_Scatt = {}
    Scatt_ConfigIDs = {}
    CellIdentifier = 0
    Recent_T_SM = {}
    HE3OUT_config = -1
    HE3OUT_attenuators = -1
    filenumber = start_number
    while filenumber < end_number + 1:
        filename = path + "sans" + str(filenumber) + ".nxs.ngv"
        config = Path(filename)
        fn = str(filenumber)
        if config.is_file() and filenumber not in Excluded_Files and filenumber not in BlockedBeamFiles and filenumber not in Mask_Files:
            f = h5py.File(filename)
            Listed_Config = str(f['entry/DAS_logs/configuration/key'][0])
            Listed_Config = Listed_Config[2:]
            Listed_Config = Listed_Config[:-1]
            Descrip = str(f['entry/sample/description'][0])
            Descrip = Descrip[2:]
            Descrip = Descrip[:-1]
            print('Reading in file number:', filenumber, '  ', Descrip)
            Sample_Name = Descrip.replace(Listed_Config, '')
            Not_Sample = ['T_UU', 'T_DU', 'T_DD', 'T_UD', 'T_SM', 'HeIN', 'HeOUT', 'S_UU', 'S_DU', 'S_DD', 'S_UD', 'T_NP', 'S_NP']
            for i in Not_Sample:
                Sample_Name = Sample_Name.replace(i, '')
            
            Purpose = f['entry/reduction/file_purpose'][()]
            Intent = f['entry/reduction/intent'][()]
            #ID = str(f['entry/sample/group_id'][0])
            ID = Sample_Name
            End_time = dateutil.parser.parse(f['entry/end_time'][0])
            Count_time = f['entry/collection_time'][0]
            #Trans_Counts = f['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
            trans_mask = Trans_masks['MR']
            trans_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=TransPanel)])
            trans_data = trans_data*trans_mask
            Trans_Counts = trans_data.sum()
            MonCounts = f['entry/control/monitor_counts'][0]
            Trans_Distance = f['entry/instrument/detector_{ds}/distance'.format(ds=TransPanel)][0]
            Attenuation = f['entry/DAS_logs/attenuator/attenuator'][0]
            Wavelength = f['entry/DAS_logs/wavelength/wavelength'][0]
            Config = Unique_Config_ID(filenumber)
            FrontPolDirection = f['entry/DAS_logs/frontPolarization/direction'][()]
            if Use_Adam4021 > 0:
                Voltage = int(f['entry/DAS_logs/adam4021/voltage'][(0)]*100)
                Voltage = Voltage / 100
            else:
                Voltage = NA
            if Use_Lakeshore340 > 0:
                Desired_Temp = f['entry/DAS_logs/temp/desiredPrimaryNode'][0]
            else:
                Desired_Temp = NA

            if filenumber == start_number:
                Scatt_ConfigIDs[Config] = {'Example_File' : [filenumber]}

            if ID in Desired_GroupIDs or Select_GroupIDs == 0:
                if ID not in Undesired_GroupIDs:

                    ID = ID + '_' + str(Voltage) + 'V_' + str(Desired_Temp) + 'K'

                    if "backPolarization" in f['entry/DAS_logs/']:
                        BackPolDirection = f['entry/DAS_logs/backPolarization/direction'][()]
                    else:
                        BackPolDirection = [b'UNPOLARIZED']
                    Type = str(f['entry/sample/description'][()])
    
                    if str(Purpose).find("SCATT") != -1:
                        if str(FrontPolDirection).find("UNPOLARIZED") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                            if ID not in Unpol_Scatt:
                                Unpol_Scatt[ID] = {Config : {'File' :[filenumber]}}
                            elif Config not in Unpol_Scatt[ID]:
                                Unpol_Scatt[ID] = {Config : {'File' :[filenumber]}}
                            else:
                                Unpol_Scatt[ID][Config]['File'].append(filenumber)
                        elif str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                            print('Need to define HalfPol_UP scattering')
                        elif str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                            print('Need to define HalfPol_DOWN scattering')
                        elif str(FrontPolDirection).find("UNPOLARIZED") == -1 and str(BackPolDirection).find("UNPOLARIZED") == -1:#i.e. fully polarized   
                            if ID not in Pol_Scatt:
                                Pol_Scatt[ID] = {Config : {'S_UU_files' : [-1], 'S_DU_files' : [-1], 'S_DD_files' : [-1], 'S_UD_files' : [-1]}}
                            if Config not in Pol_Scatt[ID]:
                                Pol_Scatt[ID][Config] = {'S_UU_files' : [-1], 'S_DU_files' : [-1], 'S_DD_files' : [-1], 'S_UD_files' : [-1]}
                            if Type[-6:-2] == 'S_UU': # and str(BackPolDirection).find("UP") != -1: #and str(FrontPolDirection).find("UP") != -1     
                                if Pol_Scatt[ID][Config]['S_UU_files'] == [-1]:
                                    Pol_Scatt[ID][Config]['S_UU_files'] = [filenumber]
                                else:
                                    Pol_Scatt[ID][Config]['S_UU_files'].append(filenumber)        
                            if Type[-6:-2] == 'S_UD': # and str(BackPolDirection).find("UP") != -1: #and str(FrontPolDirection).find("DOWN") != -1
                                if Pol_Scatt[ID][Config]['S_UD_files'] == [-1]:
                                    Pol_Scatt[ID][Config]['S_UD_files'] = [filenumber]
                                else:
                                    Pol_Scatt[ID][Config]['S_UD_files'].append(filenumber)
                            if Type[-6:-2] == 'S_DD': # and str(BackPolDirection).find("DOWN") != -1: #and str(FrontPolDirection).find("DOWN") != -1
                                if Pol_Scatt[ID][Config]['S_DD_files'] == [-1]:
                                    Pol_Scatt[ID][Config]['S_DD_files'] = [filenumber]
                                else:
                                    Pol_Scatt[ID][Config]['S_DD_files'].append(filenumber)
                            if Type[-6:-2] == 'S_DU': # and str(BackPolDirection).find("DOWN") != -1: #and str(FrontPolDirection).find("UP") != -1
                                if Pol_Scatt[ID][Config]['S_DU_files'] == [-1]:
                                    Pol_Scatt[ID][Config]['S_DU_files'] = [filenumber]
                                else:
                                    Pol_Scatt[ID][Config]['S_DU_files'].append(filenumber)
                        
                    if str(Purpose).find("TRANS") != -1 or str(Purpose).find("HE3") != -1:
                        BlockBeamRate = 0
                        BlockBeam_filenumber = -1
                        if Config in BlockBeam_Trans:
                            BlockBeamRate = BlockBeam_Trans[Config]['CountsPerSecond']
                            BlockBeam_filenumber = BlockBeam_Trans[Config]['File']
                        CountPerMon = (Trans_Counts - BlockBeamRate*Count_time)*1E8/MonCounts
                        
                        if str(FrontPolDirection).find("UNPOLARIZED") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                            if ID not in Unpol_Trans:
                                Unpol_Trans[ID] = {Config : {'File' : [filenumber], 'Abs_Trans': [CountPerMon], 'BlockBeamFile': [BlockBeam_filenumber]}}
                            elif Config not in Unpol_Trans[ID]:
                                Unpol_Trans[ID] = {Config : {'File' : [filenumber], 'Abs_Trans': [CountPerMon], 'BlockBeamFile': [BlockBeam_filenumber]}}
                            else:
                                Unpol_Trans[ID][Config]['File'].append(filenumber)
                                Unpol_Trans[ID][Config]['Abs_Trans'].append(CountPerMon)
                                Unpol_Trans[ID][Config]['BlockBeamFile'].append(BlockBeam_filenumber)
                        elif str(FrontPolDirection).find("UP") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                            print('Need to define HalfPol_UP transmission')
                        elif str(FrontPolDirection).find("DOWN") != -1 and str(BackPolDirection).find("UNPOLARIZED") != -1:
                            print('Need to define HalfPol_DOWN transmission')

                            
                        if str(FrontPolDirection).find("UNPOLARIZED") == -1:
                            if str(BackPolDirection).find("UNPOLARIZED") == -1 or Type[-6:-2] == 'T_SM': #i.e. fully polarized
                                if Type[-6:-2] == 'T_UU':
                                    UU_Transfilenumber = filenumber
                                    UU_Trans = (Trans_Counts - BlockBeamRate*Count_time)*1E8/MonCounts
                                    UU_Trans_ID = ID
                                    UU_Trans_Config = Config
                                    UU_Trans_Attn = Attenuation
                                    UU_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                elif Type[-6:-2] == 'T_DU':
                                    DU_Transfilenumber = filenumber
                                    DU_Trans = (Trans_Counts - BlockBeamRate*Count_time)*1E8/MonCounts
                                    DU_Trans_ID = ID
                                    DU_Trans_Config = Config
                                    DU_Trans_Attn = Attenuation
                                    DU_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                elif Type[-6:-2] == 'T_DD':
                                    DD_Transfilenumber = filenumber
                                    DD_Trans = (Trans_Counts - BlockBeamRate*Count_time)*1E8/MonCounts
                                    DD_Trans_ID = ID
                                    DD_Trans_Config = Config
                                    DD_Trans_Attn = Attenuation
                                    DD_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                elif Type[-6:-2] == 'T_UD':
                                    UD_Transfilenumber = filenumber
                                    UD_Trans = (Trans_Counts - BlockBeamRate*Count_time)*1E8/MonCounts
                                    UD_Trans_ID = ID
                                    UD_Trans_Config = Config
                                    UD_Trans_Attn = Attenuation
                                    UD_Time = (End_time.timestamp() - Count_time/2)/3600.0
                                elif Type[-6:-2] == 'T_SM':
                                    SM_Transfilenumber = filenumber
                                    SM_Trans = (Trans_Counts - BlockBeamRate*Count_time)*1E8/MonCounts
                                    SM_Trans_ID = ID
                                    SM_Trans_Config = Config
                                    SM_Trans_Attn = Attenuation

                                    if UU_Trans_Config == DD_Trans_Config and UD_Trans_Config == DU_Trans_Config and UU_Trans_Config == UD_Trans_Config:
                                        if UU_Trans_ID == DD_Trans_ID and UD_Trans_ID == DU_Trans_ID and UU_Trans_ID == UD_Trans_ID:
                                            if UU_Trans_Attn == DD_Trans_Attn and UD_Trans_Attn == DU_Trans_Attn and UU_Trans_Attn == UD_Trans_Attn:
                                                if ID not in Pol_Trans:
                                                    Pol_Trans[ID] = {'T_UU': {'File': [UU_Transfilenumber], 'Trans' : [UU_Trans/SM_Trans], 'Meas_Time' : [UU_Time]},
                                                         'T_DU': {'File': [DU_Transfilenumber], 'Trans' : [DU_Trans/SM_Trans], 'Meas_Time' : [DU_Time]},
                                                         'T_DD': {'File': [DD_Transfilenumber], 'Trans' : [DD_Trans/SM_Trans], 'Meas_Time' : [DD_Time]},
                                                         'T_UD': {'File': [UD_Transfilenumber], 'Trans' : [UD_Trans/SM_Trans], 'Meas_Time' : [UD_Time]},
                                                         'T_SM': {'File': [SM_Transfilenumber], 'Abs_Trans' : [SM_Trans]},
                                                         'BlockBeam': {'File': [BlockBeam_filenumber]},
                                                         'Config' : [UU_Trans_Config]
                                                         }
                                                else:
                                                    Pol_Trans[ID]['T_UU']['File'].append(UU_Transfilenumber)
                                                    Pol_Trans[ID]['T_UU']['Trans'].append(UU_Trans/SM_Trans)
                                                    Pol_Trans[ID]['T_UU']['Meas_Time'].append(UU_Time)
                                                    Pol_Trans[ID]['T_DU']['File'].append(DU_Transfilenumber)
                                                    Pol_Trans[ID]['T_DU']['Trans'].append(DU_Trans/SM_Trans)
                                                    Pol_Trans[ID]['T_DU']['Meas_Time'].append(DU_Time)
                                                    Pol_Trans[ID]['T_DD']['File'].append(DD_Transfilenumber)
                                                    Pol_Trans[ID]['T_DD']['Trans'].append(DD_Trans/SM_Trans)
                                                    Pol_Trans[ID]['T_DD']['Meas_Time'].append(DD_Time)
                                                    Pol_Trans[ID]['T_UD']['File'].append(UD_Transfilenumber)
                                                    Pol_Trans[ID]['T_UD']['Trans'].append(UD_Trans/SM_Trans)
                                                    Pol_Trans[ID]['T_UD']['Meas_Time'].append(UD_Time)
                                                    Pol_Trans[ID]['T_SM']['File'].append(SM_Transfilenumber)
                                                    Pol_Trans[ID]['T_SM']['Abs_Trans'].append(SM_Trans)
                                                    Pol_Trans[ID]['BlockBeam']['File'].append(BlockBeam_filenumber)
                                                    Pol_Trans[ID]['Config'].append(UU_Trans_Config)

            if str(Purpose).find("HE3") != -1:
                if YesNoManualHe3Entry == 1:
                    if filenumber in New_HE3_Files:
                        ScaledOpacity = MuValues[CellIdentifier]
                        TE = TeValues[CellIdentifier]
                        #HE3Insert_Time = (End_time.timestamp() - Count_time)/3600.0
                        CellTimeIdentifier = (End_time.timestamp() - Count_time)/3600.0
                        #f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                        HE3Insert_Time = (End_time.timestamp() - Count_time)/3600.0
                        CellIdentifier += 1
                        
                else: #i.e. YesNoManualHe3Entry != 1
                    CellTimeIdentifier = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                    if CellTimeIdentifier not in HE3_Trans:
                        HE3Insert_Time = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                        Opacity = f['/entry/DAS_logs/backPolarization/opacityAt1Ang'][0]
                        Wavelength = f['/entry/DAS_logs/wavelength/wavelength'][0]
                        ScaledOpacity = Opacity*Wavelength
                        TE = f['/entry/DAS_logs/backPolarization/glassTransmission'][0]
                           
                HE3Type = str(f['entry/sample/description'][()])
                if HE3Type[-7:-2] == 'HeOUT':
                    HE3OUT_filenumber = filenumber
                    HE3OUT_config = Config
                    HE3OUT_attenuators = int(f['entry/instrument/attenuator/num_atten_dropped'][0])
                    HE3OUT_counts = Trans_Counts
                    HE3OUT_mon = MonCounts
                    HE3OUT_count_time = Count_time
                elif HE3Type[-7:-2] == ' HeIN':
                    HE3IN_filenumber = filenumber
                    HE3IN_config = Config
                    HE3IN_attenuators = int(f['entry/instrument/attenuator/num_atten_dropped'][0])
                    HE3IN_counts = Trans_Counts
                    HE3IN_mon = MonCounts
                    HE3IN_count_time = Count_time
                    HE3IN_StartTime = (End_time.timestamp() - Count_time/2)/3600.0
                    if HE3OUT_config == HE3IN_config and HE3OUT_attenuators == HE3IN_attenuators: #This implies that you must have a 3He out before 3He in of same config and atten
                        if HE3Insert_Time not in HE3_Trans:
                            HE3_Trans[CellTimeIdentifier] = {'Te' : TE,
                                                         'Mu' : ScaledOpacity,
                                                         'Insert_time' : HE3Insert_Time}
                        Elasped_time = HE3IN_StartTime - HE3Insert_Time
                        BlockBeamRate = 0
                        BlockBeam_filenumber = 0
                        if HE3IN_config in BlockBeam_Trans:
                            BlockBeamRate = BlockBeam_Trans[Config]['CountsPerSecond']
                            BlockBeam_filenumber = BlockBeam_Trans[Config]['File']
                        HE3_transmission = (HE3IN_counts - BlockBeamRate*HE3IN_count_time)/HE3IN_mon
                        HE3_transmission = HE3_transmission /((HE3OUT_counts - BlockBeamRate*HE3OUT_count_time)/HE3OUT_mon)
                        if "Transmission" not in HE3_Trans[CellTimeIdentifier]:
                            HE3_Trans[CellTimeIdentifier]['HE3_OUT_file'] = [HE3OUT_filenumber]
                            HE3_Trans[CellTimeIdentifier]['HE3_IN_file'] = [HE3IN_filenumber]
                            HE3_Trans[CellTimeIdentifier]['BlockBeam_file'] = [BlockBeam_filenumber]
                            HE3_Trans[CellTimeIdentifier]['Elasped_time'] = [Elasped_time]
                            HE3_Trans[CellTimeIdentifier]['Transmission'] = [HE3_transmission]
                        else:
                            HE3_Trans[CellTimeIdentifier]['HE3_OUT_file'].append(HE3OUT_filenumber)
                            HE3_Trans[CellTimeIdentifier]['HE3_IN_file'].append(HE3IN_filenumber)
                            HE3_Trans[CellTimeIdentifier]['BlockBeam_file'].append(BlockBeam_filenumber)
                            HE3_Trans[CellTimeIdentifier]['Elasped_time'].append(Elasped_time)
                            HE3_Trans[CellTimeIdentifier]['Transmission'].append(HE3_transmission)
                        
                                
                        

        filenumber += 1

    for ID in Unpol_Scatt:
        for Config_ID in Unpol_Scatt[ID]:
            if Config_ID not in Scatt_ConfigIDs:
                print('Unpol ', Config_ID, 'not in', Scatt_ConfigIDs)
                filenumber = Unpol_Scatt[ID][Config_ID]['File'][0]
                Scatt_ConfigIDs[Config_ID] = {'Example_File' : [filenumber]}
    for ID in Pol_Scatt:
        for Config_ID in Pol_Scatt[ID]:
            if Config_ID not in Scatt_ConfigIDs:
                print('Pol ', Config_ID, 'not in', Scatt_ConfigIDs)
            if Config_ID not in Scatt_ConfigIDs and Pol_Scatt[ID][Config_ID]['S_UU_files'][0] != -1:
                filenumber = Pol_Scatt[ID][Config_ID]['S_UU_files'][0]
                Scatt_ConfigIDs[Config_ID] = {Scatt_ConfigIDs[Config_ID] : [filenumber]}


    return  Unpol_Trans, Unpol_Scatt, HE3_Trans, Pol_Trans, Pol_Scatt, Scatt_ConfigIDs

def SolidAngle_AllDetectors(representative_filenumber):
    Solid_Angle = {}
    filename = path + "sans" + str(representative_filenumber) + ".nxs.ngv"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        for dshort in short_detectors:
            detector_distance = f['entry/instrument/detector_{ds}/distance'.format(ds=dshort)][0]
            x_pixel_size = f['entry/instrument/detector_{ds}/x_pixel_size'.format(ds=dshort)][0]/10.0
            y_pixel_size = f['entry/instrument/detector_{ds}/y_pixel_size'.format(ds=dshort)][0]/10.0
            if dshort == 'MT' or dshort == 'MB' or dshort == 'FT' or dshort == 'FB':
                setback = f['entry/instrument/detector_{ds}/setback'.format(ds=dshort)][0]
            else:
                setback = 0
                
            realDistZ = detector_distance + setback
            theta_x_step = x_pixel_size / realDistZ
            theta_y_step = y_pixel_size / realDistZ
            Solid_Angle[dshort] = theta_x_step * theta_y_step

    return Solid_Angle
            
def QCalculationAndMasks_AllDetectors(representative_filenumber, AngleWidth):

    BeamStopShadow = {}
    Mask_Right = {}
    Mask_Left = {}
    Mask_Top = {}
    Mask_Bottom = {}
    Mask_DiagonalCW = {}
    Mask_DiagonalCCW = {}
    Mask_None = {}
    Mask_User_Defined = {}
    Q_total = {}
    deltaQ = {}
    Qx = {}
    Qy = {}
    Qz = {}
    Q_perp_unc = {}
    Q_parl_unc = {}
    dimXX = {}
    dimYY = {}

    filename = path + "sans" + str(representative_filenumber) + ".nxs.ngv"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        for dshort in short_detectors:
            data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
            Wavelength = f['entry/instrument/beam/monochromator/wavelength'][0]
            Wavelength_spread = f['entry/instrument/beam/monochromator/wavelength_spread'][0]
            dimX = f['entry/instrument/detector_{ds}/pixel_num_x'.format(ds=dshort)][0]
            dimY = f['entry/instrument/detector_{ds}/pixel_num_y'.format(ds=dshort)][0]
            dimXX[dshort] = f['entry/instrument/detector_{ds}/pixel_num_x'.format(ds=dshort)][0]
            dimYY[dshort] = f['entry/instrument/detector_{ds}/pixel_num_y'.format(ds=dshort)][0]
            beam_center_x = f['entry/instrument/detector_{ds}/beam_center_x'.format(ds=dshort)][0]
            beam_center_y = f['entry/instrument/detector_{ds}/beam_center_y'.format(ds=dshort)][0]
            beamstop_diameter = f['/entry/DAS_logs/C2BeamStop/diameter'][0]/10.0 #beam stop in cm; sits right in front of middle detector?
            detector_distance = f['entry/instrument/detector_{ds}/distance'.format(ds=dshort)][0]
            x_pixel_size = f['entry/instrument/detector_{ds}/x_pixel_size'.format(ds=dshort)][0]/10.0
            y_pixel_size = f['entry/instrument/detector_{ds}/y_pixel_size'.format(ds=dshort)][0]/10.0
            panel_gap = f['entry/instrument/detector_{ds}/panel_gap'.format(ds=dshort)][0]/10.0
            coeffs = f['entry/instrument/detector_{ds}/spatial_calibration'.format(ds=dshort)][0][0]/10.0
            SampleApInternal = f['/entry/DAS_logs/geometry/internalSampleApertureHeight'][0] #internal sample aperture in cm
            SampleApExternal = f['/entry/DAS_logs/geometry/externalSampleApertureHeight'][0] #external sample aperture in cm
            SourceAp = f['/entry/DAS_logs/geometry/sourceApertureHeight'][0] #source aperture in cm, assumes circular aperture(?) #0.75, 1.5, or 3 for guides; otherwise 6 cm for >= 1 guides
            FrontDetToGateValve = f['/entry/DAS_logs/carriage/frontTrans'][0] #400
            MiddleDetToGateValve = f['/entry/DAS_logs/carriage/middleTrans'][0] #1650
            #GateValveToSample = f['/entry/DAS_logs/geometry/samplePositionOffset'][0] #e.g. 91.4; gate valve to sample in cm ("Hand-measured distance from the center of the table the sample is mounted on to the sample. A positive value means the sample is offset towards the guides.")
            FrontDetToSample = f['/entry/DAS_logs/geometry/sampleToFrontLeftDetector'][0] #491.4
            MiddleDetToSample = f['/entry/DAS_logs/geometry/sampleToMiddleLeftDetector'][0] #1741.4
            #SampleToSampleAp = f['/entry/DAS_logs/geometry/SampleApertureOffset'][0] #e.g. 106.9; sample to sample aperture in cm ("Hand-measured distance between the Sample aperture and the sample.")            
            SampleToSourceAp = f['/entry/DAS_logs/geometry/sourceApertureToSample'][0] #1490.6; "Calculated distance between sample and source aperture" in cm
            #SampleApToSourceAp = f['/entry/DAS_logs/geometry/sourceApertureToSampleAperture'][0] #1383.7; "Calculated distance between sample aperture and source aperture" in cm
            #Note gate valve to source aperture distances are based on the number of guides used:
            #0=2441; 1=2157; 2=1976; 3=1782; 4=1582; 5=1381; 6=1181; 7=980; 8=780; 9=579 in form of # guides=distance in cm

            if dshort == 'MT' or dshort == 'MB' or dshort == 'FT' or dshort == 'FB':
                setback = f['entry/instrument/detector_{ds}/setback'.format(ds=dshort)][0]
                vertical_offset = f['entry/instrument/detector_{ds}/vertical_offset'.format(ds=dshort)][0]
                lateral_offset = 0
            else:
                setback = 0
                vertical_offset = 0
                lateral_offset = f['entry/instrument/detector_{ds}/lateral_offset'.format(ds=dshort)][0]

            realDistZ = detector_distance + setback

            position_key = dshort[1]
            if position_key == 'T':
                realDistX =  coeffs
                realDistY =  0.5 * y_pixel_size + vertical_offset + panel_gap/2.0
            elif position_key == 'B':
                realDistX =  coeffs
                realDistY =  vertical_offset - (dimY - 0.5)*y_pixel_size - panel_gap/2.0
            elif position_key == 'L':
                realDistX =  lateral_offset - (dimX - 0.5)*x_pixel_size - panel_gap/2.0
                realDistY =  coeffs
            elif position_key == 'R':
                realDistX =  x_pixel_size*(0.5) + lateral_offset + panel_gap/2.0
                realDistY =  coeffs

            X, Y = np.indices(data.shape)
            BSS = np.ones_like(data)
            x0_pos =  realDistX - beam_center_x + (X)*x_pixel_size 
            y0_pos =  realDistY - beam_center_y + (Y)*y_pixel_size
            InPlane0_pos = np.sqrt(x0_pos**2 + y0_pos**2)
            BSS[InPlane0_pos < beamstop_diameter/2.0] = 0.0
            BeamStopShadow[dshort] = BSS
            twotheta = np.arctan2(InPlane0_pos,realDistZ)
            phi = np.arctan2(y0_pos,x0_pos)

            #Q resolution from J. of Appl. Cryst. 44, 1127-1129 (2011) and file:///C:/Users/kkrycka/Downloads/SANS_2D_Resolution.pdf where
            #there seems to be an extra factor of wavelength listed that shouldn't be there in (delta_wavelength/wavelength):
            carriage_key = dshort[0]
            if carriage_key == 'F':
                L2 = FrontDetToSample
            elif carriage_key == 'M':
                L2 = MiddleDetToSample
            L1 = SampleToSourceAp
            Pix = 0.82
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
            #If no gravity correction:
            #SigmaQParlSqr = SigmaQParlSqr + np.power(Q0,2)*np.power(Wavelength_spread/np.sqrt(6.0),2)
            #Else, if adding gravity correction:
            g = 981 #in cm/s^2
            m_div_h = 252.77 #in s cm^-2
            A = -0.5*981*L2*(L1+L2)*np.power(m_div_h , 2)
            WL = Wavelength*1E-8
            SigmaQParlSqr = SigmaQParlSqr + np.power(Wavelength_spread*k/(L2),2)*(R*R -4*A*np.sin(phi)*WL*WL + 4*A*A*np.power(WL,4))/6.0 #gravity correction makes vary little difference for wavelength spread < 20%
            #VSANS IGOR 2D ASCII delta_Q seems to be way off the mark, but this 2D calculaation matches the VSANS circular average closely when pixels are converted to circular average...
            
            #This is what Greta had in comperaison, with Pixel size set to 0.82 cm:
            #deltaQ_geometry = (2.0*np.pi/(Wavelength*L2))*np.sqrt( np.power((L2*R1)/(4*L1),2) + np.power((L1+L2)/(4*L1*R1),2)+ np.power( (Pix/2.0),(2.0/3.0)) )
            #deltaQ_wavelength = Wavelength_spread/np.sqrt(6.0)
            
            Q_total[dshort] = (4.0*np.pi/Wavelength)*np.sin(twotheta/2.0)
            QQ_total = (4.0*np.pi/Wavelength)*np.sin(twotheta/2.0)
            Qx[dshort] = QQ_total*np.cos(twotheta/2.0)*np.cos(phi)
            Qy[dshort] = QQ_total*np.cos(twotheta/2.0)*np.sin(phi)
            Qz[dshort] = QQ_total*np.sin(twotheta/2.0)     
            Q_perp_unc[dshort] = np.ones_like(Q_total[dshort])*np.sqrt(SigmaQPerpSqr)
            Q_parl_unc[dshort] = np.sqrt(SigmaQParlSqr)
            Theta_deg = 180.0*np.arctan2(Qy[dshort], Qx[dshort])/np.pi #returns values between -180.0 degrees and +180.0 degrees
        
            NM = np.ones_like(data)
            TM = np.zeros_like(data)
            BM = np.zeros_like(data)
            LUM = np.zeros_like(data)
            LLM = np.zeros_like(data)
            RM = np.zeros_like(data)
            DM1 = np.zeros_like(data)
            DM2 = np.zeros_like(data)
            DM3 = np.zeros_like(data)
            DM4 = np.zeros_like(data)
            TM[np.absolute(Theta_deg - 90.0) <= AngleWidth] = 1.0
            BM[np.absolute(Theta_deg + 90.0) <= AngleWidth] = 1.0
            RM[np.absolute(Theta_deg - 0.0) <= AngleWidth] = 1.0
            LUM[np.absolute(Theta_deg - 180.0) <= AngleWidth] = 1.0
            LLM[np.absolute(Theta_deg + 180.0) <= AngleWidth] = 1.0
            LM = LUM + LLM
            DM1[np.absolute(Theta_deg - 45.0) <= AngleWidth] = 1.0
            DM2[np.absolute(Theta_deg + 135.0) <= AngleWidth] = 1.0
            DM3[np.absolute(Theta_deg - 135.0) <= AngleWidth] = 1.0
            DM4[np.absolute(Theta_deg + 45.0) <= AngleWidth] = 1.0
            QyShift = Qy[dshort] - 0.04
            SiliconWindow = np.zeros_like(data)
            SiliconWindow[np.sqrt(Qx[dshort]*Qx[dshort] + QyShift*QyShift) <= 0.123] = 1.0
            Shadow = np.zeros_like(data)
            if dshort == "MT" or dshort == "MB":
                Shadow[np.absolute(X - 64) <= 2] = 1.0
            if dshort == "ML":
                Shadow[np.absolute(Y - 68) <= 50] = 1.0
            if dshort == "ML":
                Shadow[np.absolute(X - 26) <= 1] = 0.0
            if dshort == "MR":
                Shadow[np.absolute(Y - 68) <= 50] = 1.0
            if dshort == "FT" or dshort == "FB":
                Shadow[np.absolute(X - 64) <= 40] = 1.0

            Mask_None[dshort] = NM
            Mask_Right[dshort] = RM
            Mask_Left[dshort] = LM
            Mask_Top[dshort] = TM
            Mask_Bottom[dshort] = BM
            Mask_DiagonalCW[dshort] = DM1 + DM2
            Mask_DiagonalCCW[dshort] = DM3 + DM4
            if dshort == "MT" or dshort == "MB" or dshort == "ML" or dshort == "MR":
                Mask_User_Defined[dshort] = Shadow
            else:
                Mask_User_Defined[dshort] = Shadow

    return Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, dimXX, dimYY, Mask_Right, Mask_Top, Mask_Left, Mask_Bottom, Mask_DiagonalCW, Mask_DiagonalCCW, Mask_None, Mask_User_Defined, BeamStopShadow

def SliceDataUnpolData(Q_min, Q_max, Q_bins, QGridPerDetector, masks, Data_AllDetectors, Unc_Data_AllDetectors, dimXX, dimYY, ID, Config, PlotYesNo):

    Key = masks['label']
    GroupID = ID
    print('Plotting and saving {type} cuts for GroupID {idnum} at Configuration {cf}'.format(type=Key, idnum=ID, cf = Config))
    Q_Values = np.linspace(Q_min, Q_max, Q_bins, endpoint=True)
    Q_step = (Q_max - Q_min) / Q_bins  
    Front = np.zeros_like(Q_Values)
    Front_Unc = np.zeros_like(Q_Values)
    FrontMeanQ = np.zeros_like(Q_Values)
    FrontMeanQUnc = np.zeros_like(Q_Values)
    FrontPixels = np.zeros_like(Q_Values)
    Middle = np.zeros_like(Q_Values)
    Middle_Unc = np.zeros_like(Q_Values)
    MiddleMeanQ = np.zeros_like(Q_Values)
    MiddleMeanQUnc = np.zeros_like(Q_Values)
    MiddlePixels = np.zeros_like(Q_Values)
    for dshort in short_detectors:
        dimX = dimXX[dshort]
        dimY = dimYY[dshort]
        Q_tot = QGridPerDetector['Q_total'][dshort][:][:]
        Q_unc = np.sqrt(np.power(QGridPerDetector['Q_perp_unc'][dshort][:][:],2) + np.power(QGridPerDetector['Q_parl_unc'][dshort][:][:],2))
        Unpol = Data_AllDetectors[dshort][:][:]
        Unpol = Unpol.reshape((dimX, dimY))
        Unpol_Unc = Unc_Data_AllDetectors[dshort][:][:]
        Unpol_Unc = Unpol_Unc.reshape((dimX, dimY))
        #Slice:
        Exp_bins = np.linspace(Q_min, Q_max + Q_step, Q_bins + 1, endpoint=True)
        counts, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=Unpol[masks[dshort] > 0])
        counts_Unc, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power(Unpol_Unc[masks[dshort] > 0],2))
        MeanQSum, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=Q_tot[masks[dshort] > 0])
        #MeanQUnc, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power(Q_unc[masks[dshort] > 0],2))
        MeanQUnc, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=Q_unc[masks[dshort] > 0]) 
        pixels, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.ones_like(Unpol)[masks[dshort] > 0])
        carriage_key = dshort[0]
        if carriage_key == 'F':
            Front += counts
            Front_Unc += counts_Unc
            FrontMeanQ += MeanQSum
            FrontMeanQUnc += MeanQUnc
            FrontPixels += pixels
        elif carriage_key == 'M':
            Middle += counts
            Middle_Unc += counts_Unc
            MiddleMeanQ += MeanQSum
            MiddleMeanQUnc += MeanQUnc
            MiddlePixels += pixels
    nonzero_front_mask = (FrontPixels > 0) #True False map
    nonzero_middle_mask = (MiddlePixels > 0) #True False map
    Q_Front = Q_Values[nonzero_front_mask]
    MeanQ_Front = FrontMeanQ[nonzero_front_mask] / FrontPixels[nonzero_front_mask]
    #MeanQUnc_Front = np.sqrt(FrontMeanQUnc[nonzero_front_mask]) / FrontPixels[nonzero_front_mask]
    MeanQUnc_Front = FrontMeanQUnc[nonzero_front_mask] / FrontPixels[nonzero_front_mask]
    Unpol_Front = Front[nonzero_front_mask] / FrontPixels[nonzero_front_mask]
    Q_Middle = Q_Values[nonzero_middle_mask]
    MeanQ_Middle = MiddleMeanQ[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask]
    #MeanQUnc_Middle = np.sqrt(MiddleMeanQUnc[nonzero_middle_mask]) / MiddlePixels[nonzero_middle_mask]
    MeanQUnc_Middle = MiddleMeanQUnc[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask]
    Unpol_Middle = Middle[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask]

    Sigma_Front = np.sqrt(Front_Unc[nonzero_front_mask]) / FrontPixels[nonzero_front_mask]
    Sigma_Middle = np.sqrt(Middle_Unc[nonzero_middle_mask]) / MiddlePixels[nonzero_middle_mask]

    Unpol_Middle = Unpol_Middle - LowQ_Offset_Unpol

    #***
    if UncertaintyMode == 0:#Varience between pixels udsed for sigma
        Front_DiffSqrd = np.zeros_like(Q_Values)
        Middle_DiffSqrd = np.zeros_like(Q_Values)
        #Calculate average intensity per Q_bin
        FrontPixels_Modified = FrontPixels
        FrontPixels_Modified[FrontPixels <= 0] = 1
        MiddlePixels_Modified = MiddlePixels
        MiddlePixels_Modified[MiddlePixels <= 0] = 1
        MUnpolAve = Middle /MiddlePixels_Modified
        FUnpolAve = Front/FrontPixels_Modified
        for dshort in short_detectors:
            dimX = dimXX[dshort]
            dimY = dimYY[dshort]
            Q_tot = QGridPerDetector['Q_total'][dshort][:][:]
            Unpol = Data_AllDetectors[dshort][:][:]
            Unpol = Unpol.reshape((dimX, dimY))
            carriage_key = dshort[0]
            Q_tot_modified = Q_tot
            Q_tot_modified[Q_tot > Q_max] = Q_max
            Q_tot_modified[Q_tot < Q_min] = Q_min
            inds = np.digitize(Q_tot_modified, Exp_bins, right=True) - 1
            carriage_key = dshort[0]
            if carriage_key == 'F':
                Unpol_Ave = FUnpolAve[inds]
            if carriage_key == 'M':
                Unpol_Ave = MUnpolAve[inds]    
            DiffUnpolSqrd, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power((Unpol[masks[dshort] > 0] - Unpol_Ave[masks[dshort] > 0]),2))
            if carriage_key == 'F':
                Front_DiffSqrd += DiffUnpolSqrd
            elif carriage_key == 'M':
                Middle_DiffSqrd += DiffUnpolSqrd
        Sigma_Front = np.sqrt(Front_DiffSqrd[nonzero_front_mask] / FrontPixels[nonzero_front_mask])
        Sigma_Middle = np.sqrt(Middle_DiffSqrd[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask])
        #***

    Q_Common = np.concatenate((Q_Middle, Q_Front), axis=0)
    Q_Mean = np.concatenate((MeanQ_Middle, MeanQ_Front), axis=0)
    Q_Uncertainty = np.concatenate((MeanQUnc_Middle, MeanQUnc_Front), axis=0)
    Unpol = np.concatenate((Unpol_Middle, Unpol_Front), axis=0)
    Sigma = np.concatenate((Sigma_Middle, Sigma_Front), axis=0)
    Shadow = np.ones_like(Q_Common)

    if PlotYesNo == 1:
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.errorbar(Q_Front, Unpol_Front, yerr=Sigma_Front, fmt = 'b*', label='Front, Unpol')
        ax.errorbar(Q_Middle, Unpol_Middle, yerr=Sigma_Middle, fmt = 'g*', label='Middle, Unpol')
        '''
        #If don't want to plot error bars:
        plt.loglog(Q_Middle, Unpol_Middle, 'b*', label='Middle, Unpol')
        plt.loglog(Q_Front, Unpol_Front, 'g*', label='Front, Unpol')
        '''
        plt.xlabel('Q')
        plt.ylabel('Intensity')
        plt.title('Unpol {keyword} Cuts for ID = {idnum} and Config = {cf}'.format(keyword=Key, idnum=GroupID, cf = Config))
        plt.legend()
        fig.savefig('{keyword}Unpol_Cuts_ID{idnum}_CF{cf}.png'.format(keyword=Key, idnum=GroupID, cf = Config))
        plt.show()

        #QQ, QM, UU, DU, DD, UD = SliceDataPolData(Q_min, Q_max, Q_bins, Q_total, ChosenMasks, PolCorr_AllDetectors, dimXX, dimYY, GroupID, PlotYesNo)
        text_output = np.array([Q_Common, Unpol, Sigma, Q_Uncertainty, Q_Mean, Shadow])
        text_output = text_output.T
        np.savetxt('{key}Unpol_ID={idnum}Config={cf}.txt'.format(key=Key, idnum=ID, cf = Config), text_output, header='Q, I, DI, DQ, MeanQ, Shadow', fmt='%1.4e')
    
    return Q_Common, Q_Mean, Unpol

def SliceDataPolData(Q_min, Q_max, Q_bins, QGridPerDetector, masks, PolCorr_AllDetectors, Unc_PolCorr_AllDetectors, dimXX, dimYY, ID, Config, PlotYesNo):
    
    Key = masks['label']
    print('Plotting and saving {type} cuts for GroupID {idnum} at Configuration {cf}'.format(type=Key, idnum=ID, cf = Config))
    Q_Values = np.linspace(Q_min, Q_max, Q_bins, endpoint=True)
    Q_step = (Q_max - Q_min) / Q_bins  
    FrontUU = np.zeros_like(Q_Values)
    FrontDU = np.zeros_like(Q_Values)
    FrontDD = np.zeros_like(Q_Values)
    FrontUD = np.zeros_like(Q_Values)
    FrontUU_Unc = np.zeros_like(Q_Values)
    FrontDU_Unc = np.zeros_like(Q_Values)
    FrontDD_Unc = np.zeros_like(Q_Values)
    FrontUD_Unc = np.zeros_like(Q_Values)
    FrontMeanQ = np.zeros_like(Q_Values)
    FrontMeanQUnc = np.zeros_like(Q_Values)
    FrontPixels = np.zeros_like(Q_Values)
    MiddleUU = np.zeros_like(Q_Values)
    MiddleDU = np.zeros_like(Q_Values)
    MiddleDD = np.zeros_like(Q_Values)
    MiddleUD = np.zeros_like(Q_Values)
    MiddleUU_Unc = np.zeros_like(Q_Values)
    MiddleDU_Unc = np.zeros_like(Q_Values)
    MiddleDD_Unc = np.zeros_like(Q_Values)
    MiddleUD_Unc = np.zeros_like(Q_Values)
    MiddleMeanQ = np.zeros_like(Q_Values)
    MiddleMeanQUnc = np.zeros_like(Q_Values)
    MiddlePixels = np.zeros_like(Q_Values)
    
    for dshort in short_detectors:
        dimX = dimXX[dshort]
        dimY = dimYY[dshort]
        Q_tot = QGridPerDetector['Q_total'][dshort][:][:]
        Q_unc = np.sqrt(np.power(QGridPerDetector['Q_perp_unc'][dshort][:][:],2) + np.power(QGridPerDetector['Q_parl_unc'][dshort][:][:],2))
        UU = PolCorr_AllDetectors[dshort][0][:][:]
        UU = UU.reshape((dimX, dimY))
        DU = PolCorr_AllDetectors[dshort][1][:][:]
        DU = DU.reshape((dimX, dimY))
        DD = PolCorr_AllDetectors[dshort][2][:][:]
        DD = DD.reshape((dimX, dimY))
        UD = PolCorr_AllDetectors[dshort][3][:][:]
        UD = UD.reshape((dimX, dimY))
        UU_Unc = Unc_PolCorr_AllDetectors[dshort][0][:][:]
        UU_Unc = UU_Unc.reshape((dimX, dimY))
        DU_Unc = Unc_PolCorr_AllDetectors[dshort][1][:][:]
        DU_Unc = DU_Unc.reshape((dimX, dimY))
        DD_Unc = Unc_PolCorr_AllDetectors[dshort][2][:][:]
        DD_Unc = DD_Unc.reshape((dimX, dimY))
        UD_Unc = Unc_PolCorr_AllDetectors[dshort][3][:][:]
        UD_Unc = UD_Unc.reshape((dimX, dimY))

        Exp_bins = np.linspace(Q_min, Q_max + Q_step, Q_bins + 1, endpoint=True)
        countsUU, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=UU[masks[dshort] > 0])
        countsDU, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=DU[masks[dshort] > 0])
        countsDD, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=DD[masks[dshort] > 0])
        countsUD, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=UD[masks[dshort] > 0])
        
        UncUU, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power(UU_Unc[masks[dshort] > 0],2))
        UncDU, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power(DU_Unc[masks[dshort] > 0],2))
        UncDD, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power(DD_Unc[masks[dshort] > 0],2))
        UncUD, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power(UD_Unc[masks[dshort] > 0],2))
        
        MeanQSum, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=Q_tot[masks[dshort] > 0])
        MeanQUnc, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power(Q_unc[masks[dshort] > 0],2)) 
        pixels, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.ones_like(UU)[masks[dshort] > 0])  
        carriage_key = dshort[0]
        if carriage_key == 'F':
            FrontUU += countsUU
            FrontDU += countsDU
            FrontDD += countsDD
            FrontUD += countsUD
            FrontUU_Unc += UncUU
            FrontDU_Unc += UncDU
            FrontDD_Unc += UncDD
            FrontUD_Unc += UncUD
            FrontMeanQ += MeanQSum
            FrontMeanQUnc += MeanQUnc
            FrontPixels += pixels
        elif carriage_key == 'M':
            MiddleUU += countsUU
            MiddleDU += countsDU
            MiddleDD += countsDD
            MiddleUD += countsUD
            MiddleUU_Unc += UncUU
            MiddleDU_Unc += UncDU
            MiddleDD_Unc += UncDD
            MiddleUD_Unc += UncUD
            MiddleMeanQ += MeanQSum
            MiddleMeanQUnc += MeanQUnc
            MiddlePixels += pixels

    nonzero_front_mask = (FrontPixels > 0) #True False map
    nonzero_middle_mask = (MiddlePixels > 0) #True False map
    Q_Front = Q_Values[nonzero_front_mask]
    MeanQ_Front = FrontMeanQ[nonzero_front_mask] / FrontPixels[nonzero_front_mask]
    MeanQUnc_Front = np.sqrt(FrontMeanQUnc[nonzero_front_mask]) / FrontPixels[nonzero_front_mask]
    UUF = FrontUU[nonzero_front_mask] / FrontPixels[nonzero_front_mask]
    DUF = FrontDU[nonzero_front_mask] / FrontPixels[nonzero_front_mask]
    DDF = FrontDD[nonzero_front_mask] / FrontPixels[nonzero_front_mask]
    UDF = FrontUD[nonzero_front_mask] / FrontPixels[nonzero_front_mask]
    Q_Middle = Q_Values[nonzero_middle_mask]
    MeanQ_Middle = MiddleMeanQ[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask]
    MeanQUnc_Middle = np.sqrt(MiddleMeanQUnc[nonzero_middle_mask]) / MiddlePixels[nonzero_middle_mask]
    UUM = MiddleUU[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask]
    DUM = MiddleDU[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask]
    DDM = MiddleDD[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask]
    UDM = MiddleUD[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask]

    Sigma_UUF = np.sqrt(FrontUU_Unc[nonzero_front_mask]) / FrontPixels[nonzero_front_mask]
    Sigma_DUF = np.sqrt(FrontDU_Unc[nonzero_front_mask]) / FrontPixels[nonzero_front_mask]
    Sigma_DDF = np.sqrt(FrontDD_Unc[nonzero_front_mask]) / FrontPixels[nonzero_front_mask]
    Sigma_UDF = np.sqrt(FrontUD_Unc[nonzero_front_mask]) / FrontPixels[nonzero_front_mask]
    Sigma_UUM = np.sqrt(MiddleUU_Unc[nonzero_middle_mask]) / MiddlePixels[nonzero_middle_mask]
    Sigma_DUM = np.sqrt(MiddleDU_Unc[nonzero_middle_mask]) / MiddlePixels[nonzero_middle_mask]
    Sigma_DDM = np.sqrt(MiddleDD_Unc[nonzero_middle_mask]) / MiddlePixels[nonzero_middle_mask]
    Sigma_UDM = np.sqrt(MiddleUD_Unc[nonzero_middle_mask]) / MiddlePixels[nonzero_middle_mask]

    UUM = UUM - LowQ_Offset_NSF
    DUM = DUM - LowQ_Offset_SF
    DDM = DDM - LowQ_Offset_NSF
    UDM = UDM - LowQ_Offset_SF

    if UncertaintyMode == 0:#Varience between pixels udsed for sigma
        FrontUU_DiffSqrd = np.zeros_like(Q_Values)
        FrontDU_DiffSqrd = np.zeros_like(Q_Values)
        FrontDD_DiffSqrd = np.zeros_like(Q_Values)
        FrontUD_DiffSqrd = np.zeros_like(Q_Values)
        MiddleUU_DiffSqrd = np.zeros_like(Q_Values)
        MiddleDU_DiffSqrd = np.zeros_like(Q_Values)
        MiddleDD_DiffSqrd = np.zeros_like(Q_Values)
        MiddleUD_DiffSqrd = np.zeros_like(Q_Values)
        '''Calculate average intensity per Q_bin'''
        FrontPixels_Modified = FrontPixels
        FrontPixels_Modified[FrontPixels <= 0] = 1
        MiddlePixels_Modified = MiddlePixels
        MiddlePixels_Modified[MiddlePixels <= 0] = 1
        MUUAve = MiddleUU /MiddlePixels_Modified
        MDUAve = MiddleDU /MiddlePixels_Modified
        MDDAve = MiddleDD /MiddlePixels_Modified
        MUDAve = MiddleUD /MiddlePixels_Modified
        FUUAve = FrontUU/FrontPixels_Modified
        FDUAve = FrontDU/FrontPixels_Modified
        FDDAve = FrontDD/FrontPixels_Modified
        FUDAve = FrontUD/FrontPixels_Modified
        for dshort in short_detectors:
            dimX = dimXX[dshort]
            dimY = dimYY[dshort]
            Q_tot = QGridPerDetector[dshort][:][:]
            UU = PolCorr_AllDetectors[dshort][0][:][:]
            UU = UU.reshape((dimX, dimY))
            DU = PolCorr_AllDetectors[dshort][1][:][:]
            DU = DU.reshape((dimX, dimY))
            DD = PolCorr_AllDetectors[dshort][2][:][:]
            DD = DD.reshape((dimX, dimY))
            UD = PolCorr_AllDetectors[dshort][3][:][:]
            UD = UD.reshape((dimX, dimY))
            carriage_key = dshort[0]
            Q_tot_modified = Q_tot
            Q_tot_modified[Q_tot > Q_max] = Q_max
            Q_tot_modified[Q_tot < Q_min] = Q_min
            inds = np.digitize(Q_tot_modified, Exp_bins, right=True) - 1
            carriage_key = dshort[0]
            if carriage_key == 'F':
                UU_Ave = FUUAve[inds]
                DU_Ave = FDUAve[inds]
                DD_Ave = FDDAve[inds]
                UD_Ave = FUDAve[inds]
            if carriage_key == 'M':
                UU_Ave = MUUAve[inds]
                DU_Ave = MDUAve[inds] 
                DD_Ave = MDDAve[inds]
                UD_Ave = MUDAve[inds]     
            DiffUUSqrd, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power((UU[masks[dshort] > 0] - UU_Ave[masks[dshort] > 0]),2))
            DiffDUSqrd, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power((UD[masks[dshort] > 0] - UD_Ave[masks[dshort] > 0]),2))
            DiffDDSqrd, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power((DD[masks[dshort] > 0] - DD_Ave[masks[dshort] > 0]),2))
            DiffUDSqrd, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power((UD[masks[dshort] > 0] - UD_Ave[masks[dshort] > 0]),2))
            if carriage_key == 'F':
                FrontUU_DiffSqrd += DiffUUSqrd
                FrontDU_DiffSqrd += DiffDUSqrd
                FrontDD_DiffSqrd += DiffDDSqrd
                FrontUD_DiffSqrd += DiffUDSqrd
            elif carriage_key == 'M':
                MiddleUU_DiffSqrd += DiffUUSqrd
                MiddleDU_DiffSqrd += DiffDUSqrd
                MiddleDD_DiffSqrd += DiffDDSqrd
                MiddleUD_DiffSqrd += DiffUDSqrd
        Sigma_UUF = np.sqrt(FrontUU_DiffSqrd[nonzero_front_mask] / FrontPixels[nonzero_front_mask])
        Sigma_DUF = np.sqrt(FrontDU_DiffSqrd[nonzero_front_mask] / FrontPixels[nonzero_front_mask])
        Sigma_DDF = np.sqrt(FrontDD_DiffSqrd[nonzero_front_mask] / FrontPixels[nonzero_front_mask])
        Sigma_UDF = np.sqrt(FrontUD_DiffSqrd[nonzero_front_mask] / FrontPixels[nonzero_front_mask])
        Sigma_UUM = np.sqrt(MiddleUU_DiffSqrd[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask])
        Sigma_DUM = np.sqrt(MiddleDU_DiffSqrd[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask])
        Sigma_DDM = np.sqrt(MiddleDD_DiffSqrd[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask])
        Sigma_UDM = np.sqrt(MiddleUD_DiffSqrd[nonzero_middle_mask] / MiddlePixels[nonzero_middle_mask])

    Q_Common = np.concatenate((Q_Middle, Q_Front), axis=0)
    Q_Mean = np.concatenate((MeanQ_Middle, MeanQ_Front), axis=0)
    Q_Uncertainty = np.concatenate((MeanQUnc_Middle, MeanQUnc_Front), axis=0)
    UU = np.concatenate((UUM, UUF), axis=0)
    DU = np.concatenate((DUM, DUF), axis=0)
    DD = np.concatenate((DDM, DDF), axis=0)
    UD = np.concatenate((UDM, UDF), axis=0)
    SigmaUU = np.concatenate((Sigma_UUM, Sigma_UUF), axis=0)
    SigmaDU = np.concatenate((Sigma_DUM, Sigma_DUF), axis=0)
    SigmaDD = np.concatenate((Sigma_DDM, Sigma_DDF), axis=0)
    SigmaUD = np.concatenate((Sigma_UDM, Sigma_UDF), axis=0)
    Shadow = np.ones_like(Q_Common)

    if PlotYesNo == 1:
        fig = plt.figure()
        '''If don't want to plot error bars, use something like plt.loglog(Q_Front, UUF, 'b*', label='Front, UU')'''
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.errorbar(Q_Front, UUF, yerr=Sigma_UUF, fmt = 'b*', label='Front, UU')
        ax.errorbar(Q_Middle, UUM, yerr=Sigma_UUM, fmt = 'g*', label='Middle, UU')
        ax.errorbar(Q_Front, DDF, yerr=Sigma_DDF, fmt = 'm*', label='Front, DD')
        ax.errorbar(Q_Middle, DDM, yerr=Sigma_DDM, fmt = 'r*', label='Middle, DD')
        ax.errorbar(Q_Front, DUF, yerr=Sigma_DUF, fmt = 'c.', label='Front, DU')
        ax.errorbar(Q_Middle, DUM, yerr=Sigma_DUM, fmt = 'm.', label='Middle, DU')
        ax.errorbar(Q_Front, UDF, yerr=Sigma_UDF, fmt = 'y.', label='Front, UD')
        ax.errorbar(Q_Middle, UDM, yerr=Sigma_UDM, fmt = 'b.', label='Middle, UD') 
        plt.xlabel('Q')
        plt.ylabel('Intensity')
        plt.title('FullPol_{keyword}Cuts for ID = {idnum} and Config = {cf}'.format(keyword=Key, idnum=ID, cf = Config))
        plt.legend()
        fig.savefig('{keyword}FullPol_Cuts_ID{idnum}_CF{cf}.png'.format(keyword=Key, idnum=ID, cf = Config))
        plt.show()

        SFF = DUF + DUF
        Sigma_SFF = np.sqrt(np.power(Sigma_DUF,2) + np.power(Sigma_UDF,2))
        SFM = DUM + DUM
        Sigma_SFM = np.sqrt(np.power(Sigma_DUM,2) + np.power(Sigma_UDM,2))

        NSFF = UUF + DDF
        Sigma_NSFF = np.sqrt(np.power(Sigma_UUF,2) + np.power(Sigma_DDF,2))
        NSFM = UUM + DDM
        Sigma_NSFM = np.sqrt(np.power(Sigma_UUM,2) + np.power(Sigma_DDM,2))

        
        fig = plt.figure()
        #If don't want to plot error bars, use something like plt.loglog(Q_Front, UUF, 'b*', label='Front, UU')
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.errorbar(Q_Middle, NSFM, yerr=Sigma_NSFM, fmt = 'm*', label='Middle, UU + DD')
        ax.errorbar(Q_Front, NSFF, yerr=Sigma_NSFF, fmt = 'r*', label='Front, UU + DD')
        ax.errorbar(Q_Middle, SFM, yerr=Sigma_SFM, fmt = 'b*', label='Middle, UD + DU')
        ax.errorbar(Q_Front, SFF, yerr=Sigma_SFF, fmt = 'g*', label='Front, UD + DU')
        plt.xlabel('Q')
        plt.ylabel('Intensity')
        plt.title('FullPol_{keyword}Cuts for ID = {idnum} and Config = {cf}'.format(keyword=Key, idnum=ID, cf = Config))
        plt.legend()
        fig.savefig('{keyword}FullPol_Combined_ID{idnum}_CF{cf}.png'.format(keyword=Key, idnum=ID, cf = Config))
        plt.show()
        

        SF = UD + DU
        SF_Unc = np.sqrt(np.power(SigmaDU,2) + np.power(SigmaUD,2))
        NSF = UU + DD
        NSF_Unc = np.sqrt(np.power(SigmaUU,2) + np.power(SigmaDD,2))
        NSFDiff = DD - UU

        text_output = np.array([Q_Common, UU, SigmaUU, DU, SigmaDU, DD, SigmaDD, UD, SigmaUD, Q_Uncertainty, Q_Mean, Shadow])
        text_output = text_output.T
        np.savetxt('{key}FullPol_ID={idnum}Config={cf}.txt'.format(key=Key, idnum=ID, cf = Config), text_output, header='Q, UU, DelUU, DU, DelUD, DD, DelDD, UD, DelUD, DQ, MeanQ, Shadow', fmt='%1.4e')

        '''
        text_output2 = np.array([Q_Common, SF, SF_Unc, NSF, NSF_Unc, NSFDiff, Q_Uncertainty, Q_Mean, Shadow])
        text_output2 = text_output2.T
        np.savetxt('{key}FullPol_ID={idnum}Config={cf}.txt'.format(key=Key, idnum=ID, cf = Config), text_output2, header='Q, SF, DelSF, NSF, DelNSF, NSFDiff, DelQ, MeanQ, Shadow', fmt='%1.4e')
        '''
        
        Output = {}
        Output['Q_Common'] = Q_Common
        Output['Q_Mean'] = Q_Mean
        Output['SF'] = SF
        Output['SF_Unc'] = SF_Unc
        Output['NSF'] = NSF
        Output['NSFDiff'] = NSFDiff
        Output['NSF_Unc'] = NSF_Unc
        Output['Q_Uncertainty'] = Q_Uncertainty
        Output['Q_Mean'] = Q_Mean
        Output['Shadow'] = Shadow
     
    return Output

def He3Decay_func(t, p, gamma):
    return p * np.exp(-t / gamma)

def HE3_Pol_AtGivenTime(entry_time, HE3_Cell_Summary):
    #Predefine HE3_Cell_Summary[HE3_Trans[entry]['Insert_time']] = {'Atomic_P0' : P0, 'Gamma(hours)' : gamma, 'Mu' : Mu, 'Te' : Te}
    #He3Decay_func must be predefined

    counter = 0
    for time in HE3_Cell_Summary:
        if counter == 0:
            holder_time = time
            counter += 1
        if entry_time >= time:
            holder_time = time
        if entry_time < time:
            break
        
    delta_time = entry_time - holder_time     
    P0 = HE3_Cell_Summary[holder_time]['Atomic_P0']
    gamma = HE3_Cell_Summary[holder_time]['Gamma(hours)']
    Mu = HE3_Cell_Summary[holder_time]['Mu']
    Te = HE3_Cell_Summary[holder_time]['Te']
    AtomicPol = P0 * np.exp(-delta_time / gamma)
    NeutronPol = np.tanh(Mu * AtomicPol)
    UnpolHE3Trans = Te * np.exp(-Mu)*np.cosh(Mu * AtomicPol)
    T_MAJ = Te * np.exp(-Mu*(1.0 - AtomicPol))
    T_MIN = Te * np.exp(-Mu*(1.0 + AtomicPol))
        
    return NeutronPol, UnpolHE3Trans, T_MAJ, T_MIN

def HE3_DecayCurves(HE3_Trans):
    #Uses predefined He3Decay_func
    #Creates and returns HE3_Cell_Summary

    HE3_Cell_Summary = {}
    
    for entry in HE3_Trans:
        Mu = HE3_Trans[entry]['Mu']
        Te = HE3_Trans[entry]['Te']
        xdata = np.array(HE3_Trans[entry]['Elasped_time'])
        trans_data = np.array(HE3_Trans[entry]['Transmission'])
        ydata = np.arccosh(np.array(trans_data)/(np.e**(-Mu)*Te))/Mu

        if xdata.size < 2:
            P0 = ydata[0]
            gamma = 1000.0 #assumes no appreciable time decay until more data obtained
            PCell0 = np.tanh(Mu * P0)
        else:
            popt, pcov = curve_fit(He3Decay_func, xdata, ydata)
            P0, gamma = popt
            #fit = He3Decay_func(xdata, popt[0], popt[1])
            PCell0 = np.tanh(Mu * P0)

        HE3_Cell_Summary[HE3_Trans[entry]['Insert_time']] = {'Atomic_P0' : P0, 'Gamma(hours)' : gamma, 'Mu' : Mu, 'Te' : Te}
        print('He3Cell Summary for Cell Identity', entry, ':')
        print('PolCell0', PCell0, 'AtomicPol0: ', P0, ' Gamma: ', gamma)
        print('     ')

        if xdata.size >= 2:
            print('Graphing He3 decay curve....(close generated plot to continue)')
            fit = He3Decay_func(xdata, popt[0], popt[1])
            plt.plot(xdata, ydata, 'b*', label='data')
            plt.plot(xdata, fit, 'r-', label='fit of data')
            plt.xlabel('time (hours)')
            plt.ylabel('3He atomic polarization')
            plt.title('He3 Cell Decay')
            plt.legend()
            plt.show()

        if xdata.size >= 2:
            print('Graphing current and projected decay curves....(close generated plot to continue)')
            TMAJ_data = Te * np.exp(-Mu*(1.0 - ydata))
            TMIN_data = Te * np.exp(-Mu*(1.0 + ydata))
            xdatalonger = HE3_Trans[entry]['Elasped_time']
            L = len(xdata)
            last_time = xdata[L-1]
            for i in range(49):
                extra_time = last_time + i*1
                xdatalonger.append(extra_time)
            xdataextended = np.array(xdatalonger)
            AtomicPol_fitlonger = He3Decay_func(xdataextended, popt[0], popt[1])
            TMAJ_fit = Te * np.exp(-Mu*(1.0 - AtomicPol_fitlonger))
            TMIN_fit = Te * np.exp(-Mu*(1.0 + AtomicPol_fitlonger))
            
            plt.plot(xdata, TMAJ_data, 'b*', label='T_MAJ data')
            plt.plot(xdataextended, TMAJ_fit, 'c-', label='T_MAJ predicted')

            plt.plot(xdata, TMIN_data, 'r*', label='T_MIN data')
            plt.plot(xdataextended, TMIN_fit, 'm-', label='T_MIN predicted')
            
            plt.xlabel('time (hours)')
            plt.ylabel('Spin Transmission')
            plt.title('Predicted He3 Cell Transmission')
            plt.legend()
            plt.show()
            
        #NeutronPol = np.tanh(Mu * AtomicPol)
        #UnpolHE3Trans = Te * np.exp(-Mu)*np.cosh(Mu * AtomicPol)
        #T_MAJ = Te * np.exp(-Mu*(1.0 - AtomicPol))
        #T_MIN = Te * np.exp(-Mu*(1.0 + AtomicPol))

    return HE3_Cell_Summary

def Pol_SuppermirrorAndFlipper(Pol_Trans):
    #Uses time of measurement from Pol_Trans,
    #saves PSM and PF values into Pol_Trans.
    #Uses prefefined HE3_Pol_AtGivenTime function.
    
    for ID in Pol_Trans:
        for Time in Pol_Trans[ID]['T_UU']['Meas_Time']:
            NP, UT, T_MAJ, T_MIN = HE3_Pol_AtGivenTime(Time, HE3_Cell_Summary)
            if 'Neutron_Pol' not in Pol_Trans[ID]['T_UU']:
                Pol_Trans[ID]['T_UU']['Neutron_Pol'] = [NP]
                Pol_Trans[ID]['T_UU']['Unpol_Trans'] = [UT]
            else:
                Pol_Trans[ID]['T_UU']['Neutron_Pol'].append(NP)
                Pol_Trans[ID]['T_UU']['Unpol_Trans'].append(UT)
        for Time in Pol_Trans[ID]['T_DD']['Meas_Time']:
            NP, UT, T_MAJ, T_MIN = HE3_Pol_AtGivenTime(Time, HE3_Cell_Summary)
            if 'Neutron_Pol' not in Pol_Trans[ID]['T_DD']:
                Pol_Trans[ID]['T_DD']['Neutron_Pol'] = [NP]
                Pol_Trans[ID]['T_DD']['Unpol_Trans'] = [UT]
            else:
                Pol_Trans[ID]['T_DD']['Neutron_Pol'].append(NP)
                Pol_Trans[ID]['T_DD']['Unpol_Trans'].append(UT)       
        for Time in Pol_Trans[ID]['T_DU']['Meas_Time']:
            NP, UT, T_MAJ, T_MIN = HE3_Pol_AtGivenTime(Time, HE3_Cell_Summary)
            if 'Neutron_Pol' not in Pol_Trans[ID]['T_DU']:
                Pol_Trans[ID]['T_DU']['Neutron_Pol'] = [NP]
                Pol_Trans[ID]['T_DU']['Unpol_Trans'] = [UT]
            else:
                Pol_Trans[ID]['T_DU']['Neutron_Pol'].append(NP)
                Pol_Trans[ID]['T_DU']['Unpol_Trans'].append(UT)     
        for Time in Pol_Trans[ID]['T_UD']['Meas_Time']:
            NP, UT,T_MAJ, T_MIN = HE3_Pol_AtGivenTime(Time, HE3_Cell_Summary)
            if 'Neutron_Pol' not in Pol_Trans[ID]['T_UD']:
                Pol_Trans[ID]['T_UD']['Neutron_Pol'] = [NP]
                Pol_Trans[ID]['T_UD']['Unpol_Trans'] = [UT]
            else:
                Pol_Trans[ID]['T_UD']['Neutron_Pol'].append(NP)
                Pol_Trans[ID]['T_UD']['Unpol_Trans'].append(UT)
            

    for ID in Pol_Trans:
        ABS = np.array(Pol_Trans[ID]['T_SM']['Abs_Trans'])
        Pol_Trans[ID]['AbsScale'] = np.average(ABS)

        UU = np.array(Pol_Trans[ID]['T_UU']['Trans'])
        UU_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_UU']['Unpol_Trans'])
        UU_NeutronPol = np.array(Pol_Trans[ID]['T_UU']['Neutron_Pol'])
        DD = np.array(Pol_Trans[ID]['T_DD']['Trans'])
        DD_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_DD']['Unpol_Trans'])
        DD_NeutronPol = np.array(Pol_Trans[ID]['T_DD']['Neutron_Pol'])

        RevSF = 1

        if RevSF == 0:
            UD = np.array(Pol_Trans[ID]['T_UD']['Trans'])
            UD_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_UD']['Unpol_Trans'])
            UD_NeutronPol = np.array(Pol_Trans[ID]['T_UD']['Neutron_Pol'])
            DU = np.array(Pol_Trans[ID]['T_DU']['Trans'])
            DU_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_DU']['Unpol_Trans'])
            DU_NeutronPol = np.array(Pol_Trans[ID]['T_DU']['Neutron_Pol'])
        else:
            DU = np.array(Pol_Trans[ID]['T_UD']['Trans'])
            DU_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_UD']['Unpol_Trans'])
            DU_NeutronPol = np.array(Pol_Trans[ID]['T_UD']['Neutron_Pol'])
            UD = np.array(Pol_Trans[ID]['T_DU']['Trans'])
            UD_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_DU']['Unpol_Trans'])
            UD_NeutronPol = np.array(Pol_Trans[ID]['T_DU']['Neutron_Pol'])
        '''
        #Method 0 (the one that has worked so far):
        PF = 1.00
        Pol_Trans[ID]['P_F'] = np.average(PF)
        PSM1 = (DD/DD_UnpolHe3Trans - DU/DU_UnpolHe3Trans)/((DD_NeutronPol+DU_NeutronPol)*PF)
        PSM2 = (UU/UU_UnpolHe3Trans - UD/UD_UnpolHe3Trans)/(UU_NeutronPol+UD_NeutronPol)
        PSM = (PSM1 + PSM2)/ 2.0
        Pol_Trans[ID]['P_SM'] = np.average(PSM)
        print('Ave P_SM is', Pol_Trans[ID]['P_SM'])

        #Method 1: 
        PSMxPCAve = (UU/UU_UnpolHe3Trans - UD/UD_UnpolHe3Trans)/(UU/UU_UnpolHe3Trans + UD/UD_UnpolHe3Trans)
        PCAve = (UU_NeutronPol + UD_NeutronPol)/2.0
        PSM = PSMxPCAve/PCAve
        PFPSMxPCAve = (DD/DD_UnpolHe3Trans - DU/DU_UnpolHe3Trans)/(DD/DD_UnpolHe3Trans + DU/DU_UnpolHe3Trans)
        PCAve = (DD_NeutronPol + DU_NeutronPol)/2.0
        PFPSM = PFPSMxPCAve/PCAve
        PF = PFPSM / PSM
        Pol_Trans[ID]['P_SM'] = np.average(PSM)
        Pol_Trans[ID]['P_F'] = np.average(PF)
        print('P_SM', ID, PSM, 'Ave:', Pol_Trans[ID]['P_SM'])
        print('P_F', ID, PF, 'Ave:', Pol_Trans[ID]['P_F'])
        
        #Method 2:
        PF = 1.00
        Pol_Trans[ID]['P_F'] = np.average(PF)
        print('Ave P_F is set to', Pol_Trans[ID]['P_F'])  
        PSMxPCAve = (UU/UU_UnpolHe3Trans - UD/UD_UnpolHe3Trans)/(UU/UU_UnpolHe3Trans + UD/UD_UnpolHe3Trans)
        PCAve = (UU_NeutronPol + UD_NeutronPol)/2.0
        PSM = PSMxPCAve/PCAve
        PSM2xPCAve = (DD/DD_UnpolHe3Trans - DU/DU_UnpolHe3Trans)/(DD/DD_UnpolHe3Trans + DU/DU_UnpolHe3Trans)
        PCAve = (DD_NeutronPol + DU_NeutronPol)/2.0
        PSM2 = PSM2xPCAve/PCAve
        print('P_SM', ID, PSM, PSM2)
        PSM_All = (PSM + PSM2) / 2.0
        Pol_Trans[ID]['P_SM'] = np.average(PSM_All)
        print('Ave P_SM is', Pol_Trans[ID]['P_SM'])

        '''

        #Method 3:
        PF = 1.00
        Pol_Trans[ID]['P_F'] = np.average(PF)
        print('Ave P_F is set to', Pol_Trans[ID]['P_F'])  
        PSMxPC = (UU/UU_UnpolHe3Trans - UD/UD_UnpolHe3Trans)
        PC = (UU_NeutronPol + UD_NeutronPol)/1.0
        PSM = PSMxPC/PC
        PSM2xPC = (DD/DD_UnpolHe3Trans - DU/DU_UnpolHe3Trans)
        PC = (DD_NeutronPol + DU_NeutronPol)/1.0
        PSM2 = PSM2xPC/PC
        print('P_SM', ID, PSM, PSM2)
        PSM_All = (PSM + PSM2) / 2.0
        Pol_Trans[ID]['P_SM'] = np.average(PSM_All)
        print('Ave P_SM is', Pol_Trans[ID]['P_SM'])

        ADD1 = (UU/UU_UnpolHe3Trans + UD/UD_UnpolHe3Trans)/2.0
        ADD2 = (DD/DD_UnpolHe3Trans + DU/DU_UnpolHe3Trans)/2.0
        print('UU + UD', ADD1, 'DD + DU', ADD2)

        if UsePolCorr == 0:#0 Means no, turn it off
            Pol_Trans[ID]['P_SM'] = 1.0
            Pol_Trans[ID]['P_F'] = 1.0
            print('Manually reset P_SM and P_F to unity')

    return


def AbsScale(GroupID, configuration, Trans, Scatt):

    Data_AllDetectors = {}
    Unc_Data_AllDetectors = {}
    BB = {}
    if GroupID in Trans:
        ABS_Scale = np.average(Trans[GroupID][configuration]['Abs_Trans'])
    else:
        ABS_Scale = 1.0  
    SD = np.zeros((8,6144))
    Unc_SD = np.zeros((8,6144))
    for filenumber in Scatt[GroupID][configuration]['File']:
        filename = path + "sans" + str(filenumber) + ".nxs.ngv"
        config = Path(filename)
        if config.is_file():
            f = h5py.File(filename)
            MonCounts = f['entry/control/monitor_counts'][0]
            Count_time = f['entry/collection_time'][0]
            for dshort in short_detectors:
                if configuration in BlockBeam_ScattPerPixel:
                    BB[dshort] = BlockBeam_ScattPerPixel[configuration][dshort]['AvePerSec']*Count_time
                else:
                    BB[dshort] = 0.0
                    
            Det_Index = 0
            for dshort in short_detectors:
                data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                unc = np.sqrt(data)
                data = data - BB[dshort]
                SD[Det_Index][:] += ((1E8/MonCounts)/ABS_Scale)*data.flatten()
                Unc_SD[Det_Index][:] += ((1E8/MonCounts)/ABS_Scale)*unc.flatten()
                #unc = np.sqrt(data)
                #UncScaled_Data[Det_Index][:] += ((1E8/MonCounts)/ABS_Scale)*unc.flatten()
                Det_Index += 1

            Det_Index = 0
            for dshort in short_detectors:
                Data_AllDetectors[dshort] = SD[Det_Index]
                Unc_Data_AllDetectors[dshort] = Unc_SD[Det_Index]
                Det_Index += 1
                
    return Data_AllDetectors, Unc_Data_AllDetectors

def AbsScaleAndPolarizationCorrectData(GroupID, configuration, Pol_Trans, Pol_Scatt):

    BB = {}
    #Populate matricies
    PSM =  Pol_Trans[GroupID]['P_SM']
    PF = Pol_Trans[GroupID]['P_F']
    if GroupID in Pol_Trans:
        ABS_Scale = np.average(Pol_Trans[GroupID]['AbsScale'])
    else:
        ABS_Scale = 1.0
    Pol_Efficiency = np.zeros((4,4))
    #Q = np.zeros((8,6144))
    Scaled_Data = np.zeros((8,4,6144))
    UncScaled_Data = np.zeros((8,4,6144))
    Scatt_Type = ["S_UU_files", "S_DU_files", "S_DD_files", "S_UD_files"]
    for type in Scatt_Type:
        for filenumber in Pol_Scatt[GroupID][configuration][type]:
            filename = path + "sans" + str(filenumber) + ".nxs.ngv"
            config = Path(filename)
            if config.is_file():
                f = h5py.File(filename)
                MonCounts = f['entry/control/monitor_counts'][0]
                Count_time = f['entry/collection_time'][0]

                for dshort in short_detectors:
                    if configuration in BlockBeam_ScattPerPixel:
                        BB[dshort] = BlockBeam_ScattPerPixel[configuration][dshort]['AvePerSec']*Count_time
                    else:
                        BB[dshort] = 0.0
                    
                
                End_time = dateutil.parser.parse(f['entry/end_time'][0])
                entry = (End_time.timestamp() - Count_time/2)/3600.0
                NP, UT, T_MAJ, T_MIN = HE3_Pol_AtGivenTime(entry, HE3_Cell_Summary)
                if type == "S_UU_files":
                    CrossSection_Index = 0
                    #Pol_Efficiency[CrossSection_Index][:] += [0.25*UT*(1.0 + PSM)*(1.0 + NP), 0.25*UT*(1.0 - PSM)*(1.0 + NP), 0.25*UT*(1.0 - PSM)*(1.0 - NP), 0.25*UT*(1.0 + PSM)*(1.0 - NP)]
                    Pol_Efficiency[CrossSection_Index][:] += [0.25*(1.0 + PSM)*(T_MAJ), 0.25*(1.0 - PSM)*(T_MAJ), 0.25*(1.0 - PSM)*(T_MIN), 0.25*(1.0 + PSM)*(T_MIN)]
                    Det_Index = 0
                    for dshort in short_detectors:
                        data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)]) #- BlockBeam_Scatt[configuration][dshort]*Count_time
                        unc = np.sqrt(data)
                        data = data - BB[dshort]
                        Scaled_Data[Det_Index][CrossSection_Index][:] += ((1E8/MonCounts)/ABS_Scale)*data.flatten()
                        UncScaled_Data[Det_Index][CrossSection_Index][:] += ((1E8/MonCounts)/ABS_Scale)*unc.flatten()
                        Det_Index += 1
                elif type == "S_DU_files":
                    #Note these seem to be the S_UD files by counts and behaviour
                    CrossSection_Index = 1
                    #Pol_Efficiency[CrossSection_Index][:] += [0.25*UT*(1.0 - PSM*PF)*(1.0 + NP), 0.25*UT*(1.0 + PSM*PF)*(1.0 + NP), 0.25*UT*(1.0 + PSM*PF)*(1.0 - NP), 0.25*UT*(1.0 - PSM*PF)*(1.0 - NP)]   
                    #Pol_Efficiency[CrossSection_Index][:] += [0.25*((1.0 + PSM)*(1.0 - NP)*UT),0.25*((1.0 - PSM)*(1.0 - NP)*UT),0.25*((1.0 - PSM)*(1.0 + NP)*UT),0.25*((1.0 + PSM)*(1.0 + NP)*UT)]
                    Pol_Efficiency[CrossSection_Index][:] += [0.25*((1.0 + PSM)*(T_MIN)),0.25*((1.0 - PSM)*(T_MIN)),0.25*((1.0 - PSM)*(T_MAJ)),0.25*((1.0 + PSM)*(T_MAJ))]   
                    Det_Index = 0
                    for dshort in short_detectors:
                        data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)]) #- BlockBeam_Scatt[configuration][dshort]*Count_time
                        unc = np.sqrt(data)
                        data = data - BB[dshort]
                        Scaled_Data[Det_Index][CrossSection_Index][:] += ((1E8/MonCounts)/ABS_Scale)*data.flatten()
                        UncScaled_Data[Det_Index][CrossSection_Index][:] += ((1E8/MonCounts)/ABS_Scale)*unc.flatten()
                        Det_Index += 1
                elif type == "S_DD_files":
                    CrossSection_Index = 2
                    #Pol_Efficiency[CrossSection_Index][:] += [0.25*((1.0 - PSM*PF)*(1.0 - NP)*UT),0.25*((1.0 + PSM*PF)*(1.0 - NP)*UT),0.25*((1.0 + PSM*PF)*(1.0 + NP)*UT), 0.25*((1.0 - PSM*PF)*(1.0 + NP)*UT)]
                    Pol_Efficiency[CrossSection_Index][:] += [0.25*((1.0 - PSM*PF)*(T_MIN)),0.25*((1.0 + PSM*PF)*(T_MIN)),0.25*((1.0 + PSM*PF)*(T_MAJ)), 0.25*((1.0 - PSM*PF)*(T_MAJ))]
                    Det_Index = 0
                    for dshort in short_detectors:
                        data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)]) #- BlockBeam_Scatt[configuration][dshort]*Count_time
                        unc = np.sqrt(data)
                        data = data - BB[dshort]
                        Scaled_Data[Det_Index][CrossSection_Index][:] += ((1E8/MonCounts)/ABS_Scale)*data.flatten()
                        UncScaled_Data[Det_Index][CrossSection_Index][:] += ((1E8/MonCounts)/ABS_Scale)*unc.flatten()
                        Det_Index += 1
                elif type == "S_UD_files":
                    #Note these seem to be the S_DU files by counts and behaviour
                    CrossSection_Index = 3
                    #Pol_Efficiency[CrossSection_Index][:] += [0.25*((1.0 + PSM)*(1.0 - NP)*UT),0.25*((1.0 - PSM)*(1.0 - NP)*UT),0.25*((1.0 - PSM)*(1.0 + NP)*UT),0.25*((1.0 + PSM)*(1.0 + NP)*UT)]
                    #Pol_Efficiency[CrossSection_Index][:] += [0.25*UT*(1.0 - PSM*PF)*(1.0 + NP), 0.25*UT*(1.0 + PSM*PF)*(1.0 + NP), 0.25*UT*(1.0 + PSM*PF)*(1.0 - NP), 0.25*UT*(1.0 - PSM*PF)*(1.0 - NP)]
                    Pol_Efficiency[CrossSection_Index][:] += [0.25*(1.0 - PSM*PF)*(T_MAJ), 0.25*(1.0 + PSM*PF)*(T_MAJ), 0.25*(1.0 + PSM*PF)*(T_MIN), 0.25*(1.0 - PSM*PF)*(T_MIN)]   
                    Det_Index = 0
                    for dshort in short_detectors:
                        data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)]) #- BlockBeam_Scatt[configuration][dshort]*Count_time
                        unc = np.sqrt(data)
                        data = data - BB[dshort]
                        Scaled_Data[Det_Index][CrossSection_Index][:] += ((1E8/MonCounts)/ABS_Scale)*data.flatten()
                        UncScaled_Data[Det_Index][CrossSection_Index][:] += ((1E8/MonCounts)/ABS_Scale)*unc.flatten()
                        Det_Index += 1
    PolCorr_AllDetectors = {}
    Uncertainty_PolCorr_AllDetectors = {}
    Prefactor = inv(Pol_Efficiency)
    Det_Index = 0
    for dshort in short_detectors:
        UncData_Per_Detector = UncScaled_Data[Det_Index][:][:]
        Data_Per_Detector = Scaled_Data[Det_Index][:][:]
        PolCorr_Data = np.dot(Prefactor, Data_Per_Detector)
        #Below is the code that allows true matrix error propagation, but it takes a while...so may want to optimize more before impleneting.
        #Also will need to uncomment from uncertainties import unumpy (top).
        #Data_Per_Detector2 = unumpy.umatrix(Scaled_Data[Det_Index][:][:], UncScaled_Data[Det_Index][:][:])
        #PolCorr_Data2 = np.dot(Prefactor, Data_Per_Detector2)
        #PolCorr_Data = unumpy.nominal_values(PolCorr_Data2)
        #PolCorr_Unc = unumpy.std_devs(PolCorr_Data2)
        PolCorr_AllDetectors[dshort] = PolCorr_Data
        Uncertainty_PolCorr_AllDetectors[dshort] = UncData_Per_Detector
        Det_Index += 1

    return PolCorr_AllDetectors, Uncertainty_PolCorr_AllDetectors

def ASCIIlike_Output(Type, ID, Config, Data_AllDetectors, Unc_Data_AllDetectors, QGridPerDetector):

    for dshort in short_detectors:

        Q_tot = QGridPerDetector['Q_total'][dshort][:][:]
        Q_unc = np.sqrt(np.power(QGridPerDetector['Q_perp_unc'][dshort][:][:],2) + np.power(QGridPerDetector['Q_parl_unc'][dshort][:][:],2))
        QQX = QGridPerDetector['QX'][dshort][:][:]
        QQX = QQX.T
        QXData = QQX.flatten()
        QQY = QGridPerDetector['QY'][dshort][:][:]
        QQY = QQY.T
        QYData = QQY.flatten()
        QQZ = QGridPerDetector['QZ'][dshort][:][:]
        QQZ = QQZ.T
        QZData = QQZ.flatten()
        QPP = QGridPerDetector['Q_perp_unc'][dshort][:][:]
        QPP = QPP.T
        QPerpUnc = QPP.flatten()
        QPR = QGridPerDetector['Q_parl_unc'][dshort][:][:]
        QPR = QPR.T
        QParlUnc = QPR.flatten()
        Shadow = np.ones_like(Q_tot)

        if Type == 'Unpol':
            print('Outputting Unpol data into ASCII-like format for {det}, GroupID = {idnum} '.format(det=dshort, idnum=ID))
            Intensity = Data_AllDetectors[dshort]
            Intensity = Intensity.T
            Int = Intensity.flatten()
            IntensityUnc = Unc_Data_AllDetectors[dshort]
            IntensityUnc = IntensityUnc.T
            DeltaInt = IntensityUnc.flatten()
            ASCII_like = np.array([QXData, QYData, Int, DeltaInt, QZData, QParlUnc, QPerpUnc])
            ASCII_like = ASCII_like.T
            np.savetxt('UnpolScatt_{det}.DAT'.format(det=dshort), ASCII_like, header='Qx, Qy, I, DI, Qz, UncQParl, UncQPerp', fmt='%1.4e')
            #np.savetxt('UnpolScatt_ID={idnum}_{CF}_{det}.DAT'.format(idnum=ID, CF=Config, det=dshort), ASCII_like, header='Qx, Qy, I, DI, Qz, UncQParl, UncQPerp, Shadow', fmt='%1.4e')
            #np.savetxt('Unpol_ID={idnum}_(CF}_{det}.DAT'.format(idnum=ID, CF=Config, det=dshort), ASCII_like, header='Qx, Qy, I, DI, QZ, UncQParl, UncQPerp, Shadow', fmt='%1.4e')
        if Type == 'Fullpol':
            print('Outputting Fullpol data into ASCII-like format for {det}, GroupID = {idnum} '.format(det=dshort, idnum=ID))
            Intensity_FourCrossSections = Data_AllDetectors[dshort]
            Uncertainty_FourCrossSections = Unc_Data_AllDetectors[dshort]
            Cross_Section = 0
            List = ['UU', 'DU', 'DD', 'UD']
            while Cross_Section < 4:
                Intensity = Intensity_FourCrossSections[Cross_Section][:][:]
                Intensity = Intensity.T
                Int = Intensity.flatten()
                Uncertainty = Uncertainty_FourCrossSections[Cross_Section][:][:]
                Uncertainty = Uncertainty.T
                DeltaInt = Uncertainty.flatten()
                ASCII_like = np.array([QXData, QYData, Int, DeltaInt, QZData, QParlUnc, QPerpUnc])
                ASCII_like = ASCII_like.T
                np.savetxt('{TP}Scatt_ID={idnum}_{CF}_{det}.DAT'.format(TP = List[Cross_Section], idnum=ID, CF=Config, det=dshort), ASCII_like, header='Qx, Qy, I, DI, Qz, UncQParl, UncQPerp', fmt='%1.4e')
                Cross_Section += 1

    return

#*************************************************
#***        Start of 'The Program'             ***
#*************************************************

Sample_Names, Configs, BlockBeam, Scatt, Trans, Pol_Trans, HE3_Trans = SortDataAutomatic(YesNoManualHe3Entry, New_HE3_Files, MuValues, TeValues)

Masks = ReadIn_Masks()

Process_Transmissions(BlockBeam, Masks, HE3_Trans, Pol_Trans, Trans)

HE3_Cell_Summary = HE3_DecayCurves(HE3_Trans)

Pol_SuppermirrorAndFlipper(Pol_Trans)

Plex = Plex_File()

'''

Trans_masks = Trans_Mask()

Measured_Masks = {} #Measured_Masks[ConfigID][dshort][1 and 0 mask]
threshold_counter = 0
for filenumber in Mask_Files:
    front_mask_threshold = Mask_Thresholds_Front[threshold_counter]
    middle_mask_threshold = Mask_Thresholds_Middle[threshold_counter]
    MM = Make_Mask_From_File(filenumber, front_mask_threshold, middle_mask_threshold)
    Config_ID = Unique_Config_ID(filenumber)
    Measured_Masks[Config_ID] = MM
    threshold_counter += 1

BlockBeam_Trans, BlockBeam_ScattPerPixel = BlockedBeam_Averaged(BlockedBeamFiles, Measured_Masks, Trans_masks)

Unpol_Trans, Unpol_Scatt, HE3_Trans, Pol_Trans, Pol_Scatt, Scatt_ConfigIDs = SortData(YesNoManualHe3Entry, New_HE3_Files, MuValues, TeValues, start_number, end_number)

print(HE3_Trans)

dim_All = {} #dim_All[ConfigID][X or Y][dshort]
Solid_Angle_All = {} #Solid_Angle_All[ConfigID][dshort]
Geometric_Masks = {} #Masks_Geometric[ConfigID][dshort]
QValues_All = {} #Masks_Geometric[ConfigID][dshort]
for Config_ID in Scatt_ConfigIDs:
    for filenumber in Scatt_ConfigIDs[Config_ID]['Example_File']:
        Solid_Angle_All[Config_ID] = SolidAngle_AllDetectors(filenumber)
        QX, QY, QZ, Q_total, Q_perp_unc, Q_parl_unc, dimXX, dimYY, Right_mask, Top_mask, Left_mask, Bottom_mask, DiagCW_mask, DiagCCW_mask, No_mask, Mask_User_Definedm, Shadow = QCalculationAndMasks_AllDetectors(filenumber, SectorCutAngles)
        dim_All[Config_ID] = {'X' : dimXX, 'Y' : dimYY}
        Shadow_mask = {}
        Circ_mask = {}
        Horz_mask = {}
        Vert_mask = {}
        Diag_mask = {}
        for dshort in short_detectors:
            Shadow_mask[dshort] = Shadow[dshort]
            Circ_mask[dshort] = No_mask[dshort] #*Mask_User_Defined[dshort]
            Horz_mask[dshort] = (Right_mask[dshort] + Left_mask[dshort]) #*Mask_User_Defined[dshort]
            Vert_mask[dshort] = (Top_mask[dshort] + Bottom_mask[dshort]) #*Mask_User_Defined[dshort]
            Diag_mask[dshort] = (DiagCW_mask[dshort] + DiagCCW_mask[dshort]) #*Mask_User_Defined[dshort]
        Geometric_Masks[Config_ID] = {'Horz': Horz_mask,'Vert':Vert_mask,'Diag':Diag_mask,'Circ':Circ_mask,'Shadow':Shadow_mask}
        QValues_All[Config_ID] = {'QX':QX,'QY':QY,'QZ':QZ,'Q_total':Q_total,'Q_perp_unc':Q_perp_unc,'Q_parl_unc':Q_parl_unc}

Masks_All = Geometric_Masks
for Config_ID in Masks_All:
    if Config_ID in Measured_Masks:
        for Type in Masks_All[Config_ID]:
            for dshort in short_detectors:
                Masks_All[Config_ID][Type][dshort] = Measured_Masks[Config_ID][dshort]*Geometric_Masks[Config_ID][Type][dshort]

Trunc_mask = {}

DataType = 'Unpol'
if UnpolYesNo == 1:            
    UnpolToSubtract_AllDetectors = {}            
    for ID in Unpol_Scatt:
        for Config_ID in Unpol_Scatt[ID]:
            UnpolData_AllDetectors, Unc_UnpolData_AllDetectors = AbsScale(ID, Config_ID, Unpol_Trans, Unpol_Scatt)#UnPolData_AllDetectors[dshort][128 x 48 = 6144]
            for dshort in short_detectors:
                Unc_UnpolData_AllDetectors[dshort] = Unc_UnpolData_AllDetectors[dshort]/(Solid_Angle_All[Config_ID][dshort]*Plex[dshort])
                if Config_ID in UnpolToSubtract_AllDetectors:
                    UnpolData_AllDetectors[dshort] = UnpolData_AllDetectors[dshort]/(Solid_Angle_All[Config_ID][dshort]*Plex[dshort]) - UnpolToSubtract_AllDetectors[Config_ID][dshort]
                else:
                    UnpolData_AllDetectors[dshort] = UnpolData_AllDetectors[dshort]/(Solid_Angle_All[Config_ID][dshort]*Plex[dshort])
            PlotYesNo = 1 #1 means yes
            for Key in Unpol_Key_list:
                Trunc_mask['label'] = Key
                for dshort in short_detectors:
                    Trunc_mask[dshort] = Masks_All[Config_ID][Key][dshort]
                QQ, QM, Intensity = SliceDataUnpolData(Q_min, Q_max, Q_bins, QValues_All[Config_ID], Trunc_mask, UnpolData_AllDetectors, Unc_UnpolData_AllDetectors, dim_All[Config_ID]['X'], dim_All[Config_ID]['Y'], ID, Config_ID, PlotYesNo)
        if Print_ASCII == 1:
            ASCIIlike_Output(DataType, ID, Config_ID, UnpolData_AllDetectors, Unc_UnpolData_AllDetectors, QValues_All[Config_ID]) #QX, QY, QZ, Q_perp_unc, Q_parl_unc)

DataType = 'Fullpol'
FullPolResults = {}
if FullPolYeseNo == 1:
    HE3_Cell_Summary = HE3_DecayCurves(HE3_Trans)

    Pol_SuppermirrorAndFlipper(Pol_Trans)

    PolToSubtract_AllDetectors = {}            
    for ID in Pol_Scatt:
        for Config_ID in Pol_Scatt[ID]:
            PolData_AllDetectors, Unc_PolData_AllDetectors = AbsScaleAndPolarizationCorrectData(ID, Config_ID, Pol_Trans, Pol_Scatt)#PolData_AllDetectors[dshort][0-3 cross-section][128 x 48 = 6144]
            for dshort in short_detectors:
                Unc_PolData_AllDetectors[dshort] = Unc_PolData_AllDetectors[dshort]/(Solid_Angle_All[Config_ID][dshort]*Plex[dshort])
                if Config_ID in PolToSubtract_AllDetectors:
                    PolData_AllDetectors[dshort] = PolData_AllDetectors[dshort]/(Solid_Angle_All[Config_ID][dshort]*Plex[dshort]) - PolToSubtract_AllDetectors[Config_ID][dshort]
                else:
                    PolData_AllDetectors[dshort] = PolData_AllDetectors[dshort]/(Solid_Angle_All[Config_ID][dshort]*Plex[dshort])
            PlotYesNo = 1 #1 means yes
            for Key in FullPol_Key_list:
                Trunc_mask['label'] = Key
                for dshort in short_detectors:
                    Trunc_mask[dshort] = Masks_All[Config_ID][Key][dshort]
                FullPolResults[Key] = SliceDataPolData(Q_min, Q_max, Q_bins, QValues_All[Config_ID], Trunc_mask, PolData_AllDetectors, Unc_PolData_AllDetectors, dimXX, dimYY, ID, Config_ID, PlotYesNo)
                #Q_Common, Q_Mean, SF, SF_Unc, NSF, NSFDiff, NSF_Unc, Q_Uncertainty, Q_Mean, Shadow
                
        if Print_ASCII == 1:
            ASCIIlike_Output(DataType, ID, Config_ID, PolData_AllDetectors, Unc_PolData_AllDetectors, QValues_All[Config_ID])
'''



#*************************************************
#***           End of 'The Program'            ***
#*************************************************



