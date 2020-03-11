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

path = ''

def NG7SANS_Config_ID(filenumber):
    
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
    
    filename = path + "sans" + str(filenumber) + ".nxs.ng7"
    config = Path(filename)
    if config.is_file():
        f = h5py.File(filename)
        data = np.array(f['entry/instrument/detector/data'])
        monitor_counts = f['entry/control/monitor_counts'][0]
        count_time = f['entry/collection_time'][0]
        abs_trans = np.sum(data)*1E8/monitor_counts
        
    return abs_trans

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
        #SourceAp = 22.2/20.0
        SourceAp_Descrip = str(f['/entry/DAS_logs/geometry/sourceAperture'][0]) #source aperture in mm - > cm; convert to RADIUS?
        SourceAp_Descrip = SourceAp_Descrip[2:]
        SourceAp_Descrip = SourceAp_Descrip[:-3]
        SourceAp =  float(SourceAp_Descrip)
        SourceAp = SourceAp/20.0
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

    return data, data_unc, Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimX, dimY

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
        plt.show()

    return

#*************************************************
#***        Start of 'The Program'             ***
#*************************************************

Config = NG7SANS_Config_ID(Scatt_filenumber)
print(Config)

SolidAngle = NG7SANS_SolidAngle(Scatt_filenumber)
print('Solid Angle Corr', SolidAngle)

Abs_Trans = NG7SANS_TransCountsPer1E8MonCounts(Trans_filenumber)
print('Abs. Trans', Abs_Trans)

wavelength = 6.0
attenuation = 4
Attn_Trans = NG7SANS_AttenuatorTable(wavelength, attenuation)
Abs_Trans = Abs_Trans/Attn_Trans

Data, DataUnc, Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimX, dimY  = NG7SANS_QCalculation(Scatt_filenumber)

Thickness = 0.3
Data = Data / (SolidAngle*Abs_Trans*Thickness)
DataUnc = DataUnc / (SolidAngle*Abs_Trans*Thickness)

PrimaryAngle = 42
AngleWidth = 45
BothSides = 1
SectorMask = NG7SANS_SectorMask(InPlaneAngleMap, PrimaryAngle, AngleWidth, BothSides)

Q_min = 0.001
Q_max = 0.03
Q_bins = 100
NG7SANS_TwoDimToOneDim(Q_min, Q_max, Q_bins, Q_total, Q_total, SectorMask, Data, DataUnc)

#NG7SANS_QxQyASCII(Data, DataUnc, Qx, Qy, Qz, Q_perp_unc, Q_parl_unc)
