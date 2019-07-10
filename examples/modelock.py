# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:17:21 2019

@author: Decai
"""
#%% module import
import numpy as np
import matplotlib.pyplot as plt
import pynlo
from pynlo.media.fibers import fiber
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pynlo.light.PulseBase import Pulse
#%% pulse set-up
Window  = 500   # simulation window (ps)
Steps   = 20 # simulation steps
Points  = 2**15  # simulation points


FWHM    = 0.08 # pulse duration (ps)
pulseWL = 1030   # pulse central wavelength (nm)

GDD     = 0.0    # Group delay dispersion (ps^2)
TOD     = 0.0    # Third order dispersion (ps^3)
pump_power = 1e1 # Peak power

PulseType = 'Gauss'
if PulseType == 'Noise':
    PulseIn = pynlo.light.DerivedPulses.NoisePulse(center_wavelength_nm=pulseWL,time_window_ps=Window,NPTS=Points)
elif PulseType == 'Gauss':
    T0=FWHM
    PulseIn = pynlo.light.DerivedPulses.GaussianPulse(pump_power,T0,pulseWL,time_window_ps=Window,NPTS=Points,GDD =GDD, TOD = TOD)
elif PulseType == 'Sinc':
    T0=FWHM/3.7909885
    PulseIn = pynlo.light.DerivedPulses.SincPulse(pump_power,T0,pulseWL,time_window_ps=Window,NPTS=Points,GDD =GDD, TOD = TOD)

# =============================================================================
#### 初始脉冲作图 
#fig= plt.figure()
#ax1=fig.add_subplot(2,1,1)
#ax1.plot(PulseIn.T_ps,np.abs(PulseIn.AT)**2)
#ax2=fig.add_subplot(2,1,2)
#ax2.plot(PulseIn.wl_nm,np.abs(PulseIn.AW)**2)
#plt.show()
# =============================================================================
#%% 腔参数设置
beta2=23   #ps^n/km
beta3=0.026
betas=[beta2,beta3]
Gamma=4.7e-3   # W/m
alpha=0.01

L1=2                 # oc到filter
L2=3                 # filter到增益纤
L_ysflo=2            # 增益纤
L3=12                # 单模光纤
L4=2                 # NALM到oc
L_out=1              # 输出尾纤
L_SMF1=3
L_SMF2=2.5
L_ysfhi=0.5          #NALM增益纤

OutRadio=0.7
CRatio=0.45

g_nalm=8
g_main=2.2
Esat_main=3e-9
Esat_nalm=1e-9
omeg = 30  # nm
## 滤波器
band=2 #nm

Fiber1 = fiber.FiberInstance()
Fiber2 = fiber.FiberInstance()
Fiber3 = fiber.FiberInstance()
Fiber4 = fiber.FiberInstance()
FiberYsfLo = fiber.FiberInstance()
FiberOut = fiber.FiberInstance()
FiberSmf1 = fiber.FiberInstance()
FiberSmf2 = fiber.FiberInstance()
FiberYsfHi = fiber.FiberInstance()

Fiber1.generate_fiber(L1, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km', gain=-alpha)
Fiber2.generate_fiber(L2, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km', gain=-alpha)
Fiber3.generate_fiber(L3, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km', gain=-alpha)
Fiber4.generate_fiber(L4, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km',  gain=-alpha)
FiberYsfLo.generate_fiber(L_ysflo, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km',gain=g_main,Esat=Esat_main,Omg=omeg,label='gainfiber')
FiberOut.generate_fiber(L_out, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km', gain=-alpha)
FiberSmf1.generate_fiber(L_SMF1, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km', gain=-alpha)
FiberSmf2.generate_fiber(L_SMF2, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km', gain=-alpha)
FiberYsfHi.generate_fiber(L_ysfhi, center_wl_nm=pulseWL, betas=betas,\
                              gamma_W_m=Gamma , gvd_units='ps^n/km', gain=g_nalm,Esat=Esat_nalm,Omg=omeg,label='gainfiber')

#%% 计算设置

step=100
dz=0.01
error   = 0.001
Raman = True
Steep = True
UseSimpleRaman = True
UseMvmRaman = True
tau_s = 0.00056   # ps shock

evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=error, USE_SIMPLE_RAMAN=UseSimpleRaman,
                 USE_MVM_RAMAN=UseMvmRaman,dz = dz,
                 disable_Raman              = np.logical_not(Raman), 
                 disable_self_steepening    = np.logical_not(Steep),
                tau_s=tau_s,
                suppress_iteration = True)


#%% 循环计算

Et_out=[]
Et_1=[]
Et_2=[]
Et_3=[]
Et_4=[]
Et_5=[]
Et_6=[]
Et_7=[]
Et_8=[]
Et_8.append(PulseIn)


roundtrip =20

for i in range(roundtrip):
    print('正在运行第%s次循环，共%s循环' %(i+1,roundtrip))
    Et_1.append(evol.propagate(Et_8[i], Fiber1, n_steps=Steps)[3])
#    PulseTemp = Pulse()
#    PulseTemp.clone_pulse(Et)
#    PulseTemp.set_AT(Et_1[i])
    Et_2.append(evol.filter(Et_1[i],band,Fiber1))
    Et_3.append(evol.propagate(Et_2[i], Fiber2, n_steps=Steps)[3])
    Et_4.append(evol.propagate(Et_3[i], FiberYsfLo, n_steps=Steps)[3])
    Et_5.append(evol.propagate(Et_4[i], Fiber3, n_steps=Steps)[3])
    # NALM
    #clock
    Eclock0 =  Pulse()
    Eclock0.clone_pulse(Et_5[i])
    Eclock0.set_AT(np.sqrt(CRatio)*Eclock0.AT)
    Eclock1 = evol.propagate(Eclock0, FiberSmf1, n_steps=Steps)[3]
    Eclock2 = evol.propagate(Eclock1, FiberYsfHi, n_steps=Steps)[3]
    Eclock3 = evol.propagate(Eclock2, FiberSmf2, n_steps=Steps)[3]
    #anclock
    Eanclock0 = Pulse()
    Eanclock0.clone_pulse(Et_5[i])
    Eanclock0.set_AT(1j*np.sqrt(1-CRatio)*Eanclock0.AT)
    Eanclock1 = evol.propagate(Eanclock0, FiberSmf2, n_steps=Steps)[3]
    Eanclock2 = evol.propagate(Eanclock1, FiberYsfHi, n_steps=Steps)[3]
    Eanclock3 = evol.propagate(Eanclock2, FiberSmf1, n_steps=Steps)[3]
    
    E6=Pulse()
    E6.clone_pulse(Et_5[i])
    E6.set_AT(np.sqrt(CRatio)*Eclock3.AT+1j*np.sqrt(1-CRatio)*Eanclock3.AT)
    Et_6.append(E6)
    # 非增益纤
    Et_7.append(evol.propagate(Et_6[i], Fiber4, n_steps=Steps)[3])
    # OC
    E8 = Pulse()
    E8.clone_pulse(Et_7[i])
    E8.set_AT(Et_7[i].AT*np.sqrt(1-OutRadio))
    Et_8.append(E8)
    # 输出处理
    Eout = Pulse()
    Eout.clone_pulse(Et_7[i])
    Eout.set_AT(Et_7[i].AT*np.sqrt(OutRadio))
    
    Et_out.append(evol.propagate(Eout, FiberOut, n_steps=Steps)[3])
    #Ef_out.append(FFT(Et_out[i]))

#%% 结果打印
F_reputation=3e8/1.45/(L_ysflo+L1+L2+L3+L4+L_SMF1+L_ysfhi+L_SMF2)/1e6
Energy_out=np.sum(np.abs(Et_out[-1].AT)**2)*Et_out[-1].dT_mks*1e9  #输出脉冲总能量
print('输出重频=%.3fMHz'% F_reputation)
print('输出脉冲能量:%.3fnj'   % Energy_out)

#%% figure  锁模演化
t=Et_out[-1].T_mks
Wl = Et_out[-1].wl_mks
figure=plt.figure()
pulse_number =10
font_size=20
ax=figure.add_subplot(111,projection='3d')
for z in np.arange(1,roundtrip,roundtrip//pulse_number ):
    xs=t*1e12
    ys=np.abs(Et_out[z].AT)**2
    ax.plot(xs,ys,zs=z,zdir='y',color='black')
ax.set_ylim(0,roundtrip+1)
ax.set_zlim(0,np.max(np.abs(Et_out[-1].AT)**2))
plt.title(u'Temporal evolution',fontproperties='SimHei',fontsize=20+2)
ax.set_xlabel(u'Time/ps',fontsize=font_size,labelpad = 18)
ax.set_ylabel(u'Roundtrip',fontsize=font_size,labelpad = 18)
ax.set_zlabel(u'Intensity / arb. units ',fontsize=font_size,labelpad = 18)
ax.view_init(elev=20, azim=-60)
ax.grid(False)
figure=plt.figure()
ax=figure.add_subplot(111,projection='3d')
for z in np.arange(1,roundtrip,roundtrip//pulse_number):
    xs=(Wl*1e9)
    ys=np.abs((Et_out[z].AW))**2/np.max(np.abs(Et_out[z].AW)**2)
    ax.plot(xs,ys,zs=z,zdir='y')
ax.set_ylim(0,roundtrip+1)
ax.set_zlim(0,1)
plt.title(u'Spectral evolution',fontproperties='SimHei',fontsize=font_size)
ax.set_xlabel(u'Wllength/nm',fontsize=font_size-4)
ax.set_ylabel(u'RoundTrip',fontsize=font_size-4)
ax.set_zlabel(u'Amplitude ',fontsize=font_size-4)
ax.grid(True)
plt.show()

#%% 腔内演化
plt.figure()
font_size=18
number=roundtrip-3
Wlrange=[1000,1080]
plt.plot((Wl)*1e9,np.abs(Et_1[number].AW)**2,'r--',label=u"after SMF",linewidth=2)
plt.plot((Wl)*1e9,np.abs(Et_2[number].AW)**2,'g--',label=u'after Filter',linewidth=2)
plt.plot((Wl)*1e9,np.abs(Et_3[number].AW)**2,'b--',label=u'after SMF',linewidth=2)
plt.plot((Wl)*1e9,np.abs(Et_4[number].AW)**2,'c--',label=u'after YDF-Lo ',linewidth=2)
plt.plot((Wl)*1e9,np.abs(Et_5[number].AW)**2,'m--',label=u'after long SMF',linewidth=2)
plt.plot((Wl)*1e9,np.abs(Et_6[number].AW)**2,'y--',label=u'after NALM',linewidth=2)
plt.plot((Wl)*1e9,np.abs(Et_7[number].AW)**2,'y:.',label=u'after SMF',linewidth=2)
plt.plot((Wl)*1e9,np.abs(Et_8[number+1].AW)**2,'k:',label=u'after OC',linewidth=2)
plt.xlim(Wlrange[0],Wlrange[1])
plt.xlabel(u"波长/nm",fontproperties='SimHei',fontsize=font_size)
plt.ylabel(u"强度",fontproperties='SimHei',fontsize=font_size)
plt.title(u'Esat_nalm:%.2e L_smf=%.2f脉冲在环形腔循环一次的演化'%(Esat_nalm,L_SMF2-L_SMF1),\
          fontproperties='SimHei',fontsize=font_size)
plt.grid(True)
plt.legend(loc=1,fontsize=font_size-4)

plt.figure()
plt.plot(t,np.abs(Et_1[number].AT)**2,'r--',label=u"after SMF",linewidth=2)
plt.plot(t,np.abs(Et_2[number].AT)**2,'g--',label=u'after Filter',linewidth=2)
plt.plot(t,np.abs(Et_3[number].AT)**2,'b--',label=u'after SMF',linewidth=2)
plt.plot(t,np.abs(Et_4[number].AT)**2,'c--',label=u'after YDF-Hi',linewidth=2)
plt.plot(t,np.abs(Et_5[number].AT)**2,'m--',label=u'after long SMF',linewidth=2)
plt.plot(t,np.abs(Et_6[number].AT)**2,'y--',label=u'after NALM',linewidth=2)
plt.plot(t,np.abs(Et_7[number].AT)**2,'y:',label=u'after SMF',linewidth=2)
plt.plot(t,np.abs(Et_8[number+1].AT)**2,'k:',label=u'after OC',linewidth=2)
tlength=np.max(t)-np.min(t)
#plt.xlim(np.min(t)+tleft_scale*tlength,np.min(t)+tright_scale*tlength)
plt.xlabel(u"时间/s",fontproperties='SimHei',fontsize=font_size)
plt.ylabel(u"强度",fontproperties='SimHei',fontsize=font_size)
plt.legend(loc=1,fontsize=font_size-4)
plt.grid(True)
plt.show()

#%% 最终输出作图
tleft_scale=0
tright_scale=1
waverange=[1010,1050]
line_width=2
xmajorLocator   = MultipleLocator(20) #将x主刻度标签设置为20的倍数  
xminorLocator   = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数 
figure=plt.figure()
timeout_plot=figure.add_subplot(111)
timeout_plot.plot(t,np.abs(Et_out[-1].AT)**2/np.max(np.abs(Et_out[-1].AT)**2),'b',linewidth=line_width)
tlength=np.max(t)-np.min(t)
plt.xlim(np.min(t)+tleft_scale*tlength,np.min(t)+tright_scale*tlength)
plt.ylim(0,1)
plt.title(u"输出脉冲时域图", fontproperties='SimHei',fontsize=24)
plt.xlabel(u"时间/s", fontproperties='SimHei',fontsize=20)
plt.ylabel(u"强度", fontproperties='SimHei',fontsize=20)
plt.grid(True)
#phase=np.unwrap(np.angle(IFFT((Et_out[-1].AT))))
#chirp=-np.diff(phase)/dT
#chirp=np.append(chirp[0],chirp)
#chirp_plot=timeout_plot.twinx()
#chirp_plot.plot(t,chirp,'r',linewidth=line_width,label='chirp')
#chirp_plot.legend()

figure=plt.figure()
ax=plt.subplot(111)
freqout_plot=plt.plot(Wl*1e9,np.abs(Et_out[-1].AT)**2/np.max(np.abs(Et_out[-1].AT)**2),linewidth=line_width)
plt.xlim(waverange[0],waverange[1])
plt.ylim(0,1.5)
plt.title(u'输出信号频谱图',fontproperties='SimHei',fontsize=24)
plt.xlabel(u"波长/nm", fontproperties='SimHei',fontsize=20)
plt.ylabel(u"强度", fontproperties='SimHei',fontsize=20)
ax.xaxis.set_major_locator(xmajorLocator) 
ax.xaxis.set_minor_locator(xminorLocator)  
ax.xaxis.grid(True, which='major' and 'minor') #x坐标轴的网格使用主刻度  
ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度  

plt.show()