import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys

# 测试不同频率范围设定

# sys.path.append('/home/st/Work_Space/module_st/basic-toolbox')
sys.path.append('/home/st/Work_Space/module_st/Gammatone-filters')
# from wav_tools import wav_tools
# from local_TFData import TFData
from gtf import gtf

gt_filter1 = gtf(fs=16e3,CF_low=70,CF_high=7e3,N_band=32)
ir1 = gt_filter1.get_ir()
fig1 = gt_filter1.plot_ir_spec(ir1)
fig1.savefig('images/CF_low70_CF_high7e3.png')


gt_filter1 = gtf(fs=16e3,freq_low=70,CF_high=7e3,N_band=32)
ir1 = gt_filter1.get_ir()
fig1 = gt_filter1.plot_ir_spec(ir1)
fig1.savefig('images/freq_low70_CF_high7e3.png')


gt_filter1 = gtf(fs=16e3,CF_low=70,freq_high=7e3,N_band=32)
ir1 = gt_filter1.get_ir()
fig1 = gt_filter1.plot_ir_spec(ir1)
fig1.savefig('images/cf_low70_freq_high7e3.png')

gt_filter1 = gtf(fs=16e3,freq_low=70,freq_high=7e3,N_band=32)
ir1 = gt_filter1.get_ir()
fig1 = gt_filter1.plot_ir_spec(ir1)
fig1.savefig('images/freq_low70_freq_high7e3.png')
