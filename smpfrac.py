from os import path
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from array import array
import time


from mpi_learn.train.GanModel import GANModel
gan_args = {
    'tell': False,
    'reversedorder' : True,
    'heavycheck' : False,
    'show_values' : False,
    'gen_bn' : True,
    'checkpoint' : False,
    'onepass' : True,
    'show_loss' : True,
    'with_fixed_disc' : True ## could switch back to False and check
    }

gm = GANModel(**gan_args)
#gStyle.SetOptStat(0)
gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
c.SetGrid()
gStyle.SetOptStat(0)
#c.SetLogx ()
Eprof = TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 500)
num_events=1000
latent = 200
#gweight = 'gen_rootfit_2p1p1_ep33.hdf5'
gweight1='m1_twopass_easgd_bs5_ep20_train_mpi_learn_result.h5'
gweight2='m1_twopass_easgd_bs10_ep20_train_mpi_learn_result.h5'
gweight3='m1_twopass_easgd_bs15_ep20_train_mpi_learn_result.h5'
gweight4='m1_twopass_easgd_bs20_ep20_train_mpi_learn_result.h5'
gweights = [gweight1, gweight2, gweight3,gweight4]
label = ['1 gpu', '5 gpus', '10 gpus', '15 gpus', '20 gpus']
scales = [1, 1, 1,1]
filename = 'ecal_ratio_multi.pdf'
#Get Actual Data
#d=h5py.File("/eos/project/d/dshep/LCD/V1/EleEscan/EleEscan_1_1.h5")
d=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
X=np.array(d.get('ECAL')[0:num_events], np.float64)                             
Y=np.array(d.get('target')[0:num_events][:,1], np.float64)
X[X < 1e-6] = 0
Y = Y
Data = np.sum(X, axis=(1, 2, 3))

for j in np.arange(num_events):
   Eprof.Fill(Y[j], Data[j]/Y[j])
Eprof.SetTitle("Ratio of Ecal and Ep")
Eprof.GetXaxis().SetTitle("Ep")
Eprof.GetYaxis().SetTitle("Ecal/Ep")
Eprof.Draw()
Eprof.GetYaxis().SetRangeUser(0, 0.03)
color =2
Eprof.SetLineColor(color)
legend = TLegend(0.8, 0.8, 0.9, 0.9)
legend.AddEntry(Eprof, "Data", "l")
Gprof = []
for i, gweight in enumerate(gweights):
#for i in np.arange(1):
#   gweight=gweights[i]                                                                                                                                                                     
   Gprof.append( TProfile("Gprof" +str(i), "Gprof" + str(i), 100, 0, 500))
   #Gprof[i].SetStates(0)
   #Generate events
   gm.combined.load_weights(gweight)
   noise = np.random.normal(0, 1, (num_events, latent))
   generator_in = np.multiply(np.reshape(Y/100, (-1, 1)), noise)
   generated_images = gm.generator.predict(generator_in, verbose=False, batch_size=100)
   GData = np.sum(generated_images, axis=(1, 2, 3))/scales[i]

   print GData.shape
   for j in range(num_events):
      Gprof[i].Fill(Y[j], GData[j]/Y[j])
   color = color + 2
   Gprof[i].SetLineColor(color)
   Gprof[i].Draw('sames')
   c.Modified()
   legend.AddEntry(Gprof[i], label[i], "l")
   legend.Draw()
   c.Update()
c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)
# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')
