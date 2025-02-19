import numpy as np # type: ignore
import h5py # type: ignore
import pandas as pd # type: ignore
import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore
import puma # type: ignore
import math # type: ignore
from ftag import Flavours # type: ignore
import os # type: ignore

## Sample selection

preprocess = "4dot6GeV_train"
ttbar_filepath = "NetworkSamples/100_100_40_10/"+preprocess+".h5"

plotting_path = "NetworkPlots/Trackless/"+preprocess+"/"

if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)



## Some Global variables

flav_names = ["ujets","cjets","bjets","taujets"]
prob_names = ["pu","pc","pb","ptau"]

Continue = True
loop_counter = 0

## Loop Controls

jets_per_batch = 500_000
#Max 500 (250_000_000 Jets)
loop_target = 500

bar_length = 100

## Main Binning Info

BINS = 50
BINSRANGE = [20,250]
BINWIDTH = (BINSRANGE[1]-BINSRANGE[0])/BINS

BINS_ETA = 20
BINSRANGE_ETA = [0,2.5]
BINWIDTH_ETA = (BINSRANGE_ETA[1]-BINSRANGE_ETA[0])/BINS_ETA

## Extra Binning calculations

pt_edges = np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1)
eta_edges = np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1)

pt_midpoints = 0.5*(pt_edges[1:] + pt_edges[:-1])
eta_midpoints = 0.5*(eta_edges[1:] + eta_edges[:-1])

disc_bins = 20
disc_binsrange = [-7.5,12.5]

disc_edges = np.linspace(disc_binsrange[0], disc_binsrange[1],disc_bins+1)

## variable/arrays for use outside of operations loop

# number of each jet flavour [ujets,cjets,bjets,taujets]
N_flav = [0,0,0,0]
N_flav_trackless = [0,0,0,0]

# Histograms for pt and eta (light-jets, c-jets, b-jets, tau-jets)
pt_hist_all = [np.histogram([],bins = pt_edges)[0],
               np.histogram([],bins = pt_edges)[0],
               np.histogram([],bins = pt_edges)[0],
               np.histogram([],bins = pt_edges)[0]]
pt_hist_trackless = [np.histogram([],bins = pt_edges)[0],
                     np.histogram([],bins = pt_edges)[0],
                     np.histogram([],bins = pt_edges)[0],
                     np.histogram([],bins = pt_edges)[0]]
eta_hist_all = [np.histogram([],bins = eta_edges)[0],
                np.histogram([],bins = eta_edges)[0],
                np.histogram([],bins = eta_edges)[0],
                np.histogram([],bins = eta_edges)[0]]
eta_hist_trackless = [np.histogram([],bins = eta_edges)[0],
                      np.histogram([],bins = eta_edges)[0],
                      np.histogram([],bins = eta_edges)[0],
                      np.histogram([],bins = eta_edges)[0]]

## Operations loop

print("Starting...",end="\r")

while Continue == True:

    jets_start = loop_counter*jets_per_batch

    with h5py.File(ttbar_filepath,"r") as h5file:

        jets_arr = h5file["jets"][jets_start:jets_start+jets_per_batch-1]
        tracks_arr = h5file["super_tracks_pv"][jets_start:jets_start+jets_per_batch-1]
        if loop_counter+1 == loop_target:
            Continue = False

        is_light = jets_arr["HadronConeExclTruthLabelID"] == 0
        is_c = jets_arr["HadronConeExclTruthLabelID"] == 4
        is_b = jets_arr["HadronConeExclTruthLabelID"] == 5
        is_tau = jets_arr["HadronConeExclTruthLabelID"] == 15

        flav_bools = [is_light,is_c,is_b,is_tau]

        trackless_bool = np.sum(tracks_arr["valid"], axis=1) == 0


## Flavour Loop

    for count, flav in enumerate(flav_bools):
    ## Flavour Counting

        N_flav[count]+=np.sum(flav)
        N_flav_trackless[count]+=np.sum(np.logical_and(flav,trackless_bool))

    ## Hist Additions

        ## pt hists

        pt_hist_all[count] += np.histogram(jets_arr[flav]["pt"]/1e3, bins = pt_edges)[0]
        pt_hist_trackless[count] += np.histogram(jets_arr[np.logical_and(flav,trackless_bool)]["pt"]/1e3, bins = pt_edges)[0]

        ## eta hists

        eta_hist_all[count] += np.histogram(np.absolute(jets_arr[flav]["eta"]), bins = eta_edges)[0]
        eta_hist_trackless[count] += np.histogram(np.absolute(jets_arr[np.logical_and(flav,trackless_bool)]["eta"]), bins = eta_edges)[0]

    ## Progress Bar
    loop_perc = round(100*(loop_counter+1)/loop_target)
    loop_perc_round = math.floor(loop_perc*bar_length/100)

    prog_bar = "["
    prog_bar = prog_bar + loop_perc_round*"|"+(bar_length-loop_perc_round)*"-"+"]"
    print(" "+prog_bar+" : ("+str(loop_counter+1)+"/"+str(loop_target)+") : "+str(loop_perc)+"%",end = "\r")

    loop_counter += 1

## Outputting Done Looping
print(" "*(bar_length+20),end="\r")
print("Total of "+str(loop_target*jets_per_batch/1_000_000)+" million jets collected!")


print("No. Jet Flavours:")
for i in range(4):
    print("No. "+flav_names[i]+":  " +str(N_flav[i]))
          
print("No. Jet Flavours Trackless:")
for i in range(4):
    print("No. "+flav_names[i]+":  " +str(N_flav_trackless[i]))


print("Percent Jet Flavours:")
for i in range(4):
    print("No. "+flav_names[i]+":  " +str(N_flav_trackless[i]/N_flav[i]))

## trackless ratio plots 

histplot_pt_ratio = puma.HistogramPlot(bins = pt_edges,
                                  xlabel = "$p_T$ [GeV]",
                                  ylabel = "No. Jets",
                                  n_ratio_panels = 1,
                                  atlas_second_tag = preprocess,
                                  logy = True,
                                  norm = False,
                                  underoverflow = False)

loop_flavours = [0,2]
for i in loop_flavours:
    histplot_pt_ratio.add(puma.Histogram(pt_hist_all[i],
                                    bin_edges = pt_edges,
                                    flavour = flav_names[i],
                                    ratio_group = flav_names[i]),
                                    reference = True)
    histplot_pt_ratio.add(puma.Histogram(pt_hist_trackless[i],
                                    bin_edges = pt_edges,
                                    colour = Flavours[flav_names[i]].colour,
                                    ratio_group = flav_names[i],
                                    linestyle = "dashed"))

histplot_pt_ratio.draw()
histplot_pt_ratio.make_linestyle_legend(linestyles=["solid","dashed"],
                                    labels=["All", "Trackless"],
                                    bbox_to_anchor=(0.55, 1))
histplot_pt_ratio.savefig(plotting_path+"pt_trackless_ratio.png")


## eta
histplot_eta_ratio = puma.HistogramPlot(bins = eta_edges,
                                  xlabel = "|$\\eta$|",
                                  ylabel = "No. Jets",
                                  n_ratio_panels = 1,
                                  atlas_second_tag = preprocess,
                                  logy = True,
                                  norm = False,
                                  underoverflow = False)

for i in loop_flavours:
    histplot_eta_ratio.add(puma.Histogram(eta_hist_all[i],
                                    bin_edges = eta_edges,
                                    flavour = flav_names[i],
                                    ratio_group = flav_names[i]),
                                    reference = True)
    histplot_eta_ratio.add(puma.Histogram(eta_hist_trackless[i],
                                    bin_edges = eta_edges,
                                    colour = Flavours[flav_names[i]].colour,
                                    ratio_group = flav_names[i],
                                    linestyle = "dashed"))

histplot_eta_ratio.draw()
histplot_eta_ratio.make_linestyle_legend(linestyles=["solid","dashed"],
                                    labels=["All", "Trackless"],
                                    bbox_to_anchor=(0.55, 1))
histplot_eta_ratio.savefig(plotting_path+"/eta_trackless_ratio.png")



## trackless ratio of flavour over light jets

histplot_trackless = puma.HistogramPlot(bins = pt_edges,
                                        xlabel = "$p_T$ [GeV]",
                                        ylabel = "Ratio of u-jets to b-jets (trackless)",
                                        atlas_second_tag = preprocess,
                                        logy = False,
                                        norm = False,
                                        underoverflow = False)

histplot_trackless.add(puma.Histogram(np.log(np.divide(pt_hist_trackless[2],pt_hist_trackless[0])),bin_edges = pt_edges))

histplot_trackless.draw()
histplot_trackless.savefig(plotting_path+"/trackless.png")

## pseudo disc plot

histplot_pseudo = puma.HistogramPlot(
                                     xlabel = "Pseudo Discriminant",
                                     ylabel = "A.U.",
                                     atlas_second_tag = preprocess,
                                     logy = False,
                                     norm = True,
                                     underoverflow = False)

histplot_pseudo.add(puma.Histogram(values = np.log(np.divide(pt_hist_trackless[2],pt_hist_trackless[0])), weights = pt_hist_all[0]))

histplot_pseudo.draw()
histplot_pseudo.savefig(plotting_path+"/pseudo.png")