import numpy as np # type: ignore
import h5py # type: ignore
import pandas as pd # type: ignore
import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore
import math # type: ignore
import puma # type: ignore
import os # type: ignore

filetype = "png"

preprocess = "1GeV"
ttbar_filepath = "NetworkSamples/train/"+preprocess+".h5"

plotting_path = "NetworkPlots/Corrections_"+filetype+"/"
corrections_path = "NetworkCorrections/"+preprocess+"/"

## Check if plotting and correction directory exists and create one if not
if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)

plotting_path = plotting_path + preprocess+"/"

if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)

if not os.path.exists(corrections_path):
    os.mkdir(corrections_path)


flav_names = ["ujets","cjets","bjets","taujets"]
prob_names = ["pu","pc","pb","ptau"]

Continue = True
loop_counter = 0

jets_per_batch = 500_000
#Max 500 (250_000_000 Jets)
loop_target = 500

bar_length = 100


if "4dot6GeV" in preprocess:
    BINS = 10*5
elif "1GeV" in preprocess:
    BINS = 46*5
BINSRANGE = [20,66]
BINWIDTH = (BINSRANGE[1]-BINSRANGE[0])/BINS

BINS_ETA = 20
BINSRANGE_ETA = [0,2.5]
BINWIDTH_ETA = (BINSRANGE_ETA[1]-BINSRANGE_ETA[0])/BINS_ETA

NORM = False
addition = ""
if NORM:
    addition = "Normailised "

## variable/arrays for use outside of operations loop

PT_ETA_2D_bkg = np.histogram2d([],[],
                                    bins = (np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1),
                                            np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1))
                                    )[0]

PT_ETA_2D_sig = np.histogram2d([],[],
                                    bins = (np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1),
                                            np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1))
                                    )[0]


PT_ETA_2D = [np.histogram2d([],[],bins = (np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1),
                                          np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1)))[0],
             np.histogram2d([],[],bins = (np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1),
                                          np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1)))[0],
             np.histogram2d([],[],bins = (np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1),
                                          np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1)))[0],
             np.histogram2d([],[],bins = (np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1),
                                          np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1)))[0]]


pt_edges = np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1)
eta_edges = np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1)

pt_midpoints = 0.5*(pt_edges[1:] + pt_edges[:-1])
eta_midpoints = 0.5*(eta_edges[1:] + eta_edges[:-1])

pt_scale = BINS/(BINSRANGE[1]-BINSRANGE[0])
eta_scale = BINS_ETA/(BINSRANGE_ETA[1]-BINSRANGE_ETA[0])

pt_edges_ticks = np.linspace(int(np.floor(BINSRANGE[0])),int(np.floor(BINSRANGE[1])),int(np.floor(BINSRANGE[1]))-int(np.floor(BINSRANGE[0]))+1)

## Operations loop

while Continue == True:

    jets_start = loop_counter*jets_per_batch

    with h5py.File(ttbar_filepath,"r") as h5file:

        jets_arr = h5file["jets"][jets_start:jets_start+jets_per_batch-1]

        if loop_counter+1 == loop_target:
            Continue = False

        is_light = jets_arr["HadronConeExclTruthLabelID"] == 0
        is_c = jets_arr["HadronConeExclTruthLabelID"] == 4
        is_b = jets_arr["HadronConeExclTruthLabelID"] == 5
        is_tau = jets_arr["HadronConeExclTruthLabelID"] == 15

    flav_bool = [is_light,is_c,is_b,is_tau]

## Total Hist makers

    ## 2D Correction

    for count,flav in enumerate(flav_bool):

        PT_ETA_2D[count] += np.histogram2d(jets_arr[flav]["pt"]/1e3, np.absolute(jets_arr[flav]["eta"]),
                                           bins = (np.linspace(BINSRANGE[0],BINSRANGE[1],BINS+1),
                                                   np.linspace(BINSRANGE_ETA[0],BINSRANGE_ETA[1],BINS_ETA+1)))[0]

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

## Reweight hists to equal light jets

weight_light = np.sum(PT_ETA_2D[0])

Pseudo_Outputs = []

file_adds = ["u","c","b","tau"]

for count, flav in enumerate(flav_bool):
    PT_ETA_2D[count] = PT_ETA_2D[count] * (weight_light/np.sum(PT_ETA_2D[count]))

    n_jets_arr = PT_ETA_2D[count] if count == 0 else np.add(n_jets_arr,PT_ETA_2D[count])

for count, flav in enumerate(flav_bool):

    Temp_Outputs = np.divide(PT_ETA_2D[count],n_jets_arr)

    if np.sum(Temp_Outputs[Temp_Outputs == 0]) != 0:
        print("NOOOOOOOOOOOOOOOOOOOOOOOOOOO")

    np.save(corrections_path+"Corr_Output_"+file_adds[count],Temp_Outputs)

np.save(corrections_path+"corr_bins_pt", pt_edges)
np.save(corrections_path+"corr_bins_eta",eta_edges)

## Plotting

for signal_index in range(2):

    signal_index += 2
    background_index = 0

    sig_name = file_adds[signal_index]

    PT_ETA_RATIO_2D = np.divide(PT_ETA_2D[signal_index],PT_ETA_2D[background_index])

    PT_RATIO_1D = np.divide(PT_ETA_2D[signal_index].sum(axis=1),PT_ETA_2D[background_index].sum(axis=1))
    ETA_RATIO_1D = np.divide(PT_ETA_2D[signal_index].sum(axis=0),PT_ETA_2D[background_index].sum(axis=0))

    ## outputs

    ## pt dist
    histplot_pt = puma.HistogramPlot(bins = pt_edges,
                                    n_ratio_panels = 1,
                                    xlabel = "$p_T$ [GeV]",
                                    ylabel = "Normalised No. Jets",
                                    atlas_second_tag = preprocess+", binned @ "+str(BINWIDTH)+"GeV",
                                    logy = False,
                                    norm = True,
                                    underoverflow = False)

    histplot_pt.add(puma.Histogram(PT_ETA_2D[signal_index].sum(axis=1),
                                    bin_edges = pt_edges,
                                    flavour = flav_names[signal_index]),
                                    reference = True)

    histplot_pt.add(puma.Histogram(PT_ETA_2D[background_index].sum(axis=1),
                                    bin_edges = pt_edges,
                                    flavour = flav_names[background_index]))

    histplot_pt.draw()
    histplot_pt.savefig(plotting_path+"/pt_"+sig_name+"."+filetype)

    ## eta dist

    histplot_eta = puma.HistogramPlot(bins = eta_edges,
                                    n_ratio_panels = 1,
                                    xlabel = "$\eta$",
                                    ylabel = "Normalised No. Jets",
                                    atlas_second_tag = preprocess+", binned @ "+str(BINWIDTH_ETA),
                                    logy = False,
                                    norm = True,
                                    underoverflow = False)

    histplot_eta.add(puma.Histogram(PT_ETA_2D[signal_index].sum(axis=0),
                                    bin_edges = eta_edges,
                                    flavour = flav_names[signal_index]),
                                    reference = True)

    histplot_eta.add(puma.Histogram(PT_ETA_2D[background_index].sum(axis=0),
                                    bin_edges = eta_edges,
                                    flavour = flav_names[background_index]))

    histplot_eta.draw()
    histplot_eta.savefig(plotting_path+"/eta_"+sig_name+"."+filetype)

    # 2D Dist:

    fig = plt.figure()
    ax = plt.axes((0.1,0.1,0.8,0.8))
    ax.set_aspect("auto")
    plt.matshow(np.log(PT_ETA_RATIO_2D),fignum=0)
    plt.gca().set_aspect(BINS_ETA/BINS)
    cbar = plt.colorbar(label = "log("+sig_name+"/light)")

    ax.set_xticks(eta_edges[0:-1:4]*eta_scale)
    ax.set_yticks((pt_edges_ticks[0:-1:10]-20)*pt_scale)

    ax.set_xticklabels(eta_edges[0:-1:4])
    ax.set_yticklabels(pt_edges_ticks[0:-1:10])

    ax.set_yticks((pt_edges_ticks[0:-1:2]-20)*pt_scale,minor=True)
    ax.set_xticks(eta_edges[0:-1:1]*eta_scale,minor=True)

    ax.set_xlabel("|$\\eta$|")
    ax.set_ylabel("p$_T$ [GeV]")
    ax.xaxis.set_label_position('top') 

    plt.savefig(plotting_path+"/corrections_2D_"+sig_name+"."+filetype)
    plt.clf()
    plt.cla()
    #####

    zoomed_pt_bins = 40
    zoomed_eta_bins = -1

    fig = plt.figure()
    ax = plt.axes((0.1,0.1,0.8,0.8))
    plt.matshow(np.log(PT_ETA_RATIO_2D[:zoomed_pt_bins,:]),fignum=0)
    plt.gca().set_aspect(BINS_ETA/zoomed_pt_bins)
    cbar = plt.colorbar(label = "log("+sig_name+"/light)")

    zoomed_tick_bin_pt = int(np.floor(pt_edges[zoomed_pt_bins]))-20

    ax.set_xticks(eta_edges[0:zoomed_eta_bins:4]*eta_scale)
    ax.set_yticks((pt_edges_ticks[0:zoomed_tick_bin_pt:5]-20)*pt_scale)

    ax.set_xticklabels(eta_edges[0:zoomed_eta_bins:4])
    ax.set_yticklabels(pt_edges_ticks[0:zoomed_tick_bin_pt:5])

    ax.set_yticks((pt_edges_ticks[0:zoomed_tick_bin_pt:1]-20)*pt_scale,minor=True)
    ax.set_xticks(eta_edges[0:zoomed_eta_bins:1]*eta_scale,minor=True)
    ""
    ax.set_xlabel("|$\\eta$|")
    ax.set_ylabel("p$_T$ [GeV]")
    ax.xaxis.set_label_position('top') 

    plt.savefig(plotting_path+"/Corrections_2D_"+sig_name+"_zoomed."+filetype)
