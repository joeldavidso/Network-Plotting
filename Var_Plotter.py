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
## Setting preprocess selects plotted variables automagically

filetype = "png"

preprocess = "Lucas_Vars_train"
ttbar_filepath = "NetworkSamples/100_100_40_10/"+preprocess+".h5"

plotting_path = "NetworkPlots/Variables_"+filetype+"/"

if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)

plotting_path = plotting_path+preprocess+"/"

if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)


## Some Global variables

flav_names = ["ujets","cjets","bjets","taujets"]
prob_names = ["pu","pc","pb","ptau"]

Continue = True
loop_counter = 0

## Automagic Variable Selection

# Lists of all possible variables, variable labels and variable bins
# |Variable|Label|nbins|xmin|xmax|logy|whichset( "Joel", "Lucas", "All")|

dtype = np.dtype([("var", "U32"),
                  ("label", "U32"),
                  ("nbins",np.int32),
                  ("xmin",np.float64),
                  ("xmax",np.float64),
                  ("logy",bool),
                  ("set", "U32")])

jet_vars = np.array([("pt", "$p_T$ [MeV]", 40, 20_000, 250_000, True, "All"),
                     ("eta", "$\eta$",30, -2.5, 2.5, False, "All")],
                     dtype = dtype)

track_vars = np.array([("pt", "$p_T$ [MeV]", 40, 20_000, 250_000, True, "All"),
                       ("z0SinTheta", "$z_0sin(\theta)$", 40, -10, 10, False, "Lucas")],
                       dtype = dtype)

## Complex variables

# function definintions

def sum_over_tracks(tracks, variable):
    return np.nansum(tracks[variable], axis = 1)

# array creations

dtype = np.dtype([("name", "U32"),
                  ("var", "U32"),
                  ("label", "U32"),
                  ("nbins",np.int32),
                  ("xmin",np.float64),
                  ("xmax",np.float64),
                  ("logy",bool),
                  ("set", "U32"),
                  ("jet or track", "U32")])

comp_vars = np.array([("ntracks", "valid", "nTracks", 20, 0, 20, False, "All", "jet"),
                      ("pt_summed_trackes", "pt", "summed track $p_T$ [MeV]", 40, 20_000, 250_000, True, "All", "jet")],
                      dtype = dtype)

comp_funcs = [sum_over_tracks, sum_over_tracks]

## Select which variables are plotted

def set_var_false(set):
    return set[np.full(len(set),False)]

def set_var_set(set_name, set):
    not_name = "Joel" if set_name == "Lucas" else "Lucas"
    return set[set["set"] != not_name]

if "test" in preprocess:
    track_vars = set_var_false(track_vars)
    comp_vars = set_var_false(comp_vars)

if "Lucas" in preprocess:
    jet_vars = set_var_set("Lucas", jet_vars)
    track_vars = set_var_set("Lucas", track_vars)
    comp_vars = set_var_set("Lucas", comp_vars)

elif "Joel" in preprocess:
    jet_vars = set_var_set("Joel", jet_vars)
    track_vars = set_var_set("Joel", track_vars)
    comp_vars = set_var_set("Joel", comp_vars)

## Loop Controls

jets_per_batch = 500_000
#Max 500 (250_000_000 Jets)
loop_target = 5

bar_length = 100

## Main Binning Info


jet_binwidths = []
for jet_var in range(len(jet_vars)):
    jet_binwidths.append((jet_vars["xmax"][jet_var]-jet_vars["xmin"][jet_var])/jet_vars["nbins"][jet_var])

track_binwidths = []
for track_var in range(len(track_vars)):
    track_binwidths.append((track_vars["xmax"][track_var]-track_vars["xmin"][track_var])/track_vars["nbins"][track_var])

comp_binwidths = []
for comp_var in range(len(comp_vars)):
    comp_binwidths.append((comp_vars["xmax"][comp_var]-comp_vars["xmin"][comp_var])/comp_vars["nbins"][comp_var])


## Bin edges calculations

jet_bin_edges = []
for jet_var in range(len(jet_vars)):
    jet_bin_edges.append(np.linspace(jet_vars["xmin"][jet_var],jet_vars["xmax"][jet_var],jet_vars["nbins"][jet_var]))

track_bin_edges = []
for track_var in range(len(track_vars)):
    track_bin_edges.append(np.linspace(track_vars["xmin"][track_var],track_vars["xmax"][track_var],track_vars["nbins"][track_var]))

comp_bin_edges = []
for comp_var in range(len(comp_vars)):
    comp_bin_edges.append(np.linspace(comp_vars["xmin"][comp_var],comp_vars["xmax"][comp_var],comp_vars["nbins"][comp_var]))

## variable/arrays for use outside of operations loop

# number of each jet flavour [ujets,cjets,bjets,taujets]
N_flav = [0,0,0,0]

# Histograms for plotting vars (light-jets, c-jets, b-jets, tau-jets)

jet_var_hists = []
for jet_var in range(len(jet_vars)):
    jet_var_hists.append([np.histogram([],bins = jet_bin_edges[jet_var])[0],
                          np.histogram([],bins = jet_bin_edges[jet_var])[0],
                          np.histogram([],bins = jet_bin_edges[jet_var])[0],
                          np.histogram([],bins = jet_bin_edges[jet_var])[0]])

track_var_hists = []
for track_var in range(len(track_vars)):
    track_var_hists.append([np.histogram([],bins = track_bin_edges[track_var])[0],
                            np.histogram([],bins = track_bin_edges[track_var])[0],
                            np.histogram([],bins = track_bin_edges[track_var])[0],
                            np.histogram([],bins = track_bin_edges[track_var])[0]])

comp_var_hists = []
for comp_var in range(len(comp_vars)):
    comp_var_hists.append([np.histogram([],bins = comp_bin_edges[comp_var])[0],
                           np.histogram([],bins = comp_bin_edges[comp_var])[0],
                           np.histogram([],bins = comp_bin_edges[comp_var])[0],
                           np.histogram([],bins = comp_bin_edges[comp_var])[0]])

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

## Flavour Loop

    for count, flav in enumerate(flav_bools):
    ## Flavour Counting

        N_flav[count]+=np.sum(flav)

    ## Hist Additions

        ## jet hists

        for jet_var in range(len(jet_vars)):
            jet_var_hists[jet_var][count] += np.histogram(jets_arr[flav][jet_vars["var"][jet_var]], bins = jet_bin_edges[jet_var])[0]

        ## track hists

        for track_var in range(len(track_vars)):
            track_var_hists[track_var][count] += np.histogram(tracks_arr[flav][track_vars["var"][track_var]], bins = track_bin_edges[track_var])[0]

        ## more complex var hists

        for comp_var in range(len(comp_vars)):
            comp_arr = comp_funcs[comp_var](tracks_arr[flav],comp_vars["var"][comp_var])
            comp_var_hists[comp_var][count] += np.histogram(comp_arr, bins = comp_bin_edges[comp_var])[0]

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
          

## jet var plots

if not os.path.exists(plotting_path+"jet/"):
    os.mkdir(plotting_path+"jet/")

for jet_var in range(len(jet_vars)):

    jet_histplot = puma.HistogramPlot(bins = jet_bin_edges[jet_var],
                                      xlabel = jet_vars["label"][jet_var],
                                      ylabel = "Normalised No. jets",
                                      atlas_second_tag = preprocess,
                                      logy = jet_vars["logy"][jet_var],
                                      norm = True,
                                      underoverflow = False)

    for flav_index in range(4):
        jet_histplot.add(puma.Histogram(jet_var_hists[jet_var][flav_index],
                                        bin_edges = jet_bin_edges[jet_var],
                                        flavour = flav_names[flav_index]))

    jet_histplot.draw()
    jet_histplot.savefig(plotting_path+"jet/"+jet_vars["var"][jet_var]+"."+filetype)


## track var plots

if not os.path.exists(plotting_path+"track/"):
    os.mkdir(plotting_path+"track/")

for track_var in range(len(track_vars)):

    track_histplot = puma.HistogramPlot(bins = track_bin_edges[track_var],
                                      xlabel = track_vars["label"][track_var],
                                      ylabel = "Normalised No. tracks",
                                      atlas_second_tag = preprocess,
                                      logy = track_vars["logy"][track_var],
                                      norm = True,
                                      underoverflow = False)

    for flav_index in range(4):
        track_histplot.add(puma.Histogram(track_var_hists[track_var][flav_index],
                                        bin_edges = track_bin_edges[track_var],
                                        flavour = flav_names[flav_index]))

    track_histplot.draw()
    track_histplot.savefig(plotting_path+"track/"+track_vars["var"][track_var]+"."+filetype)

## comp var plots

for comp_var in range(len(comp_vars)):

    comp_histplot = puma.HistogramPlot(bins = comp_bin_edges[comp_var],
                                       xlabel = comp_vars["label"][comp_var],
                                       ylabel = "Normalised No. jets",
                                       atlas_second_tag = preprocess,
                                       logy = comp_vars["logy"][comp_var],
                                       norm = True,
                                       underoverflow = False)

    for flav_index in range(4):
        comp_histplot.add(puma.Histogram(comp_var_hists[comp_var][flav_index],
                                         bin_edges = comp_bin_edges[comp_var],
                                         flavour = flav_names[flav_index]))

    comp_histplot.draw()
    comp_histplot.savefig(plotting_path+comp_vars["jet or track"][comp_var]+"/"+comp_vars["name"][comp_var]+"."+filetype)
