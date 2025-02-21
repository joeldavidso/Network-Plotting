import numpy as np # type: ignore
import h5py # type: ignore
import pandas as pd # type: ignore
import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore
import puma # type: ignore
from ftag import Flavours # type: ignore
import os # type: ignore

###################################
####       Configuration       ####
###################################

MCD_name = "$u$ vs $c$ vs $b$ vs $\\tau$"

dir_path = "NetworkSamples/100_100_40_10/"
file_end = "_test.h5"

## Locations for MCD Networks
## Order should be in: 4.6GeV > 1GeV > Global
#                      > Lucas Vars > Lucas Vars Global > 
#                      > All Vars 4.6GeV > All Vars Global 4.6GeV > All Vars Global 1GeV
## MAXIMUM 3

NET_SELECT = [0]

sample_names = ["4dot6GeV",
                "1GeV",
                "global"]
net_names = ["@ 4.6GeV",
             "@ 1GeV",
             "(global)"]
net_config_names = ["GNTau_6",
                    "GNTau_6",
                    "GNTau_6"]

## True = GN2 and LVT included in network list | False = Not included
comp_prior = True

## Select which discriminant corrections to use
## Available Choices: 
## |"None" = No Correction|"1D" = Only Correct in pT|"2D" = Correct in Both pT and eta|"Both" = Correct and plot with both|
correct = "1D"

filetype = "png"

## order nets for plotting and create directories
NET_SELECT.sort()
pop_count = 0
for i in range(len(sample_names)):
    if i not in NET_SELECT:
        sample_names.pop(i-pop_count)
        net_names.pop(i-pop_count)
        net_config_names.pop(i-pop_count)
        pop_count += 1

sample_filepaths = []
for i in sample_names:
    sample_filepaths.append(dir_path+i+file_end)

for i,j in enumerate(net_names):
    net_names[i] = MCD_name+" "+j


## plotting variables
linestyles = ["solid","dashed","dotted","dashdot"]
linecolours = ["mediumviolet","darkorange","forestgreen","darkblue"]

## Sets plotting area
plotting_path = "NetworkPlots/Comp_"+filetype+"/"
single_plotting_path = "NetworkPlots/Comp_"+filetype+"/_"

if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)

for name in sample_names:
    plotting_path = plotting_path + "_"+ name

if comp_prior:
    plotting_path = plotting_path + "_GN2_LVT"

plotting_path = plotting_path + "/"

if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)

## Misc variables
njets = -1

filetype = "png"

vver_WP = 0.85

flav_names = ["ujets","cjets","bjets","taujets"]
prob_names = ["pu","pc","pb","ptau"]

## function definitions

def tau_disc(arr: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return np.log((arr[1]+eps)/(arr[0]+eps))

def b_disc(arr:np.ndarray, f_c: float = 0.018) -> np.ndarray:
    eps = 1e-8
    return np.log((arr[2]+eps)/(f_c*arr[1]+(1-f_c)*arr[0]+eps))

def working_point_disc(arr,wp):
	temp_arr = np.sort(arr)
	cut_off =int(((100-wp)/100)*len(temp_arr))
	return(temp_arr[cut_off])
    
###################################
####      File Collection      ####
###################################

print("Opening Files!")

## Creates all relative arrays
NN_dataframes = []
is_light =[]
is_c =[]
is_b = []
is_tau = []

## Loop over ALL jets then TRACKLESS jets
for Network_count, Network in enumerate(sample_filepaths):
    with h5py.File(Network,"r") as h5file:

        jet_arrs = pd.DataFrame(h5file["jets"][:])


        NN_dataframes.append(pd.DataFrame(
            {
                "pu" : h5file["jets"][net_config_names[Network_count]+"_pu"],
                "pc" : h5file["jets"][net_config_names[Network_count]+"_pc"],
                "pb" : h5file["jets"][net_config_names[Network_count]+"_pb"],
                "ptau" : h5file["jets"][net_config_names[Network_count]+"_ptau"]
            }
        ))

        if Network == sample_filepaths[0]:

            is_light = jet_arrs["HadronConeExclTruthLabelID"] == 0
            is_c = jet_arrs["HadronConeExclTruthLabelID"] == 4
            is_b = jet_arrs["HadronConeExclTruthLabelID"] == 5
            is_tau = jet_arrs["HadronConeExclTruthLabelID"] == 15

            Flav_Bools = [is_light,is_c,is_b,is_tau]

        if comp_prior and Network == sample_filepaths[0]:
                    
            GN2_dataframes = pd.DataFrame(
                {
                    "pu" : h5file["jets"]["fastGN2_pu"],
                    "pc" : h5file["jets"]["fastGN2_pc"],
                    "pb" : h5file["jets"]["fastGN2_pb"]
                }
            )

            LVT_dataframes = pd.DataFrame(
                {
                    "pu" : h5file["jets"]["GNTau_pu"],
                    "ptau" : h5file["jets"]["GNTau_ptau"]
                }
            )

###################################
####     Disc Calculations     ####
###################################

print("Calculating Discriminants!")

NN_b_discs =[]
NN_tau_discs = []

for Network_count in range(len(sample_filepaths)):
    NN_b_discs.append(np.apply_along_axis(b_disc, 0, [NN_dataframes[Network_count]["pu"],
                                                        NN_dataframes[Network_count]["pc"],
                                                        NN_dataframes[Network_count]["pb"]]))

    NN_tau_discs.append(np.apply_along_axis(tau_disc, 0, [NN_dataframes[Network_count]["pu"],
                                                          NN_dataframes[Network_count]["ptau"]]))

if comp_prior:

    GN2_discs = np.apply_along_axis(b_disc, 0 ,[GN2_dataframes["pu"],
                                                GN2_dataframes["pc"],
                                                GN2_dataframes["pb"]])

    LVT_discs = np.apply_along_axis(tau_disc, 0, [LVT_dataframes["pu"],
                                                  LVT_dataframes["ptau"]])

working_points = [95,85,70,50]
working_point_labels = ["95%","85%","70%","50%"]
NN_b_working_point_values = []
NN_tau_working_point_values = []
GN2_working_point_values = []
LVT_working_point_values = []

for Network_count in range(len(sample_filepaths)):
    NN_b_working_point_values_temp = []
    NN_tau_working_point_values_temp = []

    for WP in working_points:
        NN_b_working_point_values_temp.append(working_point_disc(NN_b_discs[Network_count][is_b],WP))
        NN_tau_working_point_values_temp.append(working_point_disc(NN_tau_discs[Network_count][is_tau],WP))

    NN_b_working_point_values.append(NN_b_working_point_values_temp)
    NN_tau_working_point_values.append(NN_tau_working_point_values_temp)


if comp_prior:
    GN2_working_point_values_temp =[]
    LVT_working_point_values_temp = []

    for WP in working_points:
        GN2_working_point_values_temp.append(working_point_disc(GN2_discs[is_b],WP))
        LVT_working_point_values_temp.append(working_point_disc(LVT_discs[is_tau],WP))

    GN2_working_point_values.append(GN2_working_point_values_temp)
    LVT_working_point_values.append(LVT_working_point_values_temp)

###################################
####  Correction Calculations  ####
###################################


# ## Disc Correction Bins

# corr_bins_pt = np.load("NetworkCorrections/4dot6GeV_train/corr_bins_pt.npy")
# corr_bins_eta = np.load("NetworkCorrections/4dot6GeV_train/corr_bins_eta.npy")

# corr_bins_pt = np.append(corr_bins_pt,1e6)
# corr_bins_eta = np.append(corr_bins_eta,1e6)

# ## 1D Disc Corrections

# corr_arr_pt = np.load("NetworkCorrections/4dot6GeV_train/corr_pt_b.npy")
# corr_arr_pt = np.append(corr_arr_pt, [1])

# corr_indexed_pt = np.digitize(jet_arrs["pt"]/1e3, corr_bins_pt, right=True)
# corr_values_pt = np.array([corr_arr_pt[i-1] for i in corr_indexed_pt])


# ## 2D Disc Corrections

# corr_arr_2D_b = np.load("NetworkCorrections/4dot6GeV_train/corr_2D_b.npy")

# corr_add_1 = np.ones((len(corr_arr_2D_b[:,0]),1),corr_arr_2D_b.dtype)
# corr_arr_2D_b = np.concatenate((corr_arr_2D_b,corr_add_1),1)

# corr_add_2 = np.ones((1,len(corr_arr_2D_b[0])),corr_arr_2D_b.dtype)
# corr_arr_2D_b = np.concatenate((corr_arr_2D_b,corr_add_2),0)

# corr_indexed_2D_pt = np.digitize(jet_arrs["pt"]/1e3, corr_bins_pt, right=True)
# corr_indexed_2D_eta = np.digitize(np.absolute(jet_arrs["eta"]), corr_bins_eta, right=True)

# corr_values_2D_b = []

# for i in range(len(corr_indexed_pt)):
#     corr_values_2D_b.append(corr_arr_2D_b[corr_indexed_2D_pt[i]-1][corr_indexed_2D_eta[i]-1])

# corr_values_2D_b = np.array(corr_values_2D_b)


# ## Change the NN_b discs

# # NN_b_discs[dump] = NN_b_discs[dump] - np.log(corr_values_pt)
# # NN_b_discs[dump] = NN_b_discs[dump] - np.log(corr_values_2D_b)


###################################
####  Rejection Calculations   ####
###################################


sig_eff = np.linspace(0.5,1,50)

NN_b_rejs = []
NN_tau_rejs = []

for Network_count in range(len(sample_filepaths)):
    NN_b_rejs.append(puma.metrics.calc_rej(NN_b_discs[Network_count][is_b], NN_b_discs[Network_count][is_light], sig_eff))
    NN_tau_rejs.append(puma.metrics.calc_rej(NN_tau_discs[Network_count][is_tau], NN_tau_discs[Network_count][is_light], sig_eff))

if comp_prior:
    GN2_rejs = puma.metrics.calc_rej(GN2_discs[is_b], GN2_discs[is_light], sig_eff)
    LVT_rejs = puma.metrics.calc_rej(LVT_discs[is_tau], LVT_discs[is_light], sig_eff)

###################################
####       Prob Plotting       ####
###################################

print("Plotting!")

out_names = ["pu", "pc", "pb", "ptau"]

for Network_count, Network in enumerate(net_names):

    if not os.path.exists(single_plotting_path+sample_names[Network_count]+"/outputs/"):
        os.mkdir(single_plotting_path+sample_names[Network_count]+"/outputs/") 

    for output_count, output in enumerate(out_names):

        output_plot = puma.HistogramPlot(xlabel = "Network "+output,
                                         ylabel = "Normalized No. jets",
                                         atlas_second_tag = Network,
                                         bins = 40,
                                         norm = True,
                                         underoverflow = False)
        
        for flavour_count, flavour in enumerate(flav_names):
            output_plot.add(puma.Histogram(jet_arrs[Flav_Bools[flavour_count]][net_config_names[Network_count]+"_"+output],
                                           flavour = flavour))

        output_plot.draw()
        output_plot.savefig(single_plotting_path+sample_names[Network_count]+"/outputs/"+output+"."+filetype)

###################################
####       Disc Plotting       ####
###################################

disc_names = ["tau","b"]

for Network_count, Network in enumerate(net_names):

    if not os.path.exists(single_plotting_path+sample_names[Network_count]+"/discs/"):
        os.mkdir(single_plotting_path+sample_names[Network_count]+"/discs/") 

    ## 0 == tau-tagging
    ## 1 == b-tagging
    for disc_type in range(2):

        disc_plot = puma.HistogramPlot(xlabel = disc_names[disc_type]+" discriminant",
                                       ylabel = "Normalized No. jets",
                                       atlas_second_tag = Network,
                                       bins = 40,
                                       norm = True,
                                       underoverflow = False)
        
        for flavour_count, flavour in enumerate(flav_names):
            if disc_type == 0:
                disc_plot.add(puma.Histogram(NN_tau_discs[Network_count][Flav_Bools[flavour_count]],flavour = flavour))
            else:
                disc_plot.add(puma.Histogram(NN_b_discs[Network_count][Flav_Bools[flavour_count]],flavour = flavour))
                
        disc_plot.draw()
        disc_plot.savefig(single_plotting_path+sample_names[Network_count]+"/discs/"+disc_names[disc_type]+"_disc."+filetype)

###################################
####     Int Eff Plotting      ####
###################################

###################################
####        ROC Plotting       ####
###################################

plotting_area = "ROC"
plotting_path = plotting_path+plotting_area

if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)

## B-tagging
roc_plot_b = puma.RocPlot(n_ratio_panels = 1,
                          xlabel = "b Discriminant",
                          ylabel = "light jet rejection",
                          atlas_second_tag = "$\\mathcal{D} = log\\left(\\frac{p_{b}}{f_{c}\\cdot p_{c} + (1-f_{c})\\cdot p_{u})}\\right)$")

for Network_count, Network in enumerate(net_names):

    roc_b = puma.Roc(sig_eff,
                     NN_b_rejs[Network_count],
                     n_test = np.sum(is_light),
                     rej_class = "ujets",
                     signal_class = "bjets",
                     label = Network)

    roc_plot_b.add_roc(roc_b, reference = (Network == net_names[0] and (not comp_prior)) )

if comp_prior:

    roc_b = puma.Roc(sig_eff,
                     GN2_rejs,
                     n_test = np.sum(is_light),
                     rej_class = "ujets",
                     signal_class = "bjets",
                     label = "GN2")

    roc_plot_b.add_roc(roc_b, reference = True)


roc_plot_b.set_ratio_class(1,"ujets")
roc_plot_b.draw()
roc_plot_b.savefig(plotting_path+"/B-tagging."+filetype)

## Tau-tagging
roc_plot_tau = puma.RocPlot(n_ratio_panels = 1,
                            xlabel = "tau Discriminant",
                            ylabel = "light jet rejection",
                            atlas_second_tag = "$\\mathcal{D} = log\\left(\\frac{p_{\\tau}}{p_{u}}\\right)$")

if comp_prior:

    roc_tau = puma.Roc(sig_eff,
                       LVT_rejs,
                       n_test = np.sum(is_light),
                       rej_class = "ujets",
                       signal_class = "taujets",
                       label = "$u$ vs $\tau$")

    roc_plot_tau.add_roc(roc_tau, reference = True)

for Network_count, Network in enumerate(net_names):

    roc_tau = puma.Roc(sig_eff,
                       NN_tau_rejs[Network_count],
                       n_test = np.sum(is_light),
                       rej_class = "ujets",
                       signal_class = "taujets",
                       label = Network)

    roc_plot_tau.add_roc(roc_tau, reference = (Network == net_names[0] and (not comp_prior)) )

roc_plot_tau.set_ratio_class(1,"ujets")
roc_plot_tau.draw()
roc_plot_tau.savefig(plotting_path+"/Tau-tagging."+filetype)


plotting_path = plotting_path[:-len(plotting_area)]

###################################
####  pT vs eff/bkg Plotting   ####
###################################

plotting_area = "pt_eff_rej"
plotting_path = plotting_path+plotting_area

if not os.path.exists(plotting_path):
    os.mkdir(plotting_path)


bin_ranges = [[20, 30, 40, 60, 85, 110, 140, 175, 250], np.linspace(20,29.2,11)]

bin_ranges_names = ["fullpt",
                  "lowpt"]

modes = ["bkg_rej","sig_eff"]
mode_labels_b = ["light-jet rejection","b-jet efficiency"]
mode_labels_tau = ["light-jet rejection","tau-jet efficiency"]

flat_tag = ["inclusive WP = ", 
            "Per Bin WP = "]
flat_name = ["Inclusive", "PerBin"]

for flat_bins in range(2):
    for count, bin_range in enumerate(bin_ranges):


        pt_rej_b_plot = puma.VarVsEffPlot(mode = "bkg_rej",
                                xlabel = "$p_{T}$ [GeV]",
                                ylabel = "light-jet rejection",
                                logy = False,
                                n_ratio_panels = 1,
                                atlas_second_tag = flat_tag[flat_bins]+str(vver_WP*100)+"%"+
                                "\n $\\mathcal{D} = log\\left(\\frac{p_{b}}{f_{c}\\cdot p_{c} + (1-f_{c})\\cdot p_{u})}\\right)$")

        pt_eff_b_plot = puma.VarVsEffPlot(mode = "sig_eff",
                                xlabel = "$p_{T}$ [GeV]",
                                ylabel = "b-jet efficeincy",
                                logy = False,
                                n_ratio_panels = 1,
                                atlas_second_tag = flat_tag[flat_bins]+str(vver_WP*100)+"%"+
                                "\n $\\mathcal{D} = log\\left(\\frac{p_{b}}{f_{c}\\cdot p_{c} + (1-f_{c})\\cdot p_{u})}\\right)$")

        if comp_prior:
            vve_GN2 = puma.VarVsEff(x_var_sig = jet_arrs[is_b]["pt"] / 1e3,
                                    disc_sig = GN2_discs[is_b],
                                    x_var_bkg = jet_arrs[is_light]["pt"] / 1e3,
                                    disc_bkg = GN2_discs[is_light],
                                    bins=bin_range,
                                    working_point = vver_WP,
                                    disc_cut = None,
                                    flat_per_bin = flat_bins,
                                    label = "GN2")

            pt_rej_b_plot.add(vve_GN2, reference = True)
            pt_eff_b_plot.add(vve_GN2, reference = True)

        for Network_count, Network in enumerate(net_names):
            vve_NN_b = puma.VarVsEff(x_var_sig = jet_arrs[is_b]["pt"] / 1e3,
                                     disc_sig = NN_b_discs[Network_count][is_b],
                                     x_var_bkg = jet_arrs[is_light]["pt"] / 1e3,
                                     disc_bkg = NN_b_discs[Network_count][is_light],
                                     bins=bin_range,
                                     working_point = vver_WP,
                                     disc_cut = None,
                                     flat_per_bin = flat_bins,
                                     label = Network)

            pt_rej_b_plot.add(vve_NN_b, reference = (Network == net_names[0] and (not comp_prior)))
            pt_eff_b_plot.add(vve_NN_b, reference = (Network == net_names[0] and (not comp_prior)))


        pt_rej_b_plot.draw()
        pt_rej_b_plot.savefig(plotting_path+"/pt_rej_"+bin_ranges_names[count]+"_"+flat_name[flat_bins]+"."+filetype)


        pt_eff_b_plot.draw()
        pt_eff_b_plot.savefig(plotting_path+"/pt_eff_"+bin_ranges_names[count]+"_"+flat_name[flat_bins]+"."+filetype)

plotting_path = plotting_path[:-len(plotting_area)]

###################################
####  eta vs eff/bkg Plotting  ####
###################################
