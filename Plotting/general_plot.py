#!/bin/env python
# Author: Lin (lin1209@purdue.edu)
# Edits by: Dylan Fortney (ddfortne@purdue.edu

# Editing to expand usability of bar plot function. Want to make it function can accept one dictionary with any number of data sets, allowing any number of adjacent bars.

import sys
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import cm
from scipy import stats

def main(argv):

    # scatter plot
    # fake data
    # xs = {'linear':np.arange(1,10),'quadratic':np.arange(1,10)}
    # ys = {'linear':np.arange(1,10)*5,'quadratic':np.square(np.arange(1,10))}
    # plot_scatter(xs,ys,'scatter_test.png')

    # bar plot
    # fake data
    # stable = {'1C':90,'2C':65,'3C':80,'4C':35}
    # unstable = {'1C':20,'2C':24,'3C':50,'4C':36} 
    # All_data={'stable':stable, 'unstable':unstable}
    # plot_bar(All_data,'bar_test.png')
    # trial64={"Bead 1": 5.193, "Bead 2": 3.489,"Bead 3": 5.587,"Bead 4": 2.847,"Bead 5": 9.795}
    # trial65={"Bead 1": 2.0, "Bead 2": 2.947,"Bead 3": 7.001,"Bead 4": 3.987,"Bead 5": 8.155}
    # trial66={"Bead 1": 3.211, "Bead 2": 2.,"Bead 3": 5.837,"Bead 4": 2.,"Bead 5": 8.935}
    # trial67={"Bead 1": 3.397, "Bead 2": 3.576,"Bead 3": 5.104,"Bead 4": 3.941,"Bead 5": 8.155}
    # trial68={"Bead 1": 2.009, "Bead 2": 2.705,"Bead 3": 3.754,"Bead 4": 5.333,"Bead 5": 7.457}
    # trial64={"Bead 1": 5.615, "Bead 2": 4.981,"Bead 3": 5.915,"Bead 4": 2.0,"Bead 5": 6.33}
    # trial65={"Bead 1": 5.615, "Bead 2": 5.224,"Bead 3": 2.06,"Bead 4": 2.74,"Bead 5": 6.55}
    # trial66={"Bead 1": 4.12, "Bead 2": 3.92,"Bead 3": 5.74,"Bead 4": 2.,"Bead 5": 6.295}
    # trial67={"Bead 1": 5.35, "Bead 2": 5.47,"Bead 3": 2.30,"Bead 4": 2.83,"Bead 5": 6.22}
    # trial68={"Bead 1": 8.73, "Bead 2": 7.71,"Bead 3": 4.17,"Bead 4": 2.57,"Bead 5": 9.49}
    # All_data={"trial64":trial64, "trial65":trial65, "trial66":trial66, "trial67":trial67, "trial68":trial68}

    # Loss={"NPT": 78000/110561, "Cold": 74000/110561,"Ramp": 69000/110561}
    # Press={"NPT": 112000/110561, "Cold": 96000/110561,"Ramp": 108000/110561}
    # SLoss={"NPT": 87000/110561, "Cold": 83000/110561,"Ramp": 88000/110561}

    Loss={"NPT": 127000/38394, "Cold": 114000/38394,"Ramp": 109000/38394}
    Press={"NPT": 87000/38394, "Cold": 75000/38394,"Ramp": 80000/38394}
    SLoss={"NPT": 59000/38394, "Cold": 56000/38394,"Ramp": 59000/38394}

    # trial69={"Bead 1": 4.00, "Bead 2": 4.14,"Bead 3": 8.99,"Bead 4": 2.27,"Bead 5": 10.}
    # trial70={"Bead 1": 2.0, "Bead 2": 6.21,"Bead 3": 6.43,"Bead 4": 2.32,"Bead 5": 2.}
    # trial71={"Bead 1": 4.785, "Bead 2": 6.38,"Bead 3": 4.87,"Bead 4": 4.99,"Bead 5": 5.98}
    # trial72={"Bead 1": 6.09, "Bead 2": 2.68,"Bead 3": 9.73,"Bead 4": 9.43,"Bead 5": 4.24}
    # trial73={"Bead 1": 5.85, "Bead 2": 5.35,"Bead 3": 7.39,"Bead 4": 2.0,"Bead 5": 4.62}
    # trial69={"Bead 1": 5.21, "Bead 2": 3.99,"Bead 3": 2.33,"Bead 4": 2.0,"Bead 5": 6.10}
    # trial70={"Bead 1": 5.64, "Bead 2": 5.43,"Bead 3": 2.95,"Bead 4": 3.93,"Bead 5": 2.60}
    # trial71={"Bead 1": 5.92, "Bead 2": 2.28,"Bead 3": 3.24,"Bead 4": 3.90,"Bead 5": 4.03}
    # trial72={"Bead 1": 3.59, "Bead 2": 3.85,"Bead 3": 5.21,"Bead 4": 2.83,"Bead 5": 3.33}
    # trial73={"Bead 1": 3.31, "Bead 2": 7.51,"Bead 3": 2.0,"Bead 4": 2.25,"Bead 5": 2.0}
    #All_data={"trial69":trial69, "trial70":trial70, "trial71":trial71, "trial72":trial72, "trial73":trial73}

    # trial74={"Bead 1": 4.65, "Bead 2": 2.81,"Bead 3": 2.51,"Bead 4": 3.49,"Bead 5": 8.45}
    # trial75={"Bead 1": 3.22, "Bead 2": 4.27,"Bead 3": 4.38,"Bead 4": 2.00,"Bead 5": 9.65}
    # trial76={"Bead 1": 2.0, "Bead 2": 3.84,"Bead 3": 4.53,"Bead 4": 4.75,"Bead 5": 7.54}
    # trial77={"Bead 1": 2.16, "Bead 2": 4.03,"Bead 3": 3.12,"Bead 4": 5.17,"Bead 5": 10}
    # trial78={"Bead 1": 3.84, "Bead 2": 4.23,"Bead 3": 6.3,"Bead 4": 3.03,"Bead 5": 10}
    # trial74={"Bead 1": 2.46, "Bead 2": 5.19,"Bead 3": 7.28,"Bead 4": 4.87,"Bead 5": 6.15}
    # trial75={"Bead 1": 5.77, "Bead 2": 4.99,"Bead 3": 5.20,"Bead 4": 2.59,"Bead 5": 6.83}
    # trial76={"Bead 1": 5.62, "Bead 2": 5.75,"Bead 3": 2.0,"Bead 4": 2.69,"Bead 5": 7.09}
    # trial77={"Bead 1": 4.82, "Bead 2": 2.00,"Bead 3": 2.13,"Bead 4": 2.21,"Bead 5": 6.30}
    # trial78={"Bead 1": 5.74, "Bead 2": 5.7,"Bead 3": 2.0,"Bead 4": 3.44,"Bead 5": 8.02}
    # All_data={"trial74":trial74, "trial75":trial75, "trial76":trial76, "trial77":trial77, "trial78":trial78}

    # Loss={"NPT": 180000/110561, "Cold": 154000/110561,"Ramp": 158000/110561}
    # Press={"NPT": 95000/110561, "Cold": 84000/110561,"Ramp": 92000/110561}
    # SLoss={"NPT": 198000/110561, "Cold": 180000/110561,"Ramp": 193000/110561}
    # Loss={"NPT": 59000/38394, "Cold": 56000/38394,"Ramp": 57000/38394}
    # Press={"NPT": 39000/38394, "Cold": 38200/38394,"Ramp": 39000/38394}
    # SLoss={"NPT": 64000/38394, "Cold": 59000/38394,"Ramp": 62000/38394}
    # All_data={"Loss":Loss, "Press":Press, "SLoss":SLoss}

    # Loss={"NPT": 270474/110561, "Cold": 199635/110561,"Ramp": 199449/110561}
    # Press={"NPT": 156393/110561, "Cold": 131379/110561,"Ramp":  149839/110561}
    # SLoss={"NPT": 188983/110561, "Cold": 160339/110561,"Ramp": 178298/110561}
    # Loss={"NPT": 46225/38394, "Cold": 43996/38394,"Ramp": 46073/38394}
    # Press={"NPT": 40340/38394, "Cold": 38539/38394,"Ramp": 39722/38394}
    # SLoss={"NPT": 27674/38394, "Cold": 27883/38394,"Ramp": 27020.9741/38394}
    # All_data={"Loss":Loss, "Press":Press, "SLoss":SLoss}

    # trial79={"Bead 1": 5.26, "Bead 2": 6.13,"Bead 3": 2.31,"Bead 4": 2.44,"Bead 5": 7.62}
    # trial80={"Bead 1": 5.94, "Bead 2": 8.62,"Bead 3": 3.22,"Bead 4": 2.73,"Bead 5": 9.61}
    # trial81={"Bead 1": 5.82, "Bead 2": 5.63,"Bead 3": 3.18,"Bead 4": 2.69,"Bead 5": 6.74}
    # trial82={"Bead 1": 7.16, "Bead 2": 5.79,"Bead 3": 2.,"Bead 4": 2.29,"Bead 5": 9.85}
    # trial83={"Bead 1": 3.42, "Bead 2": 5.42,"Bead 3": 2.0,"Bead 4": 2.,"Bead 5": 6.295}
    # trial79={"Bead 1": 2.0, "Bead 2": 3.48,"Bead 3": 3.47,"Bead 4": 5.99,"Bead 5": 7.44}
    # trial80={"Bead 1": 2.0, "Bead 2": 2.0,"Bead 3": 2.15,"Bead 4": 5.22,"Bead 5": 8.03}
    # trial81={"Bead 1": 3.77, "Bead 2": 2.70,"Bead 3": 6.11,"Bead 4": 4.31,"Bead 5": 8.83}
    # trial82={"Bead 1": 3.18, "Bead 2": 4.23,"Bead 3": 5.29,"Bead 4": 2.0,"Bead 5": 10.}
    # trial83={"Bead 1": 3.55, "Bead 2": 3.74,"Bead 3": 2.0,"Bead 4": 2.45,"Bead 5": 8.71}
    # trial141={"Bead 1": 3.26, "Bead 2": 3.03,"Bead 3": 5.40,"Bead 4": 2.83,"Bead 5": 8.54}
    # trial142={"Bead 1": 3.01, "Bead 2": 3.38,"Bead 3": 5.31,"Bead 4": 4.00,"Bead 5": 8.49}
    # trial143={"Bead 1": 3.11, "Bead 2": 3.78,"Bead 3": 2.00,"Bead 4": 3.72,"Bead 5": 8.10}
    # trial144={"Bead 1": 3.45, "Bead 2": 3.21,"Bead 3": 6.125,"Bead 4": 3.42,"Bead 5": 8.155}
    # trial145={"Bead 1": 3.48, "Bead 2": 3.24,"Bead 3": 2.00,"Bead 4": 4.11,"Bead 5": 8.50}
    # trial146={"Bead 1": 3.24, "Bead 2": 3.16,"Bead 3": 5.09,"Bead 4": 3.26,"Bead 5": 8.41}
    # trial147={"Bead 1": 3.24, "Bead 2": 3.47,"Bead 3": 2.00,"Bead 4": 4.21,"Bead 5": 8.31}
    # trial148={"Bead 1": 3.14, "Bead 2": 3.69,"Bead 3": 2.18,"Bead 4": 4.10,"Bead 5": 8.16}
    # trial149={"Bead 1": 3.19, "Bead 2": 3.66,"Bead 3": 2.00,"Bead 4": 4.12,"Bead 5": 8.32}
    # trial150={"Bead 1": 3.43, "Bead 2": 3.53,"Bead 3": 2.25,"Bead 4": 3.60,"Bead 5": 8.155}
    # trial151={"Bead 1": 3.51, "Bead 2": 2.90,"Bead 3": 6.39,"Bead 4": 4.27,"Bead 5": 8.11}
    # trial152={"Bead 1": 3.70, "Bead 2": 2.76,"Bead 3": 5.01,"Bead 4": 3.32,"Bead 5": 8.66}
    # trial153={"Bead 1": 3.21, "Bead 2": 3.28,"Bead 3": 2.00,"Bead 4": 3.52,"Bead 5": 8.43}
    # trial154={"Bead 1": 3.62, "Bead 2": 3.43,"Bead 3": 2.00,"Bead 4": 4.42,"Bead 5": 8.37}
    # trial155={"Bead 1": 4.01, "Bead 2": 3.15,"Bead 3": 4.68,"Bead 4": 2.42,"Bead 5": 8.25}
    # All_data={"Replicate 1":trial151, "Replicate 2":trial152, "Replicate 3":trial153, "Replicate 4":trial154, "Replicate 5":trial155}
    # trial1={"mult": 57.92, "add": 56.28,"frac": 52.02}
    # trial2={"mult": 57.86, "add": 51.50,"frac": 60.34}
    # trial3={"mult": 44.64, "add": 47.25,"frac": 57.81}
    # trial4={"mult": 59.12, "add": 49.52,"frac": 49.67}
    # trial5={"mult": 55.18, "add": 52.45,"frac": 51.40}

    # trial1={"mult": 3452.05, "add": 861.97,"frac": 2055.69}
    # trial2={"mult": 10392.51, "add": 3652.43,"frac": 8004.74}
    # trial3={"mult": 8076.48, "add": 398.25,"frac": 2321.58}
    # trial4={"mult": 8197.16, "add": 546.08,"frac": 9390.61}
    # trial5={"mult": 5092.01, "add": 991.79,"frac": 8621.72}

    # 166_NDI_2TEMPO:  (-1233.6021252482437, 143.2030503, -1924.697425)
    # 167_NDI_2TEMPO:  (-789.0552977613843, 2036.148036, -1596.375765)
    # 168_NDI_2TEMPO:  (-1165.0632870413363, -49.11148608, -2640.948621)
    # 169_NDI_2TEMPO:  (-1094.3228461684485, 1892.664714, 61.29811095)
    # 170_NDI_2TEMPO:  (-807.091874721269, 1562.527348, -3147.981056)

    # trial1={"without": 861.97,"previous": 710.9, "current":1233.6021252482437}
    # trial2={"without": 3652.43,"previous": 6240.4, "current":789.0552977613843}
    # trial3={"without": 398.25,"previous": 2511, "current":1165.0632870413363}
    # trial4={"without": 546.08,"previous": 4439, "current":1094.3228461684485}
    # trial5={"without": 991.79,"previous": 5872, "current":807.091874721269}
    # All_data={"Replicate 1":trial1, "Replicate 2":trial2, "Replicate 3":trial3, "Replicate 4":trial4, "Replicate 5":trial5}

    # plot_bar(All_data, 'NDI166-170_Press.png', ["Pressure", "Inclusion of longer NVT runs", "Pressure Value"])#, plot_max=2.0)

    xs_list = np.linspace(0,5,100)
    y_model=2*xs_list/(xs_list+2)+1

    xs={"$\Theta_{scale}$":xs_list, "$\Delta$$\Theta$":xs_list}
    ys={"$\Theta_{scale}$":y_model, "$\Delta$$\Theta$":xs_list}
    plot_scatter(xs, ys, "Theta_Models.pdf", ymax_set=3.0, ymin_set=0.0, fig_vals=["$\Theta$ Scaling", "$\Delta$$\Theta$", "Value"])

# modify from TAFFI plot function
# xs and ys will be dictionary with keys being label and values being list of x or y
def plot_scatter(xs,ys,fig_name,xmax_set='',ymax_set='',xmin_set='',ymin_set='',fig_vals=["plot","x", "y"]):


    # Generate fit plot
    color_list = [(0.05,0.35,0.75),(0.05,0.8,0.6),(0.9,0.3,0.05),(0.35,0.7,0.9),(0.9,0.5,0.7),(0.9,0.6,0.05),(0.95,0.9,0.25),(0.05,0.05,0.05)]*10   # Supposedly more CB-friendly
    fig = plt.figure(figsize=(6,5))
    ax = plt.subplot(111)

    # Plot the scatter data
    for count_i,i in enumerate(xs):   
        # change line width if you want the line
        # change alpha for opacity 
        ax.plot(xs[i],ys[i],marker='.',markersize=20,markeredgewidth=0.0,linestyle='-',linewidth=0.0,alpha=1.0,color=color_list[count_i],label=i)

    # Set limits based on the largest range

    y_min,y_max = ax.get_ylim()
    x_min,x_max = ax.get_xlim()
    if ymax_set:
        y_max=ymax_set
    if xmax_set:
        x_max=xmax_set
    if type(ymin_set)==float:
        y_min=ymin_set
    if xmin_set:
        x_min=xmin_set

    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])


    # Generate Legend
    ax.legend(loc='best',frameon=False)
    handles, labels = ax.get_legend_handles_labels()
    # put the legend outside
    #lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5),fontsize=16)
    # put the legned inside
    lgd = ax.legend(handles, labels, loc='best',fontsize=16)

    # Format ticks
    ax.tick_params(axis='both', which='major',labelsize=24,pad=10,direction='out',width=3,length=6)
    ax.tick_params(axis='both', which='minor',labelsize=24,pad=10,direction='out',width=2,length=4)
    [j.set_linewidth(3) for j in ax.spines.values()]
    plt.title(fig_vals[0],fontsize=32,fontweight='bold')
    # Set Labels and Save Name
    ax.set_xlabel(fig_vals[1],fontsize=32,labelpad=10,fontweight='bold')
    ax.set_ylabel(fig_vals[2],fontsize=32,labelpad=10,fontweight='bold')

    # Save the figure
    plt.savefig(fig_name, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)

    return

# modified from Qiyuan's script
# y and y2 are dictionaries with keys being the categroy names
def plot_bar(y,fig_name,fig_vals,plot_max=''):
    # fig_vals=[title, xlabel, ylabel]
    color_list = [(0.05,0.35,0.75),(0.05,0.8,0.6),(0.9,0.3,0.05),(0.35,0.7,0.9),(0.9,0.5,0.7),(0.9,0.6,0.05),(0.95,0.9,0.25),(0.05,0.05,0.05)]*10   # Supposedly more CB-friendly

    # set font
    #plt.rcParams['font.family']     = 'sans-serif'
    #plt.rcParams['font.sans-serif'] = 'Helvetica'

    # set the style of the axes and the text color
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams['axes.edgecolor']='black'
    plt.rcParams['axes.linewidth']=1.0
    plt.rcParams['xtick.color']='black'
    plt.rcParams['ytick.color']='black'
    plt.rcParams['text.color']='black'

    ######################################################
    ################   Input    data     #################
    ######################################################
    #'''

    # we first need a numeric placeholder for the y axis
    bar_types=list(y.keys()) # Get the different bars that will be plotted.
    #print(bar_types)
    my_range=list(range(1,len(list(y[bar_types[0]].keys()))+1)) # Get the length of each bar type.
    #print(my_range)
    my_range_all=[[(j-1)*(len(bar_types)+1)*0.4+(i)*0.4 for j in my_range] for i in range(len(bar_types))]
    #print(my_range_all)
    #    my_rangea = [i-0.2 for i in my_range]
    #    print(my_rangea)
    #    my_rangeb = [i+0.2 for i in my_range]   
    #    print(my_rangeb) 
    fig, ax = plt.subplots(figsize=(8, 8))
    data_max=0.0
    # create for each expense type an horizontal line that starts at x = 0 with the length 
    # represented by the specific expense percentage value 
    for n,l in enumerate(bar_types): # n = label number, l is label name
        plt.vlines(x=my_range_all[n], ymin=0, ymax=list(y[l].values()), color=color_list[n], alpha=0.4, linewidth=14)
        # create for each expense type a dot at the level of the expense percentage value
        plt.plot(my_range_all[n],list(y[l].values()), "o", markersize=14, color=color_list[n], alpha=0.7,label=l)
    #plt.vlines(x=my_rangeb, ymin=0, ymax=list(y2.values()), color='#00AF00', alpha=0.4, linewidth=20)
    #plt.plot(my_rangeb,list(y2.values()), "o", markersize=18, color='#00AF00', alpha=0.7,label='unstable')
        if max(y[l].values())>=data_max:
            data_max=max(y[l].values())
    # label for each coloum
    #csfont = {'fontname':'Helvetica','fontsize':13,'fontweight':'black'}  
        csfont = {'fontsize':20,'fontweight':'black'}  
        for cx,x in enumerate(my_range_all[n]):
            height = list(y[l].values())[cx]+2
            if type(list(y[l].values()))==int:
                ax.annotate('{:2d}'.format(list(y[l].values())[cx]),
                            xy=(x, height),
                            xytext=(0, 10),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',**csfont)
            elif type(list(y[l].values()))==float:
                ax.annotate('{:2f}'.format(list(y[l].values())[cx]),
                            xy=(x, height),
                            xytext=(0, 10),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',**csfont)

    #    for cx,x in enumerate(my_rangeb):
    #        height = list(y2.values())[cx]+2
    #        ax.annotate('{:2d}'.format(list(y2.values())[cx]),
    #                    xy=(x, height),
    #                    xytext=(0, 10),  # 10 points vertical offset
    #                    textcoords="offset points",
    #                    ha='center', va='bottom',**csfont)

    # Generate Legend
    ax.legend(bbox_to_anchor=(1,1), loc="upper left",frameon=False)
    handles, labels = ax.get_legend_handles_labels()
    #font = font_manager.FontProperties(family='Helvetica',
    #                                weight='normal',
    #                                style='normal', size=13)
    font = font_manager.FontProperties(weight='black',
                                    style='normal', size=20)

    #lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5),prop=font)
    #lgd = ax.legend(handles, labels, loc='best', bbox_to_anchor=(1,0.5),prop=font,handlelength=2.5)
    lgd = ax.legend(handles, labels, bbox_to_anchor=(1,1), loc="upper left",prop=font,handlelength=2.5)


    # set labels
    ax.set_xlabel(fig_vals[1], fontsize=24, fontweight='black', color = 'black')
    ax.set_ylabel(fig_vals[2], fontsize=24, fontweight='black', color = 'black')


    # set lim
    if plot_max:
        plt.ylim(0,plot_max)
    else:
        plt.ylim(0, int(math.ceil(data_max*1.1)))

    # set axis
    #tickfont = {'fontname':'Helvetica','fontsize':24}
    tickfont = {'fontsize':20}
    mra_np=np.array(my_range_all)
    x_tick_list=[statistics.mean(mra_np[:,r-1]) for r in my_range]
    plt.xticks(x_tick_list, list(y[bar_types[0]].keys()),**tickfont)
    plt.yticks(**tickfont)
    ax.tick_params(axis="y",direction="in")
    plt.title(fig_vals[0], fontsize=24, fontweight='black', color = 'black')


    # set the spines position
    plt.savefig('{}'.format(fig_name), dpi=300, bbox_inches='tight')

def plot_density(x, y, name, label_list, color_map="viridis", color_min=0.0, save_it=True, close_it=True):
    # x: x data array, y: y data array, name: name of output png, label_list: list of labels for plot [title, x_axis, y_axis], color_map: color map used, default viridis, color_min: used for darkening colors. 
    comb=np.vstack([x, y])
    try:
        z=stats.gaussian_kde(comb)(comb)
    except:
        z=np.array([0.5 for i in range(len(x))])
    idx=z.argsort()
    x_sort, y_sort, z_sort = x[idx], y[idx], z[idx]
    z_sort=(1-color_min)*z_sort+color_min
    colors=cm.get_cmap(color_map)
    plt.scatter(x, y, c=colors(z_sort))
    if label_list[0]:
        plt.title(label_list[0])
        plt.xlabel(label_list[1])
        plt.ylabel(label_list[2])
    else:
        pass
    if save_it:
        plt.savefig("{}.pdf".format(name), bbox_inches='tight')
    else:
        pass
    if close_it:
        plt.close()
    else:
        pass


if __name__ == "__main__":
   main(sys.argv[1:])
