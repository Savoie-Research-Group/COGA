#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import warnings
from scipy import stats
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting", "/Job_Submission"]
for subf in subfolders:
    sys.path.append(script_directory+subf)
from pathlib import Path
home=str(Path.home())

def main(argv):
    parser = argparse.ArgumentParser(description='''Calculates and plot rdfs for crystal structures                                                                                                                                           
    Input: traj files, data files, name of two compared groups, name of output file.
                                                                                                                                                                                                      
    Output: rdf files, plots of rdfs.                                                                                                                                                                                           
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('traj', type=str, help='String of trajectory files')
    parser.add_argument('data', type=str, help='String of data files')
    parser.add_argument('groups', type=str, help='string of group types. Submit "all #" where "#" is the number of groups to run rdfs on all same-id pairs')
    #parser.add_argument('ref_file', type=str, help='name of reference data file to be read.')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='crys_rdf', help='Output. Default: ')
    parser.add_argument('-species', dest='species', type=str, default='Molecule', help='Name of species to be put on chart title Default: NDI2TEMPO ')
    parser.add_argument('-reuse_rdf', dest='reuse_rdf', type=str, default=0, help='Set to 0 if you want new rdfs. Provide string of rdf names for rdfs you want to keep (or 0 if you want to redo)')
    parser.add_argument('-bond_sep', dest='bond_sep', type=int, default=-1, help='Bond_sep in rdf.')
    parser.add_argument('-sr', dest='sr', type=float, default=9999999.0, help='Smooth Radius. Radius about which rdf will be introduced Gaussian error. If not specified, rdf will not be smoothed.')
    parser.add_argument('-gaus_std', dest='gaus_std', type=float, default=1.0, help='Standard deviation used in gaussian calculation.')
    #parser.add_argument('--smooth', dest='smooth', default=False, action='store_const', const=True,
    #                    help = 'When present, the calculated and plotted rdfs will be smoothed. (default: False)')
        
    # Parse Arguments
    args = parser.parse_args()
    
    # Write Script Below
    traj=args.traj.split()
    data=args.data.split()
    data_name=data[0].split('.')[0]
    if len(traj)!= len(data):
        print("ERROR: provide the same number of traj and data files")
        quit()
    groups=args.groups.split()
    if len(groups)!=2:
        print("ERROR: please provide two groups to compute rdfs over.")
    if groups[0]=="all": #for iterating over all beads.
        for g in range(1, int(groups[1])+1):
            make_rdf(traj, data, [g, g], args.output, data_name, args.reuse_rdf, args.species, args.sr, args.bond_sep, args.gaus_std)
    else:
        make_rdf(traj, data, groups, args.output, data_name, args.reuse_rdf, args.species, args.sr, args.bond_sep, args.gaus_std)

def make_rdf(traj, data, groups, output, data_name, reuse_rdf, species, sr, bond_sep, gaus_std=1.0):
    rdf_data=[[] for i in range(len(traj))] # [[],[]] # Making this a single list that is reset each time to conserve
    dump_var=open("dump.txt",'w')
    # Set up reusing information based on what's passed.
    if type(reuse_rdf)==str:
        reuse_info=reuse_rdf.split()
    else:
        reuse_info=[reuse_rdf for i in range(len(traj))]
    for i in range(len(traj)):
        if str(reuse_info[i])=='0':
            sp.call('python {}/Analysis/rdf.py {} {} {} {} type type -o {}_{} -r_max 12.0 -bond_sep {} >> rdf{}{}.log'.format(script_directory, traj[i],data[i],groups[0],groups[1],output,i, bond_sep,groups[0],groups[1]),shell=True) #, stdout=dump_var) 
            rdf_data[i]=np.loadtxt('{}_{}_{}_{}.rdf'.format(output,i,groups[0],groups[1]),skiprows=1)
        else: # Read rdf data straight from provided output file.
            rdf_data[i]=np.loadtxt('{}_{}_{}.rdf'.format(reuse_info[i],groups[0],groups[1]), skiprows=1)
        if sr!=9999999.0:
            rdf_data[i]=rdf_smoother(rdf_data[i],sr,gaus_std, i_val=i, group=groups[0])
            #rdf_data[i][:,1]=grdf[:,1]
            #print(rdf_data)
        #plt.figure(num=i)
        #plt.plot(rdf_data[:,0],rdf_data[:,1])
        #plt.savefig("{}_{}_{}_{}.png".format(args.output,i,groups[0],groups[1]))
    fig, ax1=plt.subplots()
    plot_1 = ax1.plot(rdf_data[0][:,0],rdf_data[0][:,1], color='blue', label='Model')
    ax1.set_xlabel('Distance (Ang)')
    ax1.set_ylabel('RDF Value')
    ax2=ax1.twinx()
    plot_2 = ax2.plot(rdf_data[1][:,0],rdf_data[1][:,1], color='green',label='Reference')
    lns = plot_1+plot_2
    labels=[l.get_label() for l in lns]
    plt.legend(lns,labels,loc=0)
    plt.title(species+": Beads {},{}".format(groups[0],groups[1]))
    plt.savefig("{}_{}_{}_{}_all.pdf".format(output,data_name,groups[0],groups[1]))
    #psuedo_rdf(args.ref_file,groups)
    del(rdf_data) # Delete the rdf data after to conserve memory.

def rdf_smoother(rdf, sr, gaus_std, i_val='', group=''): # Standardized function by which rdfs can be smoothed.
    grdf=np.zeros((len(rdf),2))
    grdf[:,0]=rdf[:,0]
    yes=0
    no=0
    err_count=0
    for q in range(len(rdf)):
        if rdf[q,1]==0.0:
            pass
        else:
            srind=int(sr*100) # Convert smoothing distance into number of indices
            if sr == 0:
                xlow=0
                xhigh=len(rdf)
                #print("Gaussian will cover full rdf length")
            else:
                xlow=max(q-srind, 0)
                xhigh=min(q+srind+1, len(rdf))
            xvals=grdf[xlow:xhigh,0]
            warnings.filterwarnings("error", category=RuntimeWarning)                #### Catch warnings as if an error. This should allow try/except to trigger with RunTimeWarning, allowing script to continue and warning causes to be probed.
            gaus=stats.norm.pdf(xvals, rdf[q,0], gaus_std)
            if any(gaus<0):
                print("There is a negative value in the gaussian!")
            try:
                scale=rdf[q,1]/gaus.sum()
            except Exception as e:
                err_count+=1
                if q-srind<0 or q+srind>len(rdf):
                    yes+=1
                else:
                    no+=1
                print("ERROR in RDF Calc:\n")
                print(e)
                print("gaus_sum: ", gaus.sum())
                print("gaus: ", gaus)
                print("xvals: ", xvals)
                print("current xval: ", rdf[q,0])
                print("current yval: ", rdf[q,1], type(rdf[q,1]))
                print("q: ", q)
                print("grdf range: ", grdf[q-srind:q+srind+1,0])
                print(q-srind,q+srind)
                print("rdf range: ", rdf[q-srind:q+srind+1,0])
                print("rdf length: ", len(rdf))
                print("srind: ", srind)
                print("group: ", group)
                print("i_val: ", i_val)
                if group and i_val:
                    print("faulty rdf name: {}_{}_{}".format(i_val,group,group))

                scale=1.0
                    #if np.isinf(scale)==True:
                    #    print("product: ", np.isnan(gaus*scale).any())
                    #    print("scale: ", scale)
                    #    print(rdf_data[i][q,1],"/",gaus.sum())
                    #    print("gaus: ", gaus)
                    #    print("xvals: ", xvals)
                    #print("before: ", grdf[q-100:q+101,1])
            grdf[xlow:xhigh,1]=grdf[xlow:xhigh,1]+gaus*scale
                    #print("after: ", grdf[q-100:q+101,1])
                    
                    #print('grdf: ', np.isnan(grdf[:,1]).any())
    if err_count:
        print("yes, no, err_count", yes, no, err_count)
    return grdf



#def psuedo_rdf(ref_data,groups):
#    prdf_data=np.empty(0)
#    aflag=False
#    with open(ref_data, 'r') as f:
#        for lines in f:
#            if lines.split()==[]:
#                pass
#            elif lines.split()[0]=="Atoms":
#                aflag=True
#            elif lines.split()[0]=="Bonds":
#                break
#            elif aflag==True:
#                if int(float(lines.split()[2]))==int(groups[0]):
#                    prdf_data=np.append(prdf_data,[float(lines.split()[4]),float(lines.split()[5]),float(lines.split()[6])])
#                else:
#                    pass
#    prdf_data=np.reshape(prdf_data, (-1, 3))
#    print(prdf_data)     
#    prdf_vals=np.empty(0)
#    #for i in len()   

# This is the old version of make_rdf. It creates a vector of all rdf_data, but this has proven to be very memory intensive.        
def make_rdf_old(traj, data, groups, output, data_name, reuse_rdf, species, sr, bond_sep, gaus_std=1.0):
    rdf_data=[[] for i in range(len(traj))] # [[],[]]
    dump_var=open("dump.txt",'w')
    # Set up reusing information based on what's passed.
    if type(reuse_rdf)==str:
        reuse_info=reuse_rdf.split()
    else:
        reuse_info=[reuse_rdf for i in range(len(traj))]
    for i in range(len(traj)):
        if str(reuse_info[i])=='0':
            sp.call('python {}/Analysis/rdf.py {} {} {} {} type type -o {}_{} -r_max 15.0 -bond_sep {} >> rdf{}{}.log'.format(script_directory, traj[i],data[i],groups[0],groups[1],output,i, bond_sep,groups[0],groups[1]),shell=True) #, stdout=dump_var) 
            rdf_data[i]=np.loadtxt('{}_{}_{}_{}.rdf'.format(output,i,groups[0],groups[1]),skiprows=1)
        else: # Read rdf data straight from provided output file.
            rdf_data[i]=np.loadtxt('{}_{}_{}.rdf'.format(reuse_info[i],groups[0],groups[1]), skiprows=1)
        if sr!=9999999.0:
            rdf_data[i]=rdf_smoother(rdf_data[i],sr,gaus_std, i_val=i, group=groups[0])
            #rdf_data[i][:,1]=grdf[:,1]
            #print(rdf_data)
        #plt.figure(num=i)
        #plt.plot(rdf_data[:,0],rdf_data[:,1])
        #plt.savefig("{}_{}_{}_{}.png".format(args.output,i,groups[0],groups[1]))
    fig, ax1=plt.subplots()
    plot_1 = ax1.plot(rdf_data[0][:,0],rdf_data[0][:,1], color='blue', label='Model')
    ax1.set_xlabel('Distance (Ang)')
    ax1.set_ylabel('RDF Value')
    ax2=ax1.twinx()
    plot_2 = ax2.plot(rdf_data[1][:,0],rdf_data[1][:,1], color='green',label='Reference')
    lns = plot_1+plot_2
    labels=[l.get_label() for l in lns]
    plt.legend(lns,labels,loc=0)
    plt.title(species+": Beads {},{}".format(groups[0],groups[1]))
    plt.savefig("{}_{}_{}_{}_all.pdf".format(output,data_name,groups[0],groups[1]))
    #psuedo_rdf(args.ref_file,groups)


    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])