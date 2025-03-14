import argparse, sys

def main(argv):
    parser = argparse.ArgumentParser(description='Reads gromacs .itp file parameters and converts it into lammps real units for the pair types user provides')
    
    parser.add_argument('ifile', type=str, help='Input gromacs .itp file')
    parser.add_argument('-pair_types', dest='pair_types', type=str, default='', help='Space delimited string of comma delimited string for atom-types of each pair. Default: converts [arameters it for all pairs found')
    parser.add_argument('-ofile', dest='ofile', type=str, default='lammps_para.txt', help='Output file name')

    args = parser.parse_args()
    if len(args.pair_types): args.pair_types=[_.split(',') for _ in args.pair_types.split()]
    
    para={}
    fi=open(args.ifile, 'r')
    flag='y'
    for line in fi:
        if line.strip()=='[ nonbond_params ]':
            flag='y'
            continue
        if line[0]=='[':
            flag='n'
            continue
        if line.strip() !='' and ';' != line[0] and flag=='y':
            l=line.split()
            if args.pair_types!='' and [l[0],l[1]] not in args.pair_types and [l[1],l[0]] not in args.pair_types:
                continue
            l[3],l[4]=float(l[3]),float(l[4])
            if l[3]==0 and l[4]==0:
                sgm, eps=0,0
            else:
                #sgm=(l[4]/l[3])**(1.0/6.0)
                #eps=l[3]/(4.0*(sgm**6.0))
                sgm=l[3]*10.0
                eps=l[4]/4.184
            if [l[0],l[1]] in args.pair_types:
                para[(l[0],l[1])]=[eps,sgm]
            if [l[1],l[0]] in args.pair_types:
                para[(l[1],l[0])]=[eps,sgm] 
            

    fi.close()
    fo=open(args.ofile,'w')
    for i in args.pair_types:
        if tuple(i) not in para.keys():
            fo.write('Error! Key {} not found. Exiting...'.format(i))
            quit()
        fo.write('{:6s}\t{:6s}\t{:2.2f}\t{:2.2f}\n'.format(i[0],i[1],para[tuple(i)][0],para[tuple(i)][1]))


if __name__ == '__main__':
    main(sys.argv[1:])



