import argparse, sys

def main(argv):
    parser = argparse.ArgumentParser(description='String multiple trajectories')
    
    parser.add_argument('names', type=str, help='Space delimited string of trajectory names')
    parser.add_argument('-write_time', dest='wt', type=int, default=1, help='Check timesteps and string the files as continuous trajectory')
    parser.add_argument('-o', dest='o', type=str, default='combined.lammpstrj', help='Output file name')
    parser.add_argument('-every', dest='every', type=int, default=1, help='Write every this many frames')
    parser.add_argument('-start', dest='start', type=int, default=0, help='Write from this frame')
    parser.add_argument('-end', dest='end', type=int, default=1000000000000, help='Write till this frame')
    parser.add_argument('--count_frame', dest='count_frame', default=False, action='store_const', const=True, help = 'When present, the script counts and returns the number of frames found')

    args = parser.parse_args()
    
    times=[]
    flag='n'
    names = args.names.split(' ')

    # fo=open(args.o, 'w')
    # for i, name in enumerate(names):
    #     print 'On file {}...\n'.format(name)
    #     f= open(name,'r')
    #     for j, line in enumerate(f):
    #         if flag=='t':
    #             if i==0:
    #                 times.append(int(line))
    #             else:
    #                 line=str(times[-1]+times[1]-times[0])+'\n'
    #                 times.append(times[-1]+times[1]-times[0])
    #             flag='n'
    #         if flag=='nt':
    #             if int(line)!=times[-1]:
    #                 line='TIMESTEP\n'+str(times[-1]+times[1]-times[0])+'\n'
    #                 times.append(times[-1]+times[1]-times[0])
    #                 flag='n'                    
    #         if 'TIMESTEP' in line:
    #             if i and j==0:
    #                 flag='nt'
    #             else:
    #                 flag='t'
    #         if flag !='nt':
    #             fo.write(line)

    #     f.close()

    # fo.close()


    write='n'
    times=[]
    lines=[]
    fo=open(args.o, 'w')
    for i, name in enumerate(names):
        print('On file {}...\n'.format(name))
        f= open(name,'r')
        for j, line in enumerate(f):
            if 'TIMESTEP' in line:
                if write=='y':                                        
                    if len(times) > args.end:
                        break
                    if len(times) > args.start and (len(times)-1) % args.every==0:
                        for l in lines:
                            fo.write(l)
                lines=[line]
                flag='t'
                continue
            if flag=='t':
                if not i:
                    times.append(int(line))
                    write='y'
                else:
                    if int(line)==times[-1]:
                        write='n'
                    else:
                        line=str(times[-1]+times[1]-times[0])+'\n'
                        times.append(times[-1]+times[1]-times[0])
                        write='y'
                lines.append(line)
                flag='nt'
                continue
            lines.append(line)
        f.close()
    if write=='y':
        if times[-1] >= args.start and times[-1] % args.every==0 and times[-1]<= args.end:            
            for l in lines:
                fo.write(l)

    fo.close()
    if args.count_frame:
        frame_count=len(times)
        print("frames in function", frame_count)
        with open("traj_num.txt", 'w') as f:
            f.write(str(frame_count)+' '+ str(times[0])+' '+str(times[-1]))
            f.close()



if __name__ == '__main__':
    main(sys.argv[1:])


