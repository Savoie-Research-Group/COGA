#!/bin/env python                                                                                                                                                             
import sys,os,subprocess,argparse,time
from subprocess import PIPE

class monitor_jobs() :
      
   def qstat(self):

        
        name_width = int(100)
        user = str(self.user)


        # redirect a qstat call into output
        command = "squeue --format ".split()
        command.append("\"%i %t %u\"")
        p = subprocess.Popen(command,stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        output=str(output)
        # Initialize job information dictionary
        job_dict = {}
        jobs = []
        for count_i,i in enumerate(output.split(r'\n')):
                if count_i==0: continue #skip first line(title)
                tmp = i.split("\"")
                if len(tmp) < 3: continue
                fields = tmp[1].split()
                if len(fields) == 0: continue
                current_key = fields[0]
                job_dict[current_key] = { "State":"NA" , "User":"NA"}
                job_dict[current_key]["State"] = fields[1]
                job_dict[current_key]["User"] = fields[2]
        self.job_dict = job_dict
        #template_string="{:<40s} {:<"+str(name_width)+"s} {:<20s} {:20s} {:<20s}"
        #print (template_string.format("ID","Name","User","Walltime","Status","Queue"))
        for i in job_dict.keys():
                if job_dict[i]["State"] in ["R","PD","CG"] and (user == "" or job_dict[i]["User"] == user):
                   jobs += [i]
                   #print (template_string.format(i,dict[i]["Name"],dict[i]["User"],dict[i]["Walltime"],dict[i]["State"],dict[i]["Queue"]))

        return jobs

   def main(self):
      current_jobs = self.qstat()
      if len(self.jobids) > 0:
         print("waiting for jobs {} to complete....".format(self.jobids))
      while True in [ i in current_jobs for i in self.jobids ]:
        time.sleep(10)
        current_jobs = self.qstat()  
      return

   def __init__(self,jobids,user):
      self.job_dict = {}
      self.jobids = jobids
      self.user = user
      self.main()


if __name__ == "__main__":
   user = os.getlogin()
   monitor_jobs(sys.argv[1:],user) 
