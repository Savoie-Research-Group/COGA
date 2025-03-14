# Process data from a restarted dilatometry run. Concatenates files and runs CG_slowmelt.py. Easier than retyping every time.
touch thermo_comb.avg
cat thermo.avg >> thermo_comb.avg
cat thermo2.avg >> thermo_comb.avg
python ~/bin/CG_Crystal/CG_slowmelt.py thermo_comb.avg -map y_extend.map -order "0-0-8" -T_end 500 -cycles 10