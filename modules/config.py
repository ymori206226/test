# Global arguments
import re
import sys
import os
import time

################################
# Setting for input and output #
################################
len_argv = len(sys.argv)
if len_argv == 1:
   print("Error! No input loaded.")
   exit()

# First argument = Input file,  either "***.inp" or "***" is allowed.
input_file=sys.argv[1]
input_name,ext = os.path.splitext( os.path.basename(input_file))
if ext == '':
    ext = '.inp'
input_file = input_name+ext

# Define the names of other useful files 
theta_list_file = './' + input_name + '.theta'
tmp  = './' + input_name + '.tmp'
kappa_list_file = './' + input_name + '.kappa'
chk = './' + input_name + '.chk'

# If second argument also exits, that will be your log file name
if len_argv == 3:
    log_name=sys.argv[2]
    log = './' + log_name
else:
    log_name = input_name
    log = './' + log_name + '.log'

PeriodicTable = ["H","He","Li","Be","B","C","N","O","F","Ne"]


#######################################
# Setting Jordan-Wigner Operators     #
#######################################
jw_s4 = 0

#######################################
# Setting other global parameters     #
# Default values                      #
#######################################
constraint_lambda = 0
t_old = 0
nthreads="1"
Do1PDM = 0
Do2PDM = 0
