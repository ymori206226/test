"""
#######################
#        quket        #
#######################

fileio.py

File reading/writing utilities.

"""
import os
import datetime

import numpy as np

from . import config as cf
from . import mpilib as mpi


def prints(*args, filepath=cf.log, opentype="a", end=None):
    """Function:
        Print wrapper.
        args     : arguments to be printed
        filepath : file to be printed (default = log file)
                   if None, print on command-line
        opentype : 'a' = add (default)
                 : 'w' = create the file, overwrite
        end      : if end=None, break at the end of priting
                   if end="",   don't break

    Author(s): Takashi Tsuchimochi
    """
    if mpi.main_rank:
        if filepath is None:
            print(*args, end=end)
        else:
            with open(filepath, opentype) as f:
                print(*args, file=f, end=end)


def print_geom(geometry):
    """Function:
    Print geometry in the cartesian coordinates.

    Author(s): Takashi Tsuchimochi
    """
    prints("\n*** Geometry **************************")
    for iatom in range(len(geometry)):
        prints(f"  {geometry[iatom][0]:2s}  "
               f"  {geometry[iatom][1][0]:8.4f}  "
               f"  {geometry[iatom][1][1]:8.4f}  "
               f"  {geometry[iatom][1][2]:8.4f}")
    prints("***************************************\n")


def openfermion_print_state(state, n_qubits, j_state,
                            threshold=1e-2, digit=4, filepath=cf.log):
    """Function
    print out jth wave function in state

    Author(s): Takashi Tsuchimochi
    """
    opt = f"0{n_qubits}b"
    qubit_len = n_qubits + 4 - len("Basis")
    coef_len = 2*digit + 8
    prints(" "*(qubit_len//2), "Basis",
           " "*(qubit_len//2 + coef_len//2 - 1), "Coef")
    for i in range(2**n_qubits):
        v = state[i][j_state]
        if abs(v)**2 > threshold:
            formstr = "{a.real:+." + str(digit) + "f} " \
                    + "{a.imag:+." + str(digit) + "f}i"
            prints("|", format(i, opt), "> :", formstr.format(a=v),
                   filepath=filepath)


#def SaveTheta(ndim, theta, filepath, opentype="w"):
#    """Function
#    Save theta(0:ndim-1) to filepath (overwritten)
#
#    Author(s): Takashi Tsuchimochi
#    """
#    if mpi.main_rank:
#        with open(filepath, opentype) as f:
#            for i in range(ndim):
#                print(theta[i], file=f)
#
#
#def LoadTheta(ndim, filepath):
#    """Function
#    Read theta(0:ndim-1) from filepath
#
#    Author(s): Takashi Tsuchimochi
#    """
#    if os.path.isfile(filepath):
#        f = open(filepath)
#        line = f.readlines()
#        f.close
#        if len(line) != ndim:
#            error("File length incorrect: {}".format(filepath))
#        if mpi.main_rank:
#            theta = []
#            for i in range(ndim):
#                theta.append(float(line[i]))
#        else:
#            theta = None
#    else:
#        error("No theta file! ")
#
#    theta = mpi.comm.bcast(theta, root=0)
#    return theta


def SaveTheta(ndim, save, filepath):
    if mpi.main_rank:
        if save.size != ndim:
            error(f"{save.size=} but {ndim=}")
        np.savetxt(filepath, save.reshape(-1, 1))


def LoadTheta(ndim, filepath):
    load = np.loadtxt(filepath).reshape(-1)
    if load.size != ndim:
        error(f"{load.size=} but {ndim=}")
    load = mpi.comm.bcast(load, root=0)
    return load


def error(*message):
    prints("\n", *message, "\n")
    prints("Error termination of quket.")
    prints(datetime.datetime.now())
    exit()


def print_state(state, n_qubits=None, filepath=cf.log,
                threshold=0.01, name=None, digit=4):
    """Function
    print out quantum state as qubits

    Author(s): Takashi Tsuchimochi
    """
    if isinstance(name, str):
        prints(name)
    if n_qubits is None:
        n_qubits = state.get_qubit_count()

    opt = f"0{n_qubits}b"
    qubit_len = n_qubits + 4 - len("Basis")
    coef_len = 2*digit + 8
    prints(" "*(qubit_len//2), "Basis",
           " "*(qubit_len//2 + coef_len//2 - 1), "Coef")
    for i in range(2**n_qubits):
        v = state.get_vector()[i]
        if abs(v)**2 > threshold:
            formstr = "{a.real:+." + str(digit) + "f} " \
                    + "{a.imag:+." + str(digit) + "f}i"
            prints("|", format(i, opt), "> :", formstr.format(a=v),
                   filepath=filepath)


def print_amplitudes(theta_list, noa, nob, nva, nvb,
                     threshold=1e-2, filepath=cf.log):
    """Function
    Print out amplitudes of CCSD

    Author(s): Takashi Tsuchimochi
    """
    prints("")
    prints("----Amplitudes----")
    ### print singles amplitudes ###
    ia = 0
    for a in range(nva):
        aa = a + 1 + noa
        for i in range(noa):
            ii = i + 1
            if abs(theta_list[ia]) > threshold:
                prints(f"{ii}a -> {aa}a : {theta_list[ia]:2.10f}",
                       filepath=filepath)
            ia += 1
    for a in range(nvb):
        aa = a + 1 + nob
        for i in range(nob):
            ii = i + 1
            if abs(theta_list[ia]) > threshold:
                prints(f"{ii}b -> {aa}b : {theta_list[ia]:2.10f}",
                       filepath=filepath)
            ia += 1
    ### print doubles amplitudes ###
    ijab = ia
    for b in range(nva):
        bb = b + 1 + noa
        for a in range(b):
            aa = a + 1 + noa
            for j in range(noa):
                jj = j + 1
                for i in range(j):
                    ii = i + 1
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}a -> {aa}a {bb}a : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1

    ### ab -> ab ###
    for b in range(nvb):
        bb = b + 1 + nob
        for a in range(min(b+1, nva)):
            aa = a + 1 + noa
            for j in range(nob):
                jj = j + 1
                for i in range(j+1):
                    ii = i + 1
                    # b > a, j > i
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1
                for i in range(j+1, noa):
                    ii = i + 1
                    # b > a, i > j
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1
        for a in range(min(b+1, nva), nva):
            aa = a + 1 + noa
            for j in range(nob):
                jj = j + 1
                for i in range(j+1):
                    ii = i + 1
                    # a > b, j > i
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1
                for i in range(j+1, noa):
                    ii = i + 1
                    # a > b, i > j
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1

    ### bb -> bb ###
    for b in range(nvb):
        bb = b + 1 + nob
        for a in range(b):
            aa = a + 1 + nob
            for j in range(nob):
                jj = j + 1
                for i in range(j):
                    ii = i + 1
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}b {jj}b -> {aa}b {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1

    prints("------------------")


def print_amplitudes_spinfree(theta_list, no, nv,
                              threshold=0.01, filepath=cf.log):
    """Function:
    Print out amplitudes of spin-free CCSD

    Author(s): Takashi Tsuchimochi
    """
    from .ucclib import get_baji

    prints("")
    prints("----Amplitudes----")
    ### print singles amplitudes ###
    ia = 0
    for a in range(nv):
        aa = a + 1 + no
        for i in range(no):
            ii = i + 1
            if abs(theta_list[ia]) > threshold:
                prints(f"{ii} -> {aa} : {theta_list[ia]}", filepath=filepath)
            ia += 1

    ### print doubles amplitudes ###
    for b in range(nv):
        bb = b + 1 + no
        for a in range(b):
            aa = a + 1 + no
            for j in range(no):
                jj = j + 1
                for i in range(j):
                    ii = i + 1
                    baji = get_baji(b, a, j, i, no) + ia
                    abji = get_baji(a, b, j, i, no) + ia
                    theta = theta_list[baji] + theta_list[abji]
                    if abs(theta) > threshold:
                        prints(f"{ii}a {jj}a -> {aa}a {bb}a : {theta:2.10f}",
                               filepath=filepath)
                        prints(f"{ii}b {jj}b -> {aa}b {bb}b : {theta:2.10f}",
                               filepath=filepath)

    ### ab -> ab ###
    for b in range(nv):
        bb = b + 1 + no
        for a in range(min(b+1, nv)):
            aa = a + 1 + no
            for j in range(no):
                jj = j + 1
                for i in range(j+1):
                    ii = i + 1
                    # b > a, j > i
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[baji]}",
                               filepath=filepath)
                for i in range(j+1, no):
                    ii = i + 1
                    # b > a, i > j
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[baji]:2.10f}",
                               filepath=filepath)
        for a in range(b+1, nv):
            aa = a + 1 + no
            for j in range(no):
                jj = j + 1
                for i in range(j+1):
                    ii = i + 1
                    # a > b, j > i
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[baji]:2.10f}",
                               filepath=filepath)
                for i in range(j+1, no):
                    ii = i + 1
                    # a > b, i > j
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[baji]:2.10f}",
                               filepath=filepath)

    prints("------------------")


def printmat(A, mmax=10, filepath=cf.log, name=None, n=None, m=None):
    """Function:
    Print out A in a readable format.

        A         :  1D or 2D numpy array of dimension
        filepath  :  file to be printed
        mmax      :  maxixmum number of columns to print for each block
        name      :  Name to be printed
        n,m       :  Need to be specified if A is a matrix,
                     but loaded as a 1D array

    Author(s): Takashi Tsuchimochi
    """
    if isinstance(A, list):
        dimension = 1
    elif isinstance(A, np.ndarray):
        dimension = A.ndim
    if dimension == 0 or dimension > 2:
        error("Neither scalar nor tensor is printable with printmat.")

    prints("", filepath=filepath)
    if name is not None:
        prints(name, filepath=filepath)

    if dimension == 2:
        n, m = A.shape
        imax = 0
        while imax < m:
            imin = imax + 1
            imax = imax + mmax
            if imax > m:
                imax = m
            prints(" ", filepath=filepath)
            prints("           ", end="", filepath=filepath)
            for i in range(imin-1, imax):
                prints(f"  {i:4d}          ", end="", filepath=filepath)
            prints("", filepath=filepath)
            for j in range(n):
                prints(f" {j:4d}  ", end="", filepath=filepath)
                for i in range(imin-1, imax):
                    prints(f"  {A[j][i]:12.7f}  ", end="", filepath=filepath)
                prints("", filepath=filepath)
    elif dimension == 1:
        if n is None or m is None:
            if isinstance(A, list):
                n = len(A)
                m = 1
            elif isinstance(A, np.ndarray):
                n = A.size
                m = 1
        imax = 0
        while imax < m:
            imin = imax + 1
            imax = imax + mmax
            if imax > m:
                imax = m
            prints(" ", filepath=filepath)
            prints("           ", end="", filepath=filepath)
            for i in range(imin-1, imax):
                prints(f"  {i:4d}          ", end="", filepath=filepath)
            prints("", filepath=filepath)
            for j in range(n):
                prints(f" {j:4d}  ", end="", filepath=filepath)
                for i in range(imin-1, imax):
                    prints(f"  {A[j + i*n]:12.7f}  ", end="", filepath=filepath)
                prints("", filepath=filepath)
