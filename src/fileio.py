"""
#######################
#        quket        #
#######################

fileio.py

File reading/writing utilities.

"""
import os
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
            print("".join(map(str, args)), end=end)
        else:
            with open(filepath, opentype) as f:
                print("".join(map(str, args)), file=f, end=end)


def print_geom(geometry):
    """Function:
        Print geometry in the cartesian coordinates.

    Author(s): Takashi Tsuchimochi
    """
    prints("\n *** Geometry **************************")
    for iatom in range(cf.natom):
        prints(
            "  {:2s}    {:8.4f}   {:8.4f}   {:8.4f}".format(
                geometry[iatom][0],
                geometry[iatom][1][0],
                geometry[iatom][1][1],
                geometry[iatom][1][2],
            )
        )
    prints(" ***************************************\n")


def openfermion_print_state(state, n_qubit, j_state):
    """Function
    print out jth wave function in state

    Author(s): Takashi Tsuchimochi
    """
    opt = "0" + str(n_qubit) + "b"
    for i in range(2 ** n_qubit):
        v = state[i][j_state]
        if abs(v) ** 2 > 0.01:
            prints(
                "|", format(i, opt), "> : ", "{a.real:+.4f} {a.imag:+.4f}i".format(a=v)
            )


def SaveTheta(ndim, theta, filepath, opentype="w"):
    """Function
    Save theta(0:ndim-1) to filepath (overwritten)

    Author(s): Takashi Tsuchimochi
    """
    if mpi.main_rank:
        with open(filepath, opentype) as f:
            for i in range(ndim):
                print(theta[i], file=f)


def LoadTheta(ndim, filepath):
    """Function
    Read theta(0:ndim-1) from filepath

    Author(s): Takashi Tsuchimochi
    """
    if os.path.isfile(filepath):
        f = open(filepath)
        line = f.readlines()
        f.close
        if len(line) != ndim:
            error("File length incorrect: {}".format(filepath))
        if mpi.main_rank:
            theta = []
            for i in range(ndim):
                theta.append(float(line[i]))
        else:
            theta = None
    else:
        error("No theta file! ")

    theta = mpi.comm.bcast(theta, root=0)
    return theta


def error(*message):
    import datetime

    prints("\n", *message, "\n")
    prints("Error termination of quket.")
    prints(datetime.datetime.now())
    exit()


def print_state(state, n_qubit=None, filepath=cf.log, threshold=1e-2, name=None):
    """Function
    print out quantum state as qubits

    Author(s): Takashi Tsuchimochi
    """
    if type(name) == str:
        prints(name)
    if n_qubit == None:
        n_qubit = state.get_qubit_count()
    opt = "0" + str(n_qubit) + "b"
    prints(" Basis       Coef", filepath=filepath)
    for i in range(2 ** n_qubit):
        v = state.get_vector()[i]
        if abs(v) ** 2 > threshold:
            prints(
                "|",
                format(i, opt),
                "> : ",
                "{a.real:+.4f} {a.imag:+.4f}j".format(a=v),
                filepath=filepath,
            )


def print_amplitudes(theta_list, noa, nob, nva, nvb, threshold=0.01, filepath=cf.log):
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
                prints(
                    ii,
                    "a -> ",
                    aa,
                    "a  : ",
                    "%2.10f" % theta_list[ia],
                    filepath=filepath,
                )
            ia = ia + 1
    for a in range(nvb):
        aa = a + 1 + nob
        for i in range(nob):
            ii = i + 1
            if abs(theta_list[ia]) > threshold:
                prints(
                    ii,
                    "b -> ",
                    aa,
                    "b  : ",
                    "%2.10f" % theta_list[ia],
                    filepath=filepath,
                )
            ia = ia + 1
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
                        prints(
                            ii,
                            "a",
                            jj,
                            "a -> ",
                            aa,
                            "a",
                            bb,
                            "a  : ",
                            "%2.10f" % theta_list[ijab],
                            filepath=filepath,
                        )
                    ijab = ijab + 1

    ### ab -> ab ###
    for b in range(nvb):
        bb = b + 1 + nob
        for a in range(min(b + 1, nva)):
            aa = a + 1 + noa
            for j in range(nob):
                jj = j + 1
                for i in range(j + 1):
                    ii = i + 1
                    # b > a, j > i
                    if abs(theta_list[ijab]) > threshold:
                        prints(
                            ii,
                            "a",
                            jj,
                            "b -> ",
                            aa,
                            "a",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[ijab]),
                            filepath=filepath,
                        )
                    ijab = ijab + 1
                for i in range(j + 1, noa):
                    ii = i + 1
                    # b > a, i > j
                    if abs(theta_list[ijab]) > threshold:
                        prints(
                            ii,
                            "a",
                            jj,
                            "b -> ",
                            aa,
                            "a",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[ijab]),
                            filepath=filepath,
                        )
                    ijab = ijab + 1
        for a in range(b + 1, nva):
            aa = a + 1 + noa
            for j in range(nob):
                jj = j + 1
                for i in range(j + 1):
                    ii = i + 1
                    # a > b, j > i
                    if abs(theta_list[ijab]) > threshold:
                        prints(
                            ii,
                            "a",
                            jj,
                            "b -> ",
                            aa,
                            "a",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[ijab]),
                            filepath=filepath,
                        )
                    ijab = ijab + 1
                for i in range(j + 1, noa):
                    ii = i + 1
                    # a > b, i > j
                    if abs(theta_list[ijab]) > threshold:
                        prints(
                            ii,
                            "a",
                            jj,
                            "b -> ",
                            aa,
                            "a",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[ijab]),
                            filepath=filepath,
                        )
                    ijab = ijab + 1
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
                        prints(
                            ii,
                            "b",
                            jj,
                            "b -> ",
                            aa,
                            "b",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[ijab]),
                            filepath=filepath,
                        )
                    ijab = ijab + 1

    prints("------------------")


def print_amplitudes_spinfree(theta_list, no, nv, threshold=0.01, filepath=cf.log):
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
                prints(
                    ii, " -> ", aa, "  : ", "%2.10f" % theta_list[ia], filepath=filepath
                )
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
                        prints(
                            ii,
                            "a",
                            jj,
                            "a -> ",
                            aa,
                            "a",
                            bb,
                            "a  : ",
                            "%2.10f" % theta,
                            filepath=filepath,
                        )
                        prints(
                            ii,
                            "b",
                            jj,
                            "b -> ",
                            aa,
                            "b",
                            bb,
                            "b  : ",
                            "%2.10f" % theta,
                            filepath=filepath,
                        )

    ### ab -> ab ###
    for b in range(nv):
        bb = b + 1 + no
        for a in range(min(b + 1, nv)):
            aa = a + 1 + no
            for j in range(no):
                jj = j + 1
                for i in range(j + 1):
                    ii = i + 1
                    # b > a, j > i
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(
                            ii,
                            "a",
                            jj,
                            "b -> ",
                            aa,
                            "a",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[baji]),
                            filepath=filepath,
                        )
                for i in range(j + 1, no):
                    ii = i + 1
                    # b > a, i > j
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(
                            ii,
                            "a",
                            jj,
                            "b -> ",
                            aa,
                            "a",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[baji]),
                            filepath=filepath,
                        )
        for a in range(b + 1, nv):
            aa = a + 1 + no
            for j in range(no):
                jj = j + 1
                for i in range(j + 1):
                    ii = i + 1
                    # a > b, j > i
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(
                            ii,
                            "a",
                            jj,
                            "b -> ",
                            aa,
                            "a",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[baji]),
                            filepath=filepath,
                        )
                for i in range(j + 1, no):
                    ii = i + 1
                    # a > b, i > j
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(
                            ii,
                            "a",
                            jj,
                            "b -> ",
                            aa,
                            "a",
                            bb,
                            "b  : ",
                            "%2.10f" % (theta_list[baji]),
                            filepath=filepath,
                        )

    prints("------------------")


def printmat(A, mmax=10, filepath=cf.log, name=None, n=None, m=None):
    """Function:
    Print out A in a readable format.

        A         :  1D or 2D numpy array of dimension
        filepath  :  file to be printed
        mmax      :  maxixmum number of columns to print for each block
        name      :  Name to be printed
        n,m       :  Need to be specified if A is a matrix, but loaded as a 1D array


    Author(s): Takashi Tsuchimochi
    """
    if type(A) is list:
        dimension = 1
    elif type(A) is np.ndarray:
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
            for i in range(imin - 1, imax):
                prints("  {I:4}          ".format(I=i), end="", filepath=filepath)
            prints("", filepath=filepath)
            for j in range(n):
                prints(" {J:4}  ".format(J=j), end="", filepath=filepath)
                for i in range(imin - 1, imax):
                    prints(
                        "  {v: 12.7f}  ".format(v=A[j][i]), end="", filepath=filepath
                    )
                prints("", filepath=filepath)
    elif dimension == 1:
        if n or m is None:
            if type(A) is list:
                n = len(A)
                m = 1
            elif type(A) is np.ndarray:
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
            for i in range(imin - 1, imax):
                prints("  {I:4}          ".format(I=i), end="", filepath=filepath)
            prints("", filepath=filepath)
            for j in range(n):
                prints(" {J:4}  ".format(J=j), end="", filepath=filepath)
                for i in range(imin - 1, imax):
                    prints(
                        "  {v: 12.7f}  ".format(v=A[j + i * n]),
                        end="",
                        filepath=filepath,
                    )
                prints("", filepath=filepath)
