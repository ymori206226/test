"""
#######################
#        quket        #
#######################

mpilib.py

Initiating MPI and setting relevant arguments.

"""
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if rank == 0:
    main_rank = 1
else:
    main_rank = 0


def myrange(ndim):
    """
        Calculate process-dependent range for MPI distribution of `ndim` range.
        Image
        ----------
        Process 0        :       0         ---     ndim/nprocs
        Process 1        :   ndim/procs+1  ---   2*ndim/nprocs
        Process 2        : 2*ndim/procs+1  ---   3*ndim/nprocs
          ...
        Process i        :     ipos        ---   ipos + my_ndim
          ...
        Process nprocs-1 :      ...        ---      ndim-1

        Returns `ipos` and `my_ndim`

    Author(s): Takashi Tsuchimochi
    """
    nrem = ndim % nprocs
    nblk = int((ndim - nrem) / nprocs)
    if rank < nrem:
        my_ndim = nblk + 1
        ipos = my_ndim * rank
    else:
        my_ndim = nblk
        ipos = my_ndim * rank + nrem
    return ipos, my_ndim
