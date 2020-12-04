from mpi4py import MPI

comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()

if rank==0:
    main_rank = 1
else:
    main_rank = 0

if main_rank and nprocs > 1:
    print("mpi initialized")
