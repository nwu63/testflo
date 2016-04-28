"""
This is meant to be executed using mpirun.  It is called as a subprocess
to run an MPI test.

"""

from __future__ import print_function

exit_codes = {
    'OK': 0,
    'SKIP': 42,
    'FAIL': 43,
}

if __name__ == '__main__':
    import sys
    import traceback

    from mpi4py import MPI

    from testflo.cover import save_coverage


    exitcode = 0  # use 0 for exit code of all ranks != 0 because otherwise,
                  # MPI will terminate other processes

    try:
        parent_comm = MPI.Comm.Get_parent()
        test = parent_comm.bcast(None, root=0)
        test.run()
    except:
        print(traceback.format_exc())
        test.status = 'FAIL'
        test.err_msg = traceback.format_exc()
    finally:
        save_coverage()

        parent_comm.gather(test, root=0)

        sys.stdout.flush()
        sys.stderr.flush()

        print("CLIENT disconn")
        parent_comm.Disconnect()
        print("CLIENT disconn DONE for",MPI.COMM_WORLD.rank)
        sys.stdout.flush()

    sys.exit(exitcode)
