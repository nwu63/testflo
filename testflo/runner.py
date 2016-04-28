"""
Methods and class for running tests.
"""

import os
import traceback

from six import advance_iterator
from multiprocessing import Queue, Process

from testflo.cover import setup_coverage, save_coverage
from testflo.profile import save_profile
import testflo.profile
from testflo.test import Test


def _run_single_test(test, q):
    test.run()
    q.put(test)

def run_isolated(test, q):
    """This runs the test in a subprocess,
    then puts the Test object on the queue.
    """

    p = Process(target=_run_single_test, args=(test, q))
    p.start()
    t = q.get()
    p.join()
    q.put(t)

def worker(test_queue, done_queue, worker_id):
    """This is used by concurrent test processes. It takes a test
    off of the test_queue, runs it, then puts the Test object
    on the done_queue.
    """

    # need a unique profile output file for each worker process
    testflo.profile._prof_file = 'profile_%s.out' % worker_id

    test_count = 0
    for test in iter(test_queue.get, 'STOP'):

        try:
            test_count += 1
            test.run(done_queue)
        except:
            # we generally shouldn't get here, but just in case,
            # handle it so that the main process doesn't hang at the
            # end when it tries to join all of the concurrent processes.
            done_queue.put(test)

    # don't save anything unless we actually ran a test
    if test_count > 0:
        save_coverage()
        save_profile()


class TestRunner(object):
    def __init__(self, options):
        self.stop = options.stop
        self.done_queue = Queue()

        setup_coverage(options)

    def get_iter(self, input_iter):
        """Run tests serially."""

        for test in input_iter:
            test.run(self.done_queue)
            result = self.done_queue.get()
            yield result
            if self.stop and result.status == 'FAIL':
                break

        save_coverage()


class ConcurrentTestRunner(TestRunner):
    """TestRunner that uses the multiprocessing package
    to execute tests concurrently.
    """

    def __init__(self, options):
        super(ConcurrentTestRunner, self).__init__(options)
        self.num_procs = options.num_procs

        # only do concurrent stuff if num_procs > 1
        if self.num_procs > 1:
            self.get_iter = self.run_concurrent_tests

            self.task_queue = Queue()
            self.procs = []

            # Start worker processes
            for i in range(self.num_procs):
                worker_id = "%d_%d" % (os.getpid(), i)
                self.procs.append(Process(target=worker,
                                          args=(self.task_queue,
                                                self.done_queue, worker_id)))

            for proc in self.procs:
                proc.start()

    def run_concurrent_tests(self, input_iter):
        """Run tests concurrently."""

        it = iter(input_iter)
        numtests = 0
        try:
            for proc in self.procs:
                self.task_queue.put(advance_iterator(it))
                numtests += 1
        except StopIteration:
            pass
        else:
            try:
                while numtests:
                    result = self.done_queue.get()
                    yield result
                    numtests -= 1
                    if self.stop and result.status == 'FAIL':
                        break
                    self.task_queue.put(advance_iterator(it))
                    numtests += 1
            except StopIteration:
                pass

        for proc in self.procs:
            self.task_queue.put('STOP')

        for i in range(numtests):
            yield self.done_queue.get()

        for proc in self.procs:
            proc.join()
