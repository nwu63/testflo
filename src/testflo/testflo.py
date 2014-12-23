
import sys
import os
from types import FunctionType, MethodType
import inspect
import traceback
from cStringIO import StringIO
from argparse import ArgumentParser
from fnmatch import fnmatch
import unittest

from fileutil import find_files, get_module_path, find_module

def _get_parser():
    """Sets up the plugin arg parser and all of its subcommand parsers."""

    parser = ArgumentParser()
    parser.usage = "testease [options]"
    parser.add_argument('-c', '--config', action='store', dest='cfg',
                        metavar='CONFIG',
                        help='Path of config file where tests are specified.')
    parser.add_argument('tests', metavar='test', nargs='*',
                       help='a test method/case/module/directory to run')

    return parser


def _exclude_dir(dname):
    return dname in ('devenv', 'site-packages', 'dist-packages', 'contrib')

def read_config_file(cfgfile):
    with open(os.path.abspath(cfgfile), 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

class TestResult(object):
    """Contains the path to the test function/method, status
    of the test (if finished), and error and stdout messages (if any).
    """

    def __init__(self, testpath, status='OK', out_msg='', err_msg=''):
        self.testpath = testpath
        self.status = status
        self.out_msg = out_msg
        self.err_msg = err_msg

    def __str__(self):
        stream = StringIO()
        stream.write(self.testpath)
        stream.write(' ... ')
        stream.write(self.status)
        if self.status != 'OK':
            stream.write('\n')
            stream.write(self.err_msg)
        return stream.getvalue()


def get_module(fname):
    """Given a filename or module path name, return a tuple
    of the form (filename, module).
    """

    if fname.endswith('.py'):
        modpath = get_module_path(fname)
    else:
        modpath = fname
        fname = find_module(modpath)

    if fname is None:
        return None, None

    try:
        __import__(modpath)
    except ImportError:
        sys.path.append(os.path.dirname(fname))
        try:
            __import__(modpath)
        finally:
            sys.path.pop()

    return fname, sys.modules[modpath]

def get_testcase(filename, mod, tcasename):
    """Given a module and the name of a TestCase
    class, return a TestCase object or raise an exception.
    """

    try:
        tcase = getattr(mod, tcasename)
    except AttributeError:
        raise AttributeError("Couldn't find TestCase '%s' in module '%s'" %
                               (tcasename, filename))
    if issubclass(tcase, unittest.TestCase):
        return tcase
    else:
        raise TypeError("'%s' in file '%s' is not a TestCase." % 
                        (tcasename, filename))

def parse_test_path(testpath):
    """Return a tuple of the form (fname, module, testcase, func)
    based on the given testpath.

    The format of testpath is one of the following:
        <module>
        <module>:<testcase>
        <module>:<testcase>.<method>
        <module>:<function>

    where <module> is either the python module path or the actual
    file system path to the .py file.  A value of None in the tuple
    indicates that that part of the testpath was not present.
    """

    testpath = testpath.strip()
    module, _, rest = testpath.partition(':')
    fname, mod = get_module(module)
    testcase = method = None

    if rest:
        objname, _, method = rest.partition('.')
        obj = getattr(mod, objname)
        if inspect.isclass(obj) and issubclass(obj, unittest.TestCase):
            testcase = obj
            if method:
                method = getattr(obj, method)
        elif isinstance(obj, FunctionType):
            method = obj
        else:
            raise TypeError("'%s' is not a TestCase or a function" %
                            objname)

    return (fname, mod, testcase, method)


class ResultProcessor(object):
    """Processes the TestResult objects after tests have
    been run.
    """

    def __init__(self, test_results, stream=sys.stdout):
        self.results = test_results
        self.stream = stream

    def __iter__(self):
        return self.process_results()

    def process_results(self):
        for result in self.results:
            self.process(result)
            yield result

    def process(self, result):
        stream.write("%s ... %s\n" % (result.path, result.status))
        if result.err_msg:
            stream.write(result.err_msg)
            stream.write('\n')


class TestRunner(object):
    """Runs each test specified in results."""

    def __init__(self, tests):
        self.tests = tests

    def __iter__(self):
        return self.process_tests()

    def process_tests(self):
        for test in self.tests:
            yield self.run_test(test)

    def _try_call(self, method, outstream, errstream):
        outstr = errstr = ''
        status = 'OK'
        if method:
            try:
                old_err = sys.stderr
                old_out = sys.stdout
                sys.stdout = outstream
                sys.stderr = errstream
                method()
            except Exception as e:
                err = sys.exc_info()
                if isinstance(e, unittest.SkipTest):
                    status = 'SKIP'
                else:
                    status = 'FAIL'
                    msg = ''.join(traceback.format_exception(err[0],err[1],err[2]))
                    sys.stderr.write(msg)
            finally:
                sys.stderr = old_err
                sys.stdout = old_out

        return status

    def run_test(self, test):
        sys.stdout.write("%s ... " % test)
        sys.stdout.flush()

        fname, mod, testcase, method = parse_test_path(test)
        if testcase:
            testcase = testcase(methodName=method.__name__)
            parent = testcase
        else:
            parent = mod

        outstream = StringIO()
        errstream = StringIO()

        status = self._try_call(getattr(parent, 'setUp', None),
                                outstream, errstream)
        if status == 'OK':
            status = self._try_call(getattr(parent, method.__name__,
                                            None),
                                    outstream, errstream)
            tdstatus = self._try_call(getattr(parent, 'tearDown',
                                              None),
                                      outstream, errstream)
            if status == 'OK':
                status = tdstatus

            result = TestResult(test, status,
                                outstream.getvalue(), errstream.getvalue())
        else:
            result = TestResult(test, status,
                                outstream.getvalue(), errstream.getvalue())

        sys.stdout.write(status+'\n')
        return result


class TestDiscoverer(object):
    """An iterator that returns a testpath string
    for each test found in the specified
    directories/modules/testcases.
    """

    def __init__(self, tests,
                 module_pattern='test*.py',
                 func_pattern='test*'):
        self.tests = tests
        self.module_pattern = module_pattern
        self.func_pattern = func_pattern

    def __iter__(self):
        return self.process_test_strings()

    def process_test_strings(self):
        """Returns an iterator over the expanded testpath
        strings based on the starting list of
        directories/modules/testcases/testmethods.
        """

        for test in self.tests:
            if os.path.isdir(test):
                for result in self.process_dir(test):
                    yield result
            else:
                for result in self.process_test_path(test):
                    yield result

    def process_dir(self, dname):
        for f in find_files(dname, match=self.module_pattern,
                            direxclude=_exclude_dir):
            for result in self.process_module(f):
                yield result

    def process_module(self, filename):
        try:
            fname, mod = get_module(filename)
        except ImportError:
            pass
        else:
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj):
                    if issubclass(obj, unittest.TestCase):
                        for result in self.process_testcase(filename, obj):
                            yield result

                elif inspect.isfunction(obj):
                    if fnmatch(name, self.func_pattern):
                        yield ':'.join((filename, obj.__name__))

    def process_testcase(self, fname, testcase):
        for name, method in inspect.getmembers(testcase, inspect.ismethod):
            if fnmatch(name, self.func_pattern):
                yield fname + ':' + testcase.__name__ + '.' + method.__name__

    def process_test_path(self, testpath):
        """Return an iterator of expanded testpath strings found in the
        module/testcase/method specified in testpath.  The format of
        testpath is one of the following:
            <module>
            <module>:<testcase>
            <module>:<testcase>.<method>
            <module>:<function>

        where <module> is either the python module path or the actual
        file system path to the .py file.
        """

        module, _, rest = testpath.partition(':')
        if rest:
            tcasename, _, method = rest.partition('.')
            if method:
                yield testpath
            else:  # could be a test function or a TestCase
                fname, mod = get_module(module)
                try:
                    tcase = get_testcase(fname, mod, tcasename)
                except (AttributeError, TypeError):
                    yield testpath
                else:
                    for result in self.process_testcase(fname, tcase):
                        yield result
        else:
            for result in self.process_module(module):
                yield result


def main():
    parser = _get_parser()
    options = parser.parse_args()

    # pipeline
    #   name(s) of dirs/modules/testcases/tests --> test_discoverer
    #   TestDiscoverer (TestResult) --> TestFilter
    #   TestFilter (TestResult) --> TestRunner
    #   TestRunner may branch to test workers
    #   TestRunner (TestResult) --> ResultProcessor

    tests = options.tests
    if options.cfg:
        tests.extend(read_config_file(options.cfg))
        
    if not tests:
        tests = [os.getcwd()]

    i = 0
    for result in TestRunner(TestDiscoverer(tests)):
        #print result
        i += 1

    print "\nProcessed %d tests" % i

def run_tests():
    main()

if __name__ == '__main__':
    main()
