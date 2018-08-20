#-*- coding:utf-8 -*-
from multiprocessing import Process, Queue, Pipe, Lock, Value, Array, Manager, Pool, TimeoutError
import multiprocessing
import os
import numpy as np
import time
import signal
# print 'process id:', os.getpid()


# 基础。Process
def ex1():
    def info(title):
        print title
        print 'module name:', __name__
        if hasattr(os, 'getppid'):  # only available on Unix
            print 'parent process:', os.getppid()
        print 'process id:', os.getpid()

    def f(name):
        info('function f')
        print 'hello', name

    if __name__ == '__main__':
        info('main line')
        p = Process(target=f, args=('bob',))
        p.start()
        p.join()

# 在进程之间交换对象，Queues、Pipes
def ex2():
    def f1(q, data1):
        print 'f1'
        q.put([42, None, 'hello'])
        q.put(data1)


    def f2(q, data2):
        print 'f2'
        q.put([42, None, 'hello'])
        data2 = np.array(data2) + 1
        q.put(data2)


    if __name__ == '__main__':
        q = Queue()
        data1 = [1, 2, 3]
        p1 = Process(target=f1, args=(q, data1))
        data2 = [2, 3, 4]
        p2 = Process(target=f2, args=(q, data2))
        p1.start()
        p1.join()
        p2.start()
        p2.join()
        print q.get()    # prints "[42, None, 'hello']"
        print q.get()
        print q.get()
        print q.get()
        print q.get()
        print q.get()
def ex3():
    def f(conn):
        conn.send([42, None, 'hello'])
        conn.send([42, None, 'hello2'])
        conn.close()

    if __name__ == '__main__':
        parent_conn, child_conn = Pipe()
        p = Process(target=f, args=(child_conn,))
        p.start()
        print parent_conn.recv()   # prints "[42, None, 'hello']"
        print parent_conn.recv()   # prints "[42, None, 'hello']"
        p.join()

# 进程之间的同步，Lock
def ex4():
    def f(l, i):
        l.acquire()
        print 'hello world', i
        l.release()

    if __name__ == '__main__':
        lock = Lock()
        for num in range(10):
            Process(target=f, args=(lock, num)).start()

# 在进程之间共享状态，共享内存（Value, Array）、服务器进程（Manager）
def ex5():
    def f(n, a):
        print 'process id:', os.getpid()
        n.value = 3.1415927
        for i in range(len(a)):
            a[i] = -a[i]

    if __name__ == '__main__':
        print 'process id:', os.getpid()
        num = Value('d', 0.0)
        arr = Array('i', range(10))
        print num.value
        print arr[:]

        p = Process(target=f, args=(num, arr))
        p.start()
        p.join()

        print num.value
        print arr[:]
def ex6():
    def f(d, l):
        print 'process F id:', os.getpid()
        d[1] = '1'
        d['2'] = 2
        d[0.25] = None
        l.reverse()

    if __name__ == '__main__':
        print 'process Main id:', os.getpid()
        manager = Manager()

        d = manager.dict()
        l = manager.list(range(10))

        p = Process(target=f, args=(d, l))
        p.start()
        p.join()

        print d
        print l

# 【这个没懂】
def ex7():
    def f(x):
        return x*x

    if __name__ == '__main__':
        pool = Pool(processes=4)              # start 4 worker processes

        # print "[0, 1, 4,..., 81]"
        print pool.map(f, range(10))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print i

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20,))      # runs in *only* one process
        print res.get(timeout=1)              # prints "400"

        # evaluate "os.getpid()" asynchronously
        res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        print res.get(timeout=1)              # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print [res.get(timeout=1) for res in multiple_results]

        # make a single worker sleep for 10 secs
        res = pool.apply_async(time.sleep, (10,))
        try:
            print res.get(timeout=1)
        except TimeoutError:
            print "We lacked patience and got a multiprocessing.TimeoutError"

# Process基础，is_alive()
def ex8():
    if __name__ == '__main__':
        p = multiprocessing.Process(target=time.sleep, args=(1000,))
        print p, p.is_alive()

        p.start()
        print p, p.is_alive()

        p.terminate()
        time.sleep(0.1)
        print p, p.is_alive()

        print p.exitcode
        p.exitcode == -signal.SIGTERM
        print p.exitcode




def worker_1(interval):
    print "worker_1"
    time.sleep(interval)
    print "end worker_1"

def worker_2(interval):
    print "worker_2"
    time.sleep(interval)
    print "end worker_2"

def worker_3(interval):
    print "worker_3"
    time.sleep(interval)
    print "end worker_3"

if __name__ == "__main__":
    p1 = multiprocessing.Process(target = worker_1, args = (2,))
    p2 = multiprocessing.Process(target = worker_2, args = (3,))
    p3 = multiprocessing.Process(target = worker_3, args = (4,))

    p1.start()
    p2.start()
    p3.start()

    print 'p1.is_alive()', p1.is_alive()

    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child   p.name:" + p.name + "\tp.id" + str(p.pid))
    print "END!!!!!!!!!!!!!!!!!"

