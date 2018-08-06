#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import Process
from subprocess import call
import argparse
import re
import yaml
import os
import time


def target_action(cmd):
    # exitcode = call(re.split("\s+", cmd))
    # exit(exitcode)
    cmd_str = '[ PID({0}) ] {1}'.format(os.getpid(), cmd)
    print(cmd_str)
    time.sleep(3)
    print(cmd_str + '  [ Finish! ]')


class WorkFlow(object):
    def last_node(self):
        last = []
        for task in self.dag.values():
            if not task.on_success or len(task.on_success) == 0:
                last.append(task)
        return last
    
    def run(self):
        last = self.last_node()
        self.clear_state()
        self._run_in_parallel(last)

    def _run_in_parallel(self, dependencies, current=None):
        # task, dependencies task
        # start current process then return it 
        # block in dependencies
        # if current action is empty then return None

        processes = {}
        for task in dependencies:
            if task.state == TaskNode.STATE_INIT:
                p = self._run_in_parallel(task.dependencies, task)
                if p:
                    processes[task.name] = p

        for name, p in processes.items():
            p.join()
            finished_task = self.dag[name]
            finished_task.complete(p.exitcode)
            # print("task: {} finished with exitcode: {}".format(finished_task.name, finished_task.exitcode))
            if p.exitcode != 0:
                # print("current task {} exit due to dependency process {} fail", current.name if current else 'root')
                exit(p.exitcode)
        
        if not current or not current.action or current.action == '':
            # print("task: {} finish".format(current.name if current else 'root'))
            return None
        current_process = Process(target=target_action, args=(current.action,))
        current_process.start()
        # print("task: {} start: {}".format(current.name, current.action))
        return current_process

    def clear_state(self):
        for task in self.dag.values():
            task.clear()

    def __init__(self, tasks_conf):
        self.dag = {}
        for name, conf in tasks_conf.items():
            self.dag[name] = TaskNode(name, conf)
        self.__instantiate__()

    def __repr__(self):
        return str(self.dag)
    
    def __instantiate__(self):
        for task in self.dag.values():
            if 'on-success' in task.conf:
                for next in task.conf['on-success']:
                    task.on_success.append(self.dag[next])
                    self.dag[next].dependencies.append(task)
    
    
class TaskNode(object):
    STATE_INIT = 0
    STATE_SUCCESS = 1
    STATE_FAIL = -1

    def clear(self):
        self.state = TaskNode.STATE_INIT
        self.exitcode = 0

    def complete(self, exitcode):
        self.exitcode = exitcode
        if exitcode == 0:
            self.state = TaskNode.STATE_SUCCESS
        else:
            self.state = TaskNode.STATE_FAIL

    def __init__(self, name, conf):
        self.name = name
        self.action = conf['action'] if 'action' in conf else None
        self.conf = conf
        self.on_success = []
        self.dependencies = []
        self.state = TaskNode.STATE_INIT
        self.exitcode = 0

    def __repr__(self):
        parents = ('(' + ''.join([task.name + '->,' for task in self.dependencies]) + ')') if len(self.dependencies) > 0 else ''
        children = ('(' + ''.join(['->' + task.name + ',' for task in self.on_success]) + ')') if len(self.on_success) > 0 else ''
        return parents + self.name + children


def parse_workflow(file_name):
    with open(file_name, 'r') as file:
        wf_conf = yaml.safe_load(file)
    # print(wf_conf)
    # just support one node workflow
    for wf_name, wf_val in wf_conf.items():
        if not "tasks" in wf_val:
            raise KeyError("no tasks in workflow " + wf_name)
        if 'description' in wf_val:
            # print(wf_name + ":" + wf_val['description'])
            pass
        tasks_conf = wf_val['tasks']
        break
    else:
        raise KeyError("workflow node not found")
    wf = WorkFlow(tasks_conf)
    return wf


def run_workflow(i):
    parser = argparse.ArgumentParser(description='forecast workflow')
    parser.add_argument('--conf', metavar='workflow.yaml', type=str, help='workflow.yaml')
    config = parser.parse_args()
    wf = parse_workflow(config.conf if config.conf else 'workflow.yaml')
    wf.run()
    return i


if __name__ == '__main__':
    run_workflow(1)

