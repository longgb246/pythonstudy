# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/3
"""
Usage Of 'hyperopt_test1' :
"""

from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.datasets import load_iris
import numpy as np
import hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe


# pip install networkx==1.11
def noUse():
    def anySample1():
        # Download the data and split into training and test sets
        iris = load_iris()
        X = iris.data
        y = iris.target

        # train and test 的划分
        test_size = int(0.2 * len(y))
        np.random.seed(13)
        indices = np.random.permutation(len(X))
        X_train = X[ indices[:-test_size]]
        y_train = y[ indices[:-test_size]]
        X_test = X[ indices[-test_size:]]
        y_test = y[ indices[-test_size:]]

        any_preprocessing = None
        # Instantiate a HyperoptEstimator with the search space and number of evaluations
        estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                                  preprocessing=any_preprocessing('my_pre'),
                                  algo=tpe.suggest,
                                  max_evals=100,
                                  trial_timeout=120)

        # Search the hyperparameter space based on the data
        estim.fit( X_train, y_train )

        # Show the results
        print( estim.score( X_test, y_test ) )
        # 1.0

        print( estim.best_model() )
        # {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
        #           max_depth=3, max_features='log2', max_leaf_nodes=None,
        #           min_impurity_decrease=0.0, min_impurity_split=None,
        #           min_samples_leaf=1, min_samples_split=2,
        #           min_weight_fraction_leaf=0.0, n_estimators=13, n_jobs=1,
        #           oob_score=False, random_state=1, verbose=False,
        #           warm_start=False), 'preprocs': (), 'ex_preprocs': ()}

    def anySample2():
        from hpsklearn import HyperoptEstimator, extra_trees
        from sklearn.datasets import fetch_mldata
        from hyperopt import tpe
        import numpy as np

        # Download the data and split into training and test sets
        digits = fetch_mldata('MNIST original')

        X = digits.data
        y = digits.target

        test_size = int(0.2 * len(y))
        np.random.seed(13)
        indices = np.random.permutation(len(X))
        X_train = X[ indices[:-test_size]]
        y_train = y[ indices[:-test_size]]
        X_test = X[ indices[-test_size:]]
        y_test = y[ indices[-test_size:]]

        # Instantiate a HyperoptEstimator with the search space and number of evaluations
        estim = HyperoptEstimator(classifier=extra_trees('my_clf'),
                                  preprocessing=[],
                                  algo=tpe.suggest,
                                  max_evals=10,
                                  trial_timeout=300)

        # Search the hyperparameter space based on the data
        estim.fit( X_train, y_train )

        # Show the results
        print( estim.score( X_test, y_test ) )
        # 0.962785714286

        print( estim.best_model() )
        # {'learner': ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
        #           max_depth=None, max_features=0.959202875857,
        #           max_leaf_nodes=None, min_impurity_decrease=0.0,
        #           min_impurity_split=None, min_samples_leaf=1,
        #           min_samples_split=2, min_weight_fraction_leaf=0.0,
        #           n_estimators=20, n_jobs=1, oob_score=False, random_state=3,
        #           verbose=False, warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
        pass

    def svmSample1():
        from hyperopt import fmin, tpe, hp, rand
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn import svm
        from sklearn import datasets

        # SVM的三个超参数：C为惩罚因子，kernel为核函数类型，gamma为核函数的额外参数（对于不同类型的核函数有不同的含义）
        # 有别于传统的网格搜索（GridSearch），这里只需要给出最优参数的概率分布即可，而不需要按照步长把具体的值给一个个枚举出来
        parameter_space_svc ={
            # loguniform 表示该参数取对数后符合均匀分布
            'C': hp.loguniform("C", np.log(1), np.log(100)),
            'kernel': hp.choice('kernel', ['rbf','poly']),
            'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
        }

        # 鸢尾花卉数据集，是一类多重变量分析的数据集
        # 通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类
        iris = datasets.load_digits()
        # --------------------划分训练集和测试集--------------------
        train_data = iris.data[0:1300]
        train_target = iris.target[0:1300]
        test_data = iris.data[1300:-1]
        test_target = iris.target[1300:-1]
        # -----------------------------------------------------------

        # 计数器，每一次参数组合的枚举都会使它加1
        count = 0

        def function1(args):
            print(args)
            # **可以把dict转换为关键字参数，可以大大简化复杂的函数调用
            clf = svm.SVC(**args)
            # 训练模型
            clf.fit(train_data,train_target)
            # 预测测试集
            prediction = clf.predict(test_data)
            global count
            count = count + 1
            score = accuracy_score(test_target,prediction)
            print("第%s次，测试集正确率为：" % str(count),score)
            # 由于hyperopt仅提供fmin接口，因此如果要求最大值，则需要取相反数
            return -score

        # algo指定搜索算法，目前支持以下算法：
        # ① 随机搜索(hyperopt.rand.suggest)
        # ② 模拟退火(hyperopt.anneal.suggest)
        # ③ TPE算法（hyperopt.tpe.suggest，算法全称为 Tree-structured Parzen Estimator Approach）
        # max_evals指定枚举次数上限，即使第max_evals次枚举仍未能确定全局最优解，也要结束搜索，返回目前搜索到的最优解
        best = fmin(function1, parameter_space_svc, algo=tpe.suggest, max_evals=2)

        # best["kernel"]返回的是数组下标，因此需要把它还原回来
        kernel_list = ['rbf','poly']
        best["kernel"] = kernel_list[best["kernel"]]

        print("最佳参数为：",best)

        clf = svm.SVC(**best)
        print(clf)


def anySample3():
    # define an objective function
    def objective(args):
        case, val = args
        if case == 'case 1':
            return val
        else:
            return val ** 2

    # define a search space
    space = hp.choice('a',
                      [
                          ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                          ('case 2', hp.uniform('c2', -10, 10))
                      ])

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

    print best
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print hyperopt.space_eval(space, best)
    # -> ('case 2', 0.01420615366247227}


def wikiLearn():
    """
    不是特别懂
    """
    # 1、简单的函数
    from hyperopt import fmin, tpe, hp
    best = fmin(fn=lambda x: x ** 2,
                space=hp.uniform('x', -10, 10),
                algo=tpe.suggest,
                max_evals=100)
    print best
    # 2、使用函数+ok状态
    from hyperopt import fmin, tpe, hp, STATUS_OK
    def objective(x):
        return {'loss': x ** 2, 'status': STATUS_OK }
    best = fmin(objective,
                space=hp.uniform('x', -10, 10),
                algo=tpe.suggest,
                max_evals=100)
    print best
    # 3、使用dict的返回
    import pickle
    import time
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    def objective(x):
        return {
            'loss': x ** 2,
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'other_stuff': {'type': None, 'value': [0, 1, 2]},
            # -- attachments are handled differently
            'attachments': {'time_module': pickle.dumps(time.time)}
        }
    trials = Trials()
    best = fmin(objective,
                space=hp.uniform('x', -10, 10),
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)
    print best
    print trials.trials
    print trials.results
    print trials.losses()
    print trials.statuses()
    # 没明白 attachments 是什么意思
    msg = trials.trial_attachments(trials.trials[5])['time_module']
    time_module = pickle.loads(msg)
    from hyperopt import hp
    space = hp.choice('a',
                      [
                          ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                          ('case 2', hp.uniform('c2', -10, 10))
                      ])
    import hyperopt.pyll.stochastic
    print hyperopt.pyll.stochastic.sample(space)
    # hp.choice(label, options)
    # hp.randint(label, upper)                  # [0，upper]
    # hp.uniform(label, low, high)
    # hp.quniform(label, low, high, q)          # round(uniform(low, high) / q) * q
    # hp.loguniform(label, low, high)
    # hp.qloguniform(label, low, high, q)       # round(exp(uniform(low, high)) / q) * q
    # hp.normal(label, mu, sigma)
    # hp.qnormal(label, mu, sigma, q)           # round(normal(mu, sigma) / q) * q
    # hp.lognormal(label, mu, sigma)
    # hp.qlognormal(label, mu, sigma, q)        # round(exp(normal(mu, sigma)) / q) * q
    # 4、对于sklearn使用
    from hyperopt import hp
    space = hp.choice('classifier_type', [
        {
            'type': 'naive_bayes',
        },
        {
            'type': 'svm',
            'C': hp.lognormal('svm_C', 0, 1),
            'kernel': hp.choice('svm_kernel', [
                {'ktype': 'linear'},
                {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
            ]),
        },
        {
            'type': 'dtree',
            'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
            'max_depth': hp.choice('dtree_max_depth',
                                   [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
            'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
        },
    ])
    # 5、还是没有搞懂 scope.define
    import hyperopt.pyll
    from hyperopt.pyll import scope
    @scope.define
    def foo(a, b=0):
        print 'running foo', a, b
        return a + b / 2
    # -- this will print 0, foo is called as usual.
    print foo(0)
    # In describing search spaces you can use `foo` as you
    # would in normal Python. These two calls will not actually call foo,
    # they just record that foo should be called to evaluate the graph.
    space1 = scope.foo(hp.uniform('a', 0, 10))
    space2 = scope.foo(hp.uniform('a', 0, 10), hp.normal('b', 0, 1))
    # -- this will print an pyll.Apply node
    print space1
    # -- this will draw a sample by running foo()
    print hyperopt.pyll.stochastic.sample(space1)


# 随机搜索(对应是hyperopt.rand.suggest)，模拟退火(对应是hyperopt.anneal.suggest)
# Random Search
# Tree of Parzen Estimators (TPE)
# Annealing
# Tree
# Gaussian Process Tree





