# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/10
"""  
Usage Of 'test_learn_1' : 
"""

from abc import abstractmethod, ABCMeta


# ---------------------------------------------
# annotation 部分
def accepts_and_inject_arguments(*types):
    """参数校验以及检测

    1 对输入参数进行类型校验
    2 自动将类中的某方法的参数及参数值设置到类实例中, 执行完后自动返回

    :param types:
    :return:

    例如：@accepts_and_inject_arguments(int, str, float, list)
    """

    def check_accepts(fn):
        def new_f(*args, **kwargs):
            _accepts(fn, types, args, kwargs)
            _self = _inject_arguments(fn, args, kwargs)
            fn(*args, **kwargs)
            return _self

        return new_f

    return check_accepts


def _accepts(fn, types, args, kwargs):
    arguments = []

    # 获取函数中 get all var names，按函数内变量出现的顺序（函数传参->内部局部出现的变量）包括 self 变量
    varnames = fn.func_code.co_varnames

    # 判断是否是 self 函数 division class instance and function
    s = 1 if varnames[0] == 'self' else 0
    varnames = varnames[s:]

    # set unk-v parameters
    arguments.extend(args[s:])

    # 获取函数的默认输入值
    defaults = fn.__defaults__

    # set k-v parameters
    if defaults:
        defaults_varnames = varnames[-len(defaults):]
        defaults_dict = dict(zip(defaults_varnames, defaults))
        defaults_dict.update(kwargs)
        arguments.extend([defaults_dict[varname] for varname in defaults_varnames])

    # check length of parameters and parameter type
    assert len(arguments) == len(types), \
        "args cnt %d does not match %d" % (len(arguments), len(types))
    for (a, t) in zip(arguments, types):
        assert isinstance(a, t) or a is None, \
            "arg %r does not match %s" % (a, t)


def _inject_arguments(fn, args, kwargs):
    # get class instance
    _self = args[0]

    # get all var names
    varnames = fn.func_code.co_varnames[1:]

    # get all default k-v parameter value, then set to the class instance
    defaults = fn.__defaults__
    if defaults:
        _self.__dict__.update(zip(varnames[-len(defaults):], defaults))

    # set k-v parameters to the class instance
    _self.__dict__.update(kwargs)

    # set unk-v parameters to the class instance
    _self.__dict__.update(zip([varname for varname in varnames if kwargs.get(varname) is None], args[1:]))
    return _self


# ---------------------------------------------
# 抽象类 - 元类
class APIBase(object):
    """
        算法接口的基类
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def set_inputs(self):
        """
        设置输入数据
        :return:
        """
        pass

    @abstractmethod
    def set_params(self):
        """
        设置输入参数
        :return:
        """
        pass


class ModelRegressionBase(APIBase):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def structure(self):
        """
        根据输入
        :return:
        """
        pass

    @abstractmethod
    def fit(self):
        """
        拟合参数，返回模型对象
        :return: Object
        """
        pass

    @abstractmethod
    def predict(self):
        """
        执行模型预测
        :return: Forecast
        """
        pass

    @abstractmethod
    def is_fit_available(self):
        """
        判断输入数据是否符合此模型的拟合要求
        :return: Boolean
        """
        pass

    @abstractmethod
    def is_predict_available(self):
        """
        判断输入数据是否符合此模型的预测要求
        :return: Boolean
        """
        pass


class ModelFbprophet(ModelRegressionBase):
    """
        这个文件提供了各类的均值预测类
        - ModelSimpleAvg 随机抽样平均数预测
    """

    _df = None

    _predict_start_date = None

    _model = None

    def __init__(self):
        pass

    @accepts_and_inject_arguments(Timeseries)
    def set_inputs(self, ts):
        pass

    @accepts_and_inject_arguments(int, list)
    def set_params(self, predict_length, holidays):
        pass

    def structure(self):
        if self.is_fit_available() is False:
            return None
        start_date, ts = self.ts.start_date, self.ts.y

        st = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        dates = []
        for i in range(len(ts)):
            dates.append(str(st + dt.timedelta(i)))
        df = pd.DataFrame({'ds': dates, 'y': ts})
        predict_start_date = str(st + dt.timedelta(len(ts)))

        self._df = df
        self._predict_start_date = predict_start_date
        return self

    def fit(self):
        if self._df is None:
            raise ValueError("ERROR: df of feature is None, place call struct_fearture() first")

        HOLIDAY, DS, LOWER_WINDOW, UPPER_WINDOW = 'holiday', 'ds', 'lower_window', 'upper_window'
        hs = []
        if self.holidays is not None:
            hs = [pd.DataFrame({HOLIDAY: holiday[HOLIDAY],
                                DS: pd.to_datetime(holiday[DS]),
                                LOWER_WINDOW: holiday[LOWER_WINDOW],
                                UPPER_WINDOW: holiday[UPPER_WINDOW]}) for holiday in self.holidays]
        holidays = pd.concat(hs)
        with suppress_stdout_stderr():
            m = Prophet(holidays=holidays, yearly_seasonality=True)
            m.fit(self._df)
            self._model = m
        return self

    def predict(self):
        if self._model is None:
            raise ValueError("ERROR: model is None, place call fit() first")
        if self._df is None:
            raise ValueError("ERROR: df of feature is None, place call struct_fearture() first")

        if self.is_predict_available() is False:
            return None

        with suppress_stdout_stderr():
            future = self._model.make_future_dataframe(periods=self.predict_length)
            forecast = self._model.predict(future)
            results = forecast.loc[forecast['ds'] >= self._predict_start_date, 'yhat'].tolist()
            train_y = self._df.loc[self._df['ds'] < self._predict_start_date, 'y'].tolist()
            train_y_hat = forecast.loc[forecast['ds'] < self._predict_start_date, 'yhat'].tolist()
            kpi = Metrics().MAPD(train_y, train_y_hat)

        map(lambda x: round(x, 3), results)
        return Forecast(self.ts.identity, self._predict_start_date, len(results), results, ModelFbprophet.get_name(),
                        kpi)

    def is_fit_available(self):
        return len(self.ts.y) > 180 and sum(self.ts.y) / len(self.ts.y) >= 5.0

    def is_predict_available(self):
        return len(self.ts.y) > 1

    @staticmethod
    def get_name():
        return "fbprophet"
