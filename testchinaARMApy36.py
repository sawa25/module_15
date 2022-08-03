import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import matplotlib.pyplot as plt



class arima_model:
 
    def __init__(self, ts, maxLag=9):
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxsize
 
    # Рассчитайте оптимальную модель ARIMA и назначьте соответствующие результаты соответствующим атрибутам
    def get_proper_model(self,pp=None,qq=None):
        self._proper_model(pp,qq)
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)
 
    # Для p, q в заданном диапазоне рассчитайте лучшую модель arima, вот подгонка данных с хорошей разницей, поэтому разница всегда равна 0
    def _proper_model(self,pp,qq):
        for p in [pp] if not(pp is None) else np.arange(self.maxLag):
            for q in [qq] if not(qq is None) else np.arange(self.maxLag):
                # print p,q,self.bic
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARMA.bic
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()
 
    # модель определения параметров
    def certain_model(self, p, q):
            model = ARMA(self.data_ts, order=(p, q))
            try:
                self.properModel = model.fit( disp=-1, method='css')
                self.p = p
                self.q = q
                self.bic = self.properModel.bic
                self.predict_ts = self.properModel.predict()
                self.resid_ts = deepcopy(self.properModel.resid)
            except:
                print ('You can not fit the model with this parameter p,q, ' \
                      'please use the get_proper_model method to get the best model')
 
    # Предсказать значение на второй день
    def forecast_next_day_value(self, type='day'):
        # Я изменил исходный код arima_model в пакете statsmodels и добавил постоянный атрибут, мне нужно сначала запустить метод прогноза и присвоить значение константе
        self.properModel.forecast()
        if self.data_ts.index[-1] != self.resid_ts.index[-1]:
            raise ValueError('''The index is different in data_ts and resid_ts, please add new data to data_ts.
            If you just want to forecast the next day data without add the real next day data to data_ts,
            please run the predict method which arima_model included itself''')
        if not self.properModel:
            raise ValueError('The arima model have not computed, please run the proper_model method before')
        para = self.properModel.params
 
        # print self.properModel.params
        if self.p == 0:   # It will get all the value series with setting self.data_ts[-self.p:] when p is zero
            ma_value = self.resid_ts[-self.q:]
            values = ma_value.reindex(index=ma_value.index[::-1])
        elif self.q == 0:
            ar_value = self.data_ts[-self.p:]
            values = ar_value.reindex(index=ar_value.index[::-1])
        else:
            ar_value = self.data_ts[-self.p:]
            ar_value = ar_value.reindex(index=ar_value.index[::-1])
            ma_value = self.resid_ts[-self.q:]
            ma_value = ma_value.reindex(index=ma_value.index[::-1])
            values = ar_value.append(ma_value)
 
        predict_value = np.dot(para[1:], values) #+ self.properModel.constant[0]
        self._add_new_data(self.predict_ts, predict_value, type)
        return predict_value
 
    # Динамически добавлять функции данных, которые обрабатываются отдельно для индекса месяца и дня.
    def _add_new_data(self, ts, dat, type='day'):
        if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
        elif type == 'month':
            new_index = ts.index[-1] + relativedelta(months=1)
        ts[new_index] = dat
 
    def add_today_data(self, dat, type='day'):
        self._add_new_data(self.data_ts, dat, type)
        if self.data_ts.index[-1] != self.predict_ts.index[-1]:
            raise ValueError('You must use the forecast_next_day_value method forecast the value of today before')
        self._add_new_data(self.resid_ts, self.data_ts[-1] - self.predict_ts[-1], type)
 
if __name__ == '__main__':
    # df = pd.read_csv('AirPassengers.csv', encoding='utf-8', index_col='date')
    # df.index = pd.to_datetime(df.index)
    # ts = df['x']
    DIR=r'C:\Users\User\SkillFactory\GitHub\module_15\\'
    df = pd.read_csv(f'{DIR}AirPassengers.csv', encoding='utf-8')
    df.columns = (['date','x']) 
    ts = df.set_index(pd.DatetimeIndex(df['date']))['x'] 
    #чтобы арима не ругалась на отсутствие freq в индексе
    ts.index= pd.DatetimeIndex(ts.index.values,freq=ts.index.inferred_freq)

 
    # предварительная обработка данных
    ts_log = np.log(ts)
    rol_mean = ts_log.rolling(window=12).mean()
    rol_mean.dropna(inplace=True)
    ts_diff_1 = rol_mean.diff(1)
    ts_diff_1.dropna(inplace=True)
    ts_diff_2 = ts_diff_1.diff(1)
    ts_diff_2.dropna(inplace=True)
 
    # модель подходит
    model = arima_model(ts_diff_2)
    #  Здесь используйте параметры модели для автоматической идентификации
    # model.get_proper_model() #автоперебор pq
    model.get_proper_model(0,1) 
    print ('bic:', model.bic, 'p:', model.p, 'q:', model.q)
    print (model.properModel.forecast()[0])
    print (model.forecast_next_day_value(type='month'))
 
    # Восстановление результата прогноза
    predict_ts = model.properModel.predict()
    diff_shift_ts = ts_diff_1.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts)
    rol_shift_ts = rol_mean.shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    rol_sum = ts_log.rolling(window=11).sum()
    rol_recover = diff_recover*12 - rol_sum.shift(1)
    log_recover = np.exp(rol_recover)
    log_recover.dropna(inplace=True)
 
    # Картирование результатов прогнозирования
    ts = ts[log_recover.index]
    plt.figure(facecolor='white')
    log_recover.plot(color='blue', label='Predict')
    ts.plot(color='red', label='Original')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts)**2)/ts.size))
    plt.show()
