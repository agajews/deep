import settings
import json
import pickle
import os
import shutil
import signal
import sys
from pprint import pformat


class struct(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    todict = lambda self: dict(**self)
    copy = lambda self: struct(**dict.copy(self))


class Logger(object):
    def __init__(self,
                 name,
                 hypers,
                 state,
                 log_dir=settings.log_dir,
                 parent=None,
                 metric_show_freq=100,
                 overwrite=False,
                 load=False):

        self.name = name
        if parent is not None:
            log_dir = parent.log_dir
        self.log_dir = os.path.join(log_dir, name)
        self.param_dir = os.path.join(self.log_dir, 'params')
        self.hypers = hypers
        self.state = state
        self.inventory_fnm = os.path.join(self.log_dir, 'inventory.json')

        initialize = False
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
            initialize = True

        elif overwrite:
            confirm = input(
                'Are you sure you want to overwrite log {}? yes/[NO] '.format(
                    self.log_dir))
            if not confirm == 'yes':
                print('Log directory {} exists, not overwriting'.format(
                    self.log_dir))
                sys.exit(0)
            print('Warning: overwriting log directory {}'.format(self.log_dir))
            shutil.rmtree(self.log_dir)
            os.mkdir(self.log_dir)
            initialize = True

        self.loaded = False
        if initialize:
            os.mkdir(self.param_dir)

            self.inventory = struct(
                hypers=os.path.join(self.log_dir, 'hypers.json'),
                state=os.path.join(self.log_dir, 'state.json'),
                params=[],
                metrics=os.path.join(self.log_dir, 'metrics.json'),
                n_metrics=0,
                metric_show_freq=metric_show_freq,
                logs_since_show=0)

            assert hypers is not None
            self.save_str(json.dumps(hypers), self.inventory.hypers)

            assert state is not None
            self.save_str(json.dumps(state), self.inventory.state)

            self._metrics = []
            self.save_str(json.dumps(self._metrics), self.inventory.metrics)

            self.save_str(json.dumps(self.inventory), self.inventory_fnm)

        elif load:
            # TODO: load params, inventory, hypers from checkpoint
            self.loaded = True
            self.inventory = struct(
                json.loads(self.load_str(self.inventory_fnm)))
            self.inventory.logs_since_show = 0
            self.hypers.update(
                json.loads(self.load_str(self.inventory.hypers)))
            self.state.update(json.loads(self.load_str(self.inventory.state)))
            self._metrics = json.loads(self.load_str(self.inventory.metrics))

        else:
            raise Exception('Log directory {} exists, not overwriting'.format(
                self.log_dir))

        self.params_reader = None
        self.params_writer = None

        self.saving_params = False
        self.flushing = False
        self.kill_asap = False

        signal.signal(signal.SIGINT, self.int_handler)
        signal.signal(signal.SIGTERM, self.int_handler)

    # def load_params(self):
    #     if len(self.inventory.params) > 0:
    #         saved_params = self.load_pk(self.inventory.params[-1]['fnm'])
    #         for name, param in self.params:
    #             param.data.copy_(saved_params[name])

    # def set_params(self, params):
    #     self.params = list(params)
    #     self.load_params()

    def set_params_reader(self, reader):
        self.params_reader = reader

    def set_params_writer(self, writer):
        self.params_writer = writer
        if self.loaded and len(self.inventory.params) > 0:
            self.params_writer(
                struct(**self.load_pk(self.inventory.params[-1]['fnm'])))

    def save_str(self, string, fnm):
        with open(fnm, 'w') as f:
            f.write(string)

    def load_str(self, fnm):
        with open(fnm, 'r') as f:
            return f.read()

    def save_pk(self, obj, fnm):
        with open(fnm, 'wb') as f:
            pickle.dump(obj, f)

    def load_pk(self, fnm):
        with open(fnm, 'rb') as f:
            return pickle.load(f)

    def log_params(self):
        if self.params_reader is not None:
            self.saving_params = True
            fnm = os.path.join(self.param_dir, 'e{}_bn{}.pk'.format(
                self.state.epoch, self.state.bn))
            self.inventory.params.append({
                'epoch': self.state.epoch,
                'bn': self.state.bn,
                'fnm': fnm
            })
            self.save_pk(self.params_reader().todict(), fnm)
            self.saving_params = False
        if self.kill_asap:
            self.kill()

    def log_metrics(self, metrics, desc='Train', show=False):
        metrics = metrics.todict()
        metrics.update(dict(epoch=self.state.epoch, bn=self.state.bn))
        self._metrics.append(metrics)
        self.inventory.n_metrics += 1
        self.inventory.logs_since_show += 1
        if show or (self.inventory.metric_show_freq != 0
                    and self.inventory.logs_since_show %
                    self.inventory.metric_show_freq == 0):
            print('{} {}:{}, {}'.format(desc, self.state.epoch, self.state.bn,
                                        pformat(metrics)))
            self.inventory.logs_since_show = 0

    def metrics(self):
        return [struct(**metrics) for metrics in self._metrics]

    def update_metrics(self, updater):
        for metric in self._metrics:
            metric.update(updater(struct(**metric)))

    def flush(self):
        self.flushing = True
        self.save_str(json.dumps(self.inventory), self.inventory_fnm)
        self.save_str(json.dumps(self._metrics), self.inventory.metrics)
        self.save_str(json.dumps(self.state), self.inventory.state)
        self.flushing = False
        if self.kill_asap:
            self.kill()

    def int_handler(self, signum, frame):
        if self.saving_params or self.flushing:
            self.kill_asap = True
            print('Finishing flushing or saving...')
            return
        self.kill()

    def kill(self):
        print('Done')
        sys.exit(0)
