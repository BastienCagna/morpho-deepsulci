from __future__ import print_function
from ...deeptools.dataset import extract_data
from ..method.svm import SVMPatternClassification
from capsul.api import Process
from soma import aims

import traits.api as traits
import numpy as np
import json


class PatternSVMTraining(Process):
    def __init__(self):
        super(PatternSVMTraining, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False)))
        self.add_trait('pattern', traits.Str(output=False))
        self.add_trait('names_filter', traits.ListStr(output=False))
        self.add_trait('step_1', traits.Bool(
            True, output=False, optional=True))
        self.add_trait('step_2', traits.Bool(
            True, output=False, optional=True))
        self.add_trait('step_3', traits.Bool(
            True, output=False, optional=True))

        self.add_trait('traindata_file', traits.File(output=True))
        self.add_trait('param_file', traits.File(output=True))
        self.add_trait('clf_file', traits.File(output=True))
        self.add_trait('scaler_file', traits.File(output=True))

    def _run_process(self):

        # Compute bounding box and labels
        if self.step_1:
            print()
            print('--------------------------------')
            print('---         STEP (1/3)       ---')
            print('--- EXTRACT DATA FROM GRAPHS ---')
            print('--------------------------------')
            print()

            dict_label, dict_bck, dict_bck_filtered = {}, {}, {}
            dict_searched_pattern = {}
            for gfile in self.graphs:
                print(gfile)
                graph = aims.read(gfile)
                side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
                data = extract_data(graph, flip=True if side == 'R' else False)
                label = 0
                fn, fp = [], []
                for name in data['names']:
                    if name.startswith(self.pattern):
                        label = 1
                    fn.append(sum([1 for n in self.names_filter if name.startswith(n)]))
                    fp.append(1 if name.startswith(self.pattern) else 0)
                bck_filtered = np.asarray(data['bck'])[np.asarray(fn) == 1]
                spattern = np.asarray(data['bck'])[np.asarray(fp) == 1]
                dict_label[gfile] = label
                dict_bck[gfile] = data['bck']
                dict_bck_filtered[gfile] = [list(p) for p in bck_filtered]
                dict_searched_pattern[gfile] = [list(p) for p in spattern]

            # save in parameters file
            param = {'dict_bck': dict_bck,
                     'dict_bck_filtered': dict_bck_filtered,
                     'dict_label': dict_label,
                     'dict_searched_pattern': dict_searched_pattern}
            with open(self.traindata_file, 'w') as f:
                json.dump(param, f)
        else:
            with open(self.traindata_file) as f:
                param = json.load(f)
            dict_bck = param['dict_bck']
            dict_bck_filtered = param['dict_bck_filtered']
            dict_searched_pattern = param['dict_searched_pattern']
            dict_label = param['dict_label']

        method = SVMPatternClassification(
            pattern=self.pattern, names_filter=self.names_filter,
            dict_bck=dict_bck, dict_bck_filtered=dict_bck_filtered,
            dict_searched_pattern=dict_searched_pattern, dict_label=dict_label)

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print('---------------------------')
            print('---      STEP (2/3)     ---')
            print('--- FIX HYPERPARAMETERS ---')
            print('---------------------------')
            print()
            method.find_hyperparameters(self.graphs, self.param_file)
        else:
            with open(self.param_file) as f:
                param = json.load(f)
            method.C = param['C']
            method.gamma = param['gamma']
            method.transrot_init = param['trans']

        if self.step_3:
            print()
            print('--------------------------')
            print('---      STEP (3/3)    ---')
            print('--- SAVE TRAINED MODEL ---')
            print('--------------------------')
            print()
            method.learning(self.graphs)
            method.save(self.clf_file, self.scaler_file)
