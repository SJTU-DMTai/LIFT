
from exp.exp_main import Exp_Main
from models import LIFT

import warnings
warnings.filterwarnings('ignore')


class Exp_Lead(Exp_Main):
    def __init__(self, args):
        super(Exp_Lead, self).__init__(args)

    def _get_data(self, flag, **kwargs):
        return super()._get_data(flag, prefetch_path=f'{self.args.prefetch_path}_{flag}.npz',
                                 leader_num=self.args.leader_num,
                                 prefetch_batch_size=self.args.prefetch_batch_size,
                                 variable_batch_size=self.args.variable_batch_size,
                                 efficient=self.args.efficient, local_max=True,
                                 pin_gpu=self.args.pin_gpu,
                                 pred_path=self.args.pred_path if hasattr(self.args, 'pred_path') else None,
                                 **kwargs)

    def _build_model(self, model=None, framework_class=None):
        if self.args.lift and 'LightMTS' not in self.args.model:
            framework_class = LIFT.Model
        model = super()._build_model(model, framework_class=framework_class)
        return model
