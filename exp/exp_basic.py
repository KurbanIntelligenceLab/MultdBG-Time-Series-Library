import os
import torch

from dBG.GraphEncoder import GraphEncoder_Attn, GraphEncoder_Attn_new
from data_provider.data_loader import dBG_Dataset
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, AttentivePooler
import torch.nn as nn
from data_provider.data_factory import data_provider


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'dBGAttnPool': AttentivePooler,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        if self.args.dBG:
            self.init_dbg_encoder()
        self.model = self._build_model().to(self.device)


    def init_dbg_encoder(self):
        data_set, _ = data_provider(self.args, 'train')
        data_dim = data_set[0][0].shape[1]

        dBG_datasets = [dBG_Dataset(k=self.args.k,
                                    dimensions=data_dim,
                                    disc=disc,
                                    train_data=data_set.data_x.T,
                                    num_neighbors=None, # not used for masking approach
                                    reverse=self.args.reverse,
                                    undirected=self.args.undirected,
                                    device=self.device) for disc in self.args.disc]

        if self.args.d_graph is None:
            self.args.d_graph = data_dim
        """
        dbg_encoder = GraphEncoder_Attn(k=self.args.k,
                                        d_graph=self.args.d_graph,
                                        d_data=data_dim,
                                        graph_data=dBG_dataset.data,
                                        seq_len=self.args.seq_len,
                                        device=self.device,
                                        node_count=dBG_dataset.dBG.graph.number_of_nodes())
        """
        dbg_encoders = [GraphEncoder_Attn_new(k=d.k,
                                        d_graph=self.args.d_graph,
                                        d_data=data_dim,
                                        graph_data=d.data,
                                        seq_len=self.args.seq_len,
                                        device=self.device,
                                        node_feats=d.node_feats,
                                        num_layers=self.args.dBG_enc_layers,
                                        use_gdc=self.args.use_gdc) for d in dBG_datasets]

        self.dBG_dataset = dBG_datasets
        self.args.graph_encoder = nn.ModuleList(dbg_encoders)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
