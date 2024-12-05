from loguru import logger
from engines.data import DataManager
from engines.utils.setup_seed import setup_seed
from pprint import pprint
import torch
import os

class Config:
    def __init__(self,name_model = "ptm_bilstm_crf",test_file = r'data/our_datasets/test_data_processed.json'):
        self.configure = {
    # 训练数据集
    'train_file': r'data\our_datasets\train_data_processed.json',
    # 验证数据集
    'dev_file': '',
    # 使用交叉验证
    'kfold': False,
    'fold_splits': 5,
    # 没有验证集时，从训练集抽取验证集比例
    'validation_rate': 0.1,
    # 测试数据集
    'test_file': test_file,
    # 单词还是方块字
    # 西方语法单词: word
    # 中日韩等不需要空格的方块字: char
    'token_level': 'char',
    # 存放词表的地方
    'token_file': 'data/our_datasets/token2id_train.txt',
    # 使用的预训练模型，这个地方的模型路径是和huggingface上的路径对应的
    'ptm': './chinese-bert-wwm-ext',
    # 'ptm': 'Davlan/bert-base-multilingual-cased-ner-hrl',
    # 使用的方法
    # sequence_tag:序列标注
    # span:方式
    'method': 'sequence_tag',
    # 使用的模型
    # sequence label方式:
    # ptm crf: ptm_crf
    # ptm bilstm crf: ptm_bilstm_crf
    # ptm idcnn crf: ptm_idcnn_crf
    # idcnn crf: idcnn_crf
    # bilstm crf: bilstm_crf
    # ptm: ptm
    # span方式:
    # binary pointer: ptm_bp
    # global pointer: ptm_gp
    'model_type': f'{name_model}',
    # 选择lstm时，隐藏层大小
    'hidden_dim': 200,
    # Embedding向量维度
    'embedding_dim': 300,
    # 选择idcnn时filter的个数
    'filter_nums': 64,
    # 模型保存的文件夹
    'checkpoints_dir': f'checkpoints/{name_model}_our_datasets',
    # 模型名字
    'model_name': f'{name_model}.pkl',
    # 类别列表
    'span_classes': ['PER', 'ORG', 'LOC'],
    'sequence_tag_classes': ["O", 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'],
    # 'sequence_tag_classes': ['O', 'B-DATE', 'I-DATE', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'],
    # decision_threshold
    'decision_threshold': 0.5,
    # 使用bp和gp的时候是否使用苏神的多标签分类的损失函数，默认使用BCELoss
    'use_multilabel_categorical_cross_entropy': True,
    # 使用对抗学习
    'use_gan': False,
    # fgsm:Fast Gradient Sign Method
    # fgm:Fast Gradient Method
    # pgd:Projected Gradient Descent
    # awp: Weighted Adversarial Perturbation
    'gan_method': 'pgd',
    # 对抗次数
    'attack_round': 3,
    # 使用Multisample Dropout
    # 使用Multisample Dropout后dropout会失效
    'multisample_dropout': False,
    'dropout_round': 5,
    # 随机种子
    'seed': 3407,
    # 预训练模型是否前置加入Noisy
    'noisy_tune': False,
    'noise_lambda': 0.12,
    # 是否进行warmup
    'warmup': False,
    # 是否进行随机权重平均swa
    'swa': False,
    'swa_start_step': 5000,
    'swa_lr': 1e-6,
    
    # 每个多久平均一次
    'anneal_epochs': 1,
    # 使用EMA
    'ema': False,
    # warmup方法，可选：linear、cosine
    'scheduler_type': 'linear',
    # warmup步数，-1自动推断为总步数的0.1
    'num_warmup_steps': -1,
    # 句子最大长度
    'max_sequence_length': 64,
    # epoch
    'epoch': 50,
    # batch_size
    'batch_size': 24,
    # dropout rate
    'dropout_rate': 0.5,
    # 每print_per_batch打印损失函数
    'print_per_batch': 100,
    # learning_rate
    'learning_rate': 5e-5,
    # 初始学习率为5e-5
    # 优化器选择
    'optimizer': 'AdamW',
    # 执行权重初始化，仅限于非微调
    'init_network': False,
    # 权重初始化方式，可选：xavier、kaiming、normal
    'init_network_method': 'xavier',
    # fp16混合精度训练，仅在Cuda支持下使用
    'use_fp16': True,
    # 训练是否提前结束微调
    'is_early_stop': True,
    # 训练阶段的patient
    'patient': 5,
}

        

def fold_check(configures):
    if configures['checkpoints_dir'] == '':
        raise Exception('checkpoints_dir did not set...')

    if not os.path.exists(configures['checkpoints_dir']):
        print('checkpoints fold not found, creating...')
        os.makedirs(configures['checkpoints_dir'])

def Pre(configure):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    setup_seed(configure['seed'])
    fold_check(configure)
    log_name = './logs/' + "interactive_predict" + '.log'
    logger.add(log_name, encoding='utf-8')
    
    use_cuda = True
    cuda_device = 0
    if use_cuda:
        if torch.cuda.is_available():
            if cuda_device == -1:
                device = torch.dev1ice('cuda')
            else:
                device = torch.device(f'cuda:{cuda_device}')
        else:
            raise ValueError(
                "'use_cuda' set to True when cuda is unavailable."
                " Make sure CUDA is available or set use_cuda=False."
            )
    else:
        device = 'cpu'
    from engines.predict import Predictor
    data_manager = DataManager(configure, logger=logger)
    logger.info('mode: predict_one')
    predictor = Predictor(configure, data_manager,device,logger)
    predictor.predict_one('warm up')
    return predictor
