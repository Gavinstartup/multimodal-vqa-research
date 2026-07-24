"""
Fusion: 多模态融合框架中央配置文件
本文件包含所有模型、数据处理和训练相关的参数配置
"""

import torch
import torch.nn as nn
import os
from pathlib import Path

#============================================================
# 模型架构设置
#============================================================
class ModelSettings:
    """图像和文本模型的基本设置"""
    # ViT 模型设置
    VIT_MODEL_NAME = "ViT-B-32"  # 可选: "ViT-B-32", "ViT-B-16", "ViT-L-14"
    VIT_PRETRAINED = "openai"    # 预训练权重来源，可选: "openai", "laion400m", "laion2b"
    VIT_IMAGE_SIZE = 224         # 输入图像的大小 (高=宽)
    
    # LLaMA 模型设置
    LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # LLaMA模型HF路径
    LLAMA_MAX_LENGTH = 512       # 最大处理的token数量
    FREEZE_LLM = True            # 是否冻结LLM参数(仅训练MLP投影器)

    # new quantization
    USE_QUANTIZATION = False      # 是否使用量化
    QUANTIZATION_BITS = 8        # 量化位数，8位提供更好的精度
    
    # 模型缓存设置
    USE_MODEL_CACHE = True       # 是否在多次调用时缓存模型实例


#============================================================
# 特征维度设置
#============================================================
class DimensionSettings:
    """不同阶段的特征维度配置"""
    # 原始特征维度
    IMAGE_TOKEN_DIM = 512        # ViT-B-32输出的图像特征维度
    TEXT_TOKEN_DIM = 2048        # LLaMA 3-8B的文本嵌入维度
    
    # MLP投影设置 - 5层架构
    HIDDEN_DIM_1 = 768           # 第一隐藏层维度
    HIDDEN_DIM_2 = 1024          # 第二隐藏层维度
    HIDDEN_DIM_3 = 1536          # 第三隐藏层维度
    HIDDEN_DIM_4 = 1792          # 第四隐藏层维度
    OUTPUT_DIM = 2048            # 输出维度，需与文本嵌入维度匹配
    FUSION_TOKEN_DIM = 2048      # 融合后的token维度


#============================================================
# MLP投影器设置
#============================================================
class MLPSettings:
    """MLP投影器的结构和训练设置"""
    # 网络结构
    MLP_DROPOUT = 0.2            # Dropout比率 (0-1)，增加到0.2
    USE_LAYERNORM = True         # 是否使用LayerNorm
    MLP_ACTIVATION = torch.nn.ReLU  # 激活函数类型
    
    # 模型权重
    WEIGHTS_FILENAME = "mlp_projector_final.pt"  # 权重文件名
    WEIGHTS_PATH = os.path.join("models", "mlp_weights")  # 权重保存路径
    
    # 静态方法获取完整权重路径
    @staticmethod
    def get_full_weights_path():
        path = os.path.join(MLPSettings.WEIGHTS_PATH, MLPSettings.WEIGHTS_FILENAME)
        return path


#============================================================
# 训练设置
#============================================================
class TrainingSettings:
    """MLP投影器训练相关参数"""
    # 基本训练参数
    BATCH_SIZE = 4               # 训练批次大小
    EPOCH = 30                   # 训练总轮数
    LEARNING_RATE = 5e-5         # 初始学习率
    WEIGHT_DECAY = 5e-3          # 权重衰减系数
    
    # 训练控制参数
    LOG_INTERVAL = 10            # 日志记录间隔（批次）
    VALIDATION_SPLIT = 0.1       # 验证集比例
    EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
    
    # 学习率调度
    LR_SCHEDULER = "reduce_on_plateau"  # 学习率调度策略, 可选: "reduce_on_plateau", "cosine", "linear"
    LR_PATIENCE = 4              # ReduceLROnPlateau耐心值
    LR_FACTOR = 0.3              # 学习率衰减因子
    WARMUP_STEPS = 100           # 预热步数(仅用于cosine和linear调度)
    
    # 损失函数设置
    LOSS_FUNCTION = "cosine"     # 损失函数类型, 可选: "cosine", "mse", "contrastive"
    TEMPERATURE = 0.1            # 对比损失的温度参数(仅用于contrastive损失)
    MSE_LOSS_WEIGHT = 0.7        # MSE/余弦损失权重
    CONTRASTIVE_LOSS_WEIGHT = 0.3 # 对比损失权重
    
    # 优化器设置
    OPTIMIZER = "adamw"          # 优化器类型, 可选: "adam", "adamw", "sgd"
    
    # 梯度累积设置
    GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数，有效批大小 = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS


#============================================================
# 注意力掩码设置
#============================================================
class AttentionMaskSettings:
    """控制不同模态间信息流动的注意力掩码配置"""
    # 控制开关
    USE_CUSTOM_ATTENTION_MASKS = True   # 是否使用自定义注意力掩码，而非模型默认掩码
    
    # 交互控制
    USE_CAUSAL_MASK_FOR_TEXT = True     # 文本token之间是否使用因果掩码(自回归)
    ALLOW_TEXT_TO_ATTEND_IMAGES = True  # 文本token是否可以看到图像token
    ALLOW_IMAGE_TO_ATTEND_TEXT = False  # 图像token是否可以看到文本token
    
    # 高级设置
    USE_SLIDING_WINDOW = False          # 是否使用滑动窗口注意力(用于长序列)
    WINDOW_SIZE = 512                   # 滑动窗口大小
    GLOBAL_TOKENS = 16                  # 全局token数量(若使用滑动窗口)


#============================================================
# 数据和路径设置
#============================================================
class PathSettings:
    """数据集和模型文件的路径配置"""
    # 基础路径
    BASE_DIR = Path("/content/drive/MyDrive/MODELSFUSION/MODELSFUSION/data") # 项目基础目录
                   
    
    # 数据路径
    DATA_DIR = BASE_DIR                               # 数据总目录
    IMAGE_PATH = DATA_DIR / "images"                  # 图像文件目录
    TEXT_JSON = DATA_DIR / "annotations.json"         # 文本JSON文件路径
    
    # 模型和输出路径
    MODELS_DIR = BASE_DIR / "models"                  # 模型保存目录
    OUTPUT_DIR = BASE_DIR / "outputs"                 # 输出结果目录
    LOG_DIR = BASE_DIR / "logs"                       # 日志保存目录
    
    # 权重文件
    CHECKPOINT_DIR = MODELS_DIR / "checkpoints"       # 检查点保存目录
    
    # 静态方法创建所有必要目录
    @staticmethod
    def create_directories():
        """创建所有必要的目录"""
        dirs = [
            PathSettings.DATA_DIR,
            PathSettings.IMAGE_PATH,
            PathSettings.MODELS_DIR,
            PathSettings.OUTPUT_DIR,
            PathSettings.LOG_DIR,
            PathSettings.CHECKPOINT_DIR
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)


#============================================================
# 运行时设置
#============================================================
class RuntimeSettings:
    """运行时环境设置"""
    # 设备配置
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    
    # 多GPU设置
    USE_DATAPARALLEL = torch.cuda.device_count() > 1  # 多GPU时使用DataParallel
    
    # 性能设置
    AMP_ENABLED = True           # 是否启用自动混合精度
    JIT_ENABLED = False          # 是否启用JIT编译
    
    # 内存管理
    PIN_MEMORY = True            # 数据加载器是否使用pin_memory
    CACHE_IMAGES = False         # 是否缓存预处理后的图像特征
    EMPTY_CACHE_FREQ = 10        # 清空CUDA缓存的频率(每N个批次)


#============================================================
# 文本生成设置
#============================================================
class GenerationSettings:
    """文本生成控制参数"""
    # 基本参数
    GENERATION_TEMPERATURE = 0.7       # 温度参数(较高增加随机性, 较低增加确定性) (0.1-2.0)
    GENERATION_TOP_P = 0.9             # 核采样参数 (0.0-1.0)
    GENERATION_TOP_K = 40              # Top-K采样参数 (1-100)
    GENERATION_MAX_LENGTH = 256        # 生成文本的最大长度
    GENERATION_MIN_LENGTH = 10         # 生成文本的最小长度
    
    # 控制参数
    GENERATION_REPETITION_PENALTY = 1.2  # 重复惩罚系数 (1.0-2.0)
    GENERATION_NO_REPEAT_NGRAM_SIZE = 3  # 避免重复的N-gram大小
    
    # 高级设置
    GENERATION_NUM_BEAMS = 1          # 集束搜索的束数 (1=贪婪解码)
    GENERATION_LENGTH_PENALTY = 1.0    # 长度惩罚(>1鼓励长回答, <1鼓励短回答)
    GENERATION_EARLY_STOPPING = True   # 是否在所有束都生成结束标记时提前停止


# 导出配置变量，为了向后兼容
# ======Model_Settings======
ViT_Model_Name = ModelSettings.VIT_MODEL_NAME
ViT_Pretrained = ModelSettings.VIT_PRETRAINED
LlaMa_Model_Name = ModelSettings.LLAMA_MODEL_NAME
Freeze_LLM = ModelSettings.FREEZE_LLM

# ======Token_Dimensions======
Image_Token_Dim = DimensionSettings.IMAGE_TOKEN_DIM
Text_Token_Dim = DimensionSettings.TEXT_TOKEN_DIM
HIDDEN_DIM_1 = DimensionSettings.HIDDEN_DIM_1
HIDDEN_DIM_2 = DimensionSettings.HIDDEN_DIM_2
HIDDEN_DIM_3 = DimensionSettings.HIDDEN_DIM_3
HIDDEN_DIM_4 = DimensionSettings.HIDDEN_DIM_4
Output_dim = DimensionSettings.OUTPUT_DIM
Fusion_Token_Dim = DimensionSettings.FUSION_TOKEN_DIM

# ======MLP Projector Settings=======
MLP_Dropout = MLPSettings.MLP_DROPOUT
Use_LayerNorm = MLPSettings.USE_LAYERNORM
MLP_Activation = MLPSettings.MLP_ACTIVATION
Weights_path = MLPSettings.get_full_weights_path()

#=======MLP_Projector_Training=======
Batch_Size = TrainingSettings.BATCH_SIZE
Epoch = TrainingSettings.EPOCH
Learning_Rate = TrainingSettings.LEARNING_RATE
Weight_Decay = TrainingSettings.WEIGHT_DECAY
Log_Interval = TrainingSettings.LOG_INTERVAL
MSE_LOSS_WEIGHT = TrainingSettings.MSE_LOSS_WEIGHT
CONTRASTIVE_LOSS_WEIGHT = TrainingSettings.CONTRASTIVE_LOSS_WEIGHT
GRADIENT_ACCUMULATION_STEPS = TrainingSettings.GRADIENT_ACCUMULATION_STEPS

#=======Attention_Mask_Settings=======
Use_Custom_Attention_Masks = AttentionMaskSettings.USE_CUSTOM_ATTENTION_MASKS
Use_Causal_Mask_For_Text = AttentionMaskSettings.USE_CAUSAL_MASK_FOR_TEXT
Allow_Text_To_Attend_Images = AttentionMaskSettings.ALLOW_TEXT_TO_ATTEND_IMAGES
Allow_Image_To_Attend_Text = AttentionMaskSettings.ALLOW_IMAGE_TO_ATTEND_TEXT

#=========Data/Path=========
Image_Path = str(PathSettings.IMAGE_PATH)
Text_Json = str(PathSettings.TEXT_JSON)

#=======Runtime_Device=======
Device = RuntimeSettings.DEVICE

# ======Text_Generation_Settings======
Generation_Temperature = GenerationSettings.GENERATION_TEMPERATURE
Generation_Top_P = GenerationSettings.GENERATION_TOP_P
Generation_Max_Length = GenerationSettings.GENERATION_MAX_LENGTH
Generation_Repetition_Penalty = GenerationSettings.GENERATION_REPETITION_PENALTY


# 添加别名以兼容training_projector.py
VIT_MODEL_NAME = ModelSettings.VIT_MODEL_NAME
LLAMA_MODEL_NAME = ModelSettings.LLAMA_MODEL_NAME
DEVICE = RuntimeSettings.DEVICE
BATCH_SIZE = TrainingSettings.BATCH_SIZE
EPOCH = TrainingSettings.EPOCH
LEARNING_RATE = TrainingSettings.LEARNING_RATE
WEIGHT_DECAY = TrainingSettings.WEIGHT_DECAY
LOG_INTERVAL = TrainingSettings.LOG_INTERVAL
IMAGE_PATH = str(PathSettings.IMAGE_PATH)
TEXT_JSON = str(PathSettings.TEXT_JSON)
IMAGE_TOKEN_DIM = DimensionSettings.IMAGE_TOKEN_DIM
TEXT_TOKEN_DIM = DimensionSettings.TEXT_TOKEN_DIM
OUTPUT_DIM = DimensionSettings.OUTPUT_DIM
MLP_DROPOUT = MLPSettings.MLP_DROPOUT
USE_LAYERNORM = MLPSettings.USE_LAYERNORM
MLP_ACTIVATION = MLPSettings.MLP_ACTIVATION

# 添加额外的别名以兼容encoder_ViT.py及其他文件
VIT_PRETRAINED = ModelSettings.VIT_PRETRAINED
VIT_IMAGE_SIZE = ModelSettings.VIT_IMAGE_SIZE
LLAMA_MAX_LENGTH = ModelSettings.LLAMA_MAX_LENGTH
FREEZE_LLM = ModelSettings.FREEZE_LLM
USE_MODEL_CACHE = ModelSettings.USE_MODEL_CACHE
FUSION_TOKEN_DIM = DimensionSettings.FUSION_TOKEN_DIM
WEIGHTS_FILENAME = MLPSettings.WEIGHTS_FILENAME
WEIGHTS_PATH = MLPSettings.WEIGHTS_PATH