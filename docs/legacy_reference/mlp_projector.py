import torch
import logging
from config import HIDDEN_DIM_1, HIDDEN_DIM_2, HIDDEN_DIM_3, HIDDEN_DIM_4, OUTPUT_DIM, MLP_DROPOUT, MLP_ACTIVATION, USE_LAYERNORM, DEVICE, IMAGE_TOKEN_DIM, Weights_path
import os
import glob

def mlp_projector(image_tensor, weights_path=None):
    """
    Project image features to text embedding space using a five-layer MLP.
    
    Args:
        image_tensor (torch.Tensor): Image features from ViT encoder
        weights_path (str, optional): Path to MLP weights. If None, uses path from config.
        
    Returns:
        torch.Tensor: Projected image features with matching dimension to text embeddings
    """
    try:
        # Use default weights path from config if not provided
        if weights_path is None:
            weights_path = Weights_path
            
        # 如果weights_path是目录，尝试在该目录中找到.pt文件
        if os.path.isdir(weights_path):
            # 首先尝试使用 mlp_projector_final.pt
            final_weights = os.path.join(weights_path, "mlp_projector_final.pt")
            if os.path.isfile(final_weights):
                weights_path = final_weights
                logging.info(f"Using weights from: {weights_path}")
            else:
                # 查找目录中的任何.pt文件
                pt_files = glob.glob(os.path.join(weights_path, "*.pt"))
                if pt_files:
                    weights_path = pt_files[0]  # 使用找到的第一个.pt文件
                    logging.info(f"Using weights from: {weights_path}")
                else:
                    raise FileNotFoundError(f"No .pt files found in {weights_path}")
        
        # 确认weights_path是一个文件
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
            
        # 使用与training_projector.py一致的五层MLPProjector
        class MLPProjector(torch.nn.Module):
            def __init__(self, input_dim=IMAGE_TOKEN_DIM):
                super(MLPProjector, self).__init__()
                self.input_dim = input_dim
                
                # 第一层: 512 -> 768
                self.fc1 = torch.nn.Linear(input_dim, HIDDEN_DIM_1)
                self.ln1 = torch.nn.LayerNorm(HIDDEN_DIM_1) if USE_LAYERNORM else torch.nn.Identity()
                self.activation1 = MLP_ACTIVATION()
                self.dropout1 = torch.nn.Dropout(MLP_DROPOUT)
                
                # 第二层: 768 -> 1024
                self.fc2 = torch.nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
                self.ln2 = torch.nn.LayerNorm(HIDDEN_DIM_2) if USE_LAYERNORM else torch.nn.Identity()
                self.activation2 = MLP_ACTIVATION()
                self.dropout2 = torch.nn.Dropout(MLP_DROPOUT)
                
                # 第三层: 1024 -> 1536
                self.fc3 = torch.nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3)
                self.ln3 = torch.nn.LayerNorm(HIDDEN_DIM_3) if USE_LAYERNORM else torch.nn.Identity()
                self.activation3 = MLP_ACTIVATION()
                self.dropout3 = torch.nn.Dropout(MLP_DROPOUT)
                
                # 第四层: 1536 -> 1792
                self.fc4 = torch.nn.Linear(HIDDEN_DIM_3, HIDDEN_DIM_4)
                self.ln4 = torch.nn.LayerNorm(HIDDEN_DIM_4) if USE_LAYERNORM else torch.nn.Identity()
                self.activation4 = MLP_ACTIVATION()
                self.dropout4 = torch.nn.Dropout(MLP_DROPOUT)
                
                # 输出层: 1792 -> 2048
                self.fc5 = torch.nn.Linear(HIDDEN_DIM_4, OUTPUT_DIM)
                self.ln5 = torch.nn.LayerNorm(OUTPUT_DIM) if USE_LAYERNORM else torch.nn.Identity()
            
            def forward(self, x):
                # 第一层
                x = self.fc1(x)
                x = self.ln1(x)
                x = self.activation1(x)
                x = self.dropout1(x)
                
                # 第二层
                x = self.fc2(x)
                x = self.ln2(x)
                x = self.activation2(x)
                x = self.dropout2(x)
                
                # 第三层
                x = self.fc3(x)
                x = self.ln3(x)
                x = self.activation3(x)
                x = self.dropout3(x)
                
                # 第四层
                x = self.fc4(x)
                x = self.ln4(x)
                x = self.activation4(x)
                x = self.dropout4(x)
                
                # 输出层
                x = self.fc5(x)
                x = self.ln5(x)
                
                return x
        
        # 创建模型并移动到设备
        model = MLPProjector(input_dim=IMAGE_TOKEN_DIM).to(DEVICE)
        
        # 加载权重
        try:
            weights = torch.load(weights_path, map_location=DEVICE)
            
            # 如果加载的是检查点字典而不是直接的模型权重
            if isinstance(weights, dict):
                if 'model_state_dict' in weights:
                    model.load_state_dict(weights['model_state_dict'])
                elif 'input_dim' in weights:
                    # 检查是否需要调整输入维度
                    input_dim = weights.get('input_dim')
                    if input_dim and input_dim != IMAGE_TOKEN_DIM:
                        logging.info(f"Adjusting model for input dimension: {input_dim}")
                        model = MLPProjector(input_dim=input_dim).to(DEVICE)
                    model.load_state_dict(weights['model_state_dict'] if 'model_state_dict' in weights else weights)
                else:
                    model.load_state_dict(weights)
            else:
                model.load_state_dict(weights)
                
            logging.info(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            logging.error(f"Loading weights failed: {str(e)}")
            raise
        
        # 设置为评估模式
        model.eval()
        
        # 前向传播
        with torch.no_grad():
            output_tensor = model(image_tensor)
            
        return output_tensor
    except Exception as e:
        logging.error(f"MLP projector failed: {str(e)}")
        raise