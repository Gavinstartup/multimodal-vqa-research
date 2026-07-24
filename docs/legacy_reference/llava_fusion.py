import torch
import logging
from config import DEVICE, AttentionMaskSettings
from mask_utils import create_causal_mask

def llava_fusion(image_features, text_embeddings, attention_mask, position_ids, image_positions):
    """
    LLaVA风格的特征融合 - 将图像特征插入到文本嵌入的指定位置。
    """
    try:
        batch_size, seq_len, embed_dim = text_embeddings.shape
        
        # 获取模型预期的数据类型
        target_dtype = text_embeddings.dtype
        
        # 确保图像特征维度与嵌入维度匹配
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        
        # 确保数据类型一致
        image_features = image_features.to(dtype=target_dtype)
        
        # 创建融合后的嵌入（初始化为文本嵌入的副本）
        fused_embeddings = text_embeddings.clone()
        
        # 为每个批次替换图像标记位置的嵌入
        for b in range(batch_size):
            if not image_positions[b]:  # 如果没有图像标记位置
                continue
                
            # 获取当前批次中的图像标记位置
            positions = image_positions[b]
            
            # 确保有足够的图像特征来替换所有图像标记
            for pos in positions:
                fused_embeddings[b, pos] = image_features[b, 0]
        
        # 对于LLaMA 3.2，使用2D注意力掩码
        # 创建基本的2D掩码，形状为[batch_size, seq_len]
        fused_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=DEVICE)
        
        # 创建更新后的位置ID
        fused_position_ids = position_ids.clone()
        
        logging.info(f"LLaVA风格融合完成，输出形状: {fused_embeddings.shape}，dtype: {fused_embeddings.dtype}")
        
        return fused_embeddings, fused_attention_mask, fused_position_ids
        
    except Exception as e:
        logging.error(f"LLaVA风格融合失败: {str(e)}")
        raise

"""

    这个文档是融合的核心逻辑，主要是将处理过后的图像特征，插入到刚刚刚处理过<image>标签的位置然后重新设置位置信息
    """