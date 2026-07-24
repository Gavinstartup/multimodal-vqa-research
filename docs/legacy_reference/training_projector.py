import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import json
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import glob
from pathlib import Path
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/exp1")
multiprocessing.set_start_method('spawn', force=True)
from config import (
    VIT_MODEL_NAME, LLAMA_MODEL_NAME, DEVICE, Weights_path,
    BATCH_SIZE, EPOCH, LEARNING_RATE, WEIGHT_DECAY, LOG_INTERVAL,
    IMAGE_PATH, TEXT_JSON, IMAGE_TOKEN_DIM, TEXT_TOKEN_DIM, 
    HIDDEN_DIM_1, HIDDEN_DIM_2, HIDDEN_DIM_3, HIDDEN_DIM_4, OUTPUT_DIM, 
    MLP_DROPOUT, USE_LAYERNORM, MLP_ACTIVATION,
    MSE_LOSS_WEIGHT, CONTRASTIVE_LOSS_WEIGHT, GRADIENT_ACCUMULATION_STEPS,
    TrainingSettings
)
from llama_wrapper import load_llama_model, load_llama_tokenizer
from encoder_ViT import load_vit_and_preprocessor, preprocess_image, open_image
from process_text_input import process_text_input
llama_model = load_llama_model()
token_model = load_llama_tokenizer()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_projector.log")
    ]
)

class MLPProjector(nn.Module):
    """
    MLP Projector: Projects image features to text embedding space using a five-layer MLP
    """
    def __init__(self, input_dim=None):
        super(MLPProjector, self).__init__()
        # 使用传入的input_dim或默认的IMAGE_TOKEN_DIM
        self.input_dim = input_dim if input_dim is not None else IMAGE_TOKEN_DIM
        logging.info(f"Initializing 5-layer MLPProjector with input dimension: {self.input_dim}")
        
        # 第一层: 512 -> 768
        self.fc1 = nn.Linear(self.input_dim, HIDDEN_DIM_1)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM_1) if USE_LAYERNORM else nn.Identity()
        self.activation1 = MLP_ACTIVATION()
        self.dropout1 = nn.Dropout(MLP_DROPOUT)
        
        # 第二层: 768 -> 1024
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM_2) if USE_LAYERNORM else nn.Identity()
        self.activation2 = MLP_ACTIVATION()
        self.dropout2 = nn.Dropout(MLP_DROPOUT)
        
        # 第三层: 1024 -> 1536
        self.fc3 = nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3)
        self.ln3 = nn.LayerNorm(HIDDEN_DIM_3) if USE_LAYERNORM else nn.Identity()
        self.activation3 = MLP_ACTIVATION()
        self.dropout3 = nn.Dropout(MLP_DROPOUT)
        
        # 第四层: 1536 -> 1792
        self.fc4 = nn.Linear(HIDDEN_DIM_3, HIDDEN_DIM_4)
        self.ln4 = nn.LayerNorm(HIDDEN_DIM_4) if USE_LAYERNORM else nn.Identity()
        self.activation4 = MLP_ACTIVATION()
        self.dropout4 = nn.Dropout(MLP_DROPOUT)
        
        # 输出层: 1792 -> 2048
        self.fc5 = nn.Linear(HIDDEN_DIM_4, OUTPUT_DIM)
        self.ln5 = nn.LayerNorm(OUTPUT_DIM) if USE_LAYERNORM else nn.Identity()
    
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

class ImageTextDataset(torch.utils.data.Dataset):
    """
    Dataset for image-text pairs
    """
    def __init__(self, image_path, text_json_path, vit_model, preprocess_fn, token_model, llama_model):
        self.image_path = image_path
        self.vit_model = vit_model
        self.preprocess_fn = preprocess_fn
        self.token_model = token_model
        self.llama_model = llama_model
        # 检查一个样本以确定实际的图像特征维度
        self.actual_img_feature_dim = None
        
        # 加载文本JSON
        try:
            with open(text_json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logging.info(f"Loaded {len(self.data)} image-text pairs from {text_json_path}")
        except Exception as e:
            logging.error(f"Failed to load text JSON: {str(e)}")
            raise
        
        # 验证数据格式
        self._validate_data()
        
        # 检测实际的图像特征维度
        self._detect_image_feature_dim()
    
    def _detect_image_feature_dim(self):
        """检测实际的图像特征维度"""
        if len(self.data) > 0:
            try:
                # 获取第一个图像
                image_file = os.path.join(self.image_path, self.data[0]['image'])
                image = open_image(image_file)
                image_tensor = preprocess_image(image, self.preprocess_fn)
                
                # 提取图像特征
                with torch.no_grad():
                    image_features = self.vit_model.encode_image(image_tensor)
                
                # 记录实际维度
                self.actual_img_feature_dim = image_features.shape[1]
                logging.info(f"Detected actual image feature dimension: {self.actual_img_feature_dim}")
                
                # 检查与配置是否一致
                if self.actual_img_feature_dim != IMAGE_TOKEN_DIM:
                    logging.warning(f"Actual image feature dimension ({self.actual_img_feature_dim}) "
                                   f"does not match configured IMAGE_TOKEN_DIM ({IMAGE_TOKEN_DIM})")
            except Exception as e:
                logging.error(f"Failed to detect image feature dimension: {str(e)}")
    
    def _validate_data(self):
        """验证数据格式"""
        if not isinstance(self.data, list):
            raise ValueError("Data should be a list of image-text pairs")
        
        valid_items = []
        for idx, item in enumerate(self.data):
            if not isinstance(item, dict) or 'image' not in item or 'text' not in item:
                logging.warning(f"Item at index {idx} has invalid format, skipping")
                continue
            
            # 检查图像文件是否存在
            img_path = os.path.join(self.image_path, item['image'])
            if not os.path.exists(img_path):
                logging.warning(f"Image {img_path} does not exist, skipping")
                continue
                
            valid_items.append(item)
        
        # 更新数据为有效的条目
        if len(valid_items) < len(self.data):
            logging.info(f"Filtered out {len(self.data) - len(valid_items)} invalid items")
            self.data = valid_items
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = os.path.join(self.image_path, item['image'])
        text = item['text']
        
        try:
            # 处理图像
            image = open_image(image_file)
            image_tensor = preprocess_image(image, self.preprocess_fn)
            
            # 提取图像特征
            with torch.no_grad():
                image_features = self.vit_model.encode_image(image_tensor).to(DEVICE)
                
                # 为调试打印出当前样本的特征维度
                if idx == 0 or random.random() < 0.01:  # 打印第一个样本和随机1%的样本
                    logging.debug(f"Sample {idx} image features shape: {image_features.shape}")
            
            # 处理文本
            text_embeddings, input_ids, attention_mask, position_ids = process_text_input(self.token_model, self.llama_model, text)
            
            # 改进: 使用文本嵌入的平均值作为目标
            # 计算有效的token数量（非padding）
            seq_length = attention_mask.sum(dim=1).item()
            
            # 获取有效token的平均嵌入
            valid_embeddings = text_embeddings[0, :seq_length, :]
            text_embedding = valid_embeddings.mean(dim=0).to(DEVICE)
            
            # 也保存第一个token的嵌入用于比较
            first_token_embedding = text_embeddings[0, 0, :].to(DEVICE)
            
            # 预处理问题和回答（从"问题: X 答案: Y"格式）
            question_answer = self._extract_qa(text)
            
            return {
                'image_features': image_features.squeeze(0),  # 移除批次维度
                'text_embedding': text_embedding,  # 平均嵌入
                'first_token_embedding': first_token_embedding,  # 第一个token嵌入（用于比较）
                'image_path': image_file,
                'text': text,
                'question': question_answer['question'] if question_answer else "",
                'answer': question_answer['answer'] if question_answer else ""
            }
        except Exception as e:
            logging.error(f"Error processing item {idx} (image: {image_file}): {str(e)}")
            # 返回一个简单的默认项，避免中断训练
            if idx > 0:
                # 如果不是第一个样本，可以递归获取上一个样本
                return self.__getitem__(max(0, idx-1))
            else:
                # 如果是第一个样本出错，则需要抛出异常
                raise
    
    def _extract_qa(self, text):
        """从文本中提取问题和答案"""
        try:
            if "问题:" in text and "答案:" in text:
                parts = text.split("答案:")
                if len(parts) >= 2:
                    question_part = parts[0].strip()
                    if "问题:" in question_part:
                        question = question_part.split("问题:")[1].strip()
                    else:
                        question = question_part
                    
                    answer = parts[1].strip()
                    return {'question': question, 'answer': answer}
            return None
        except:
            return None

def collate_fn(batch):
    """数据加载器的自定义整理函数"""
    image_features = torch.stack([item['image_features'] for item in batch])
    text_embeddings = torch.stack([item['text_embedding'] for item in batch])
    first_token_embeddings = torch.stack([item['first_token_embedding'] for item in batch])
    
    return {
        'image_features': image_features,
        'text_embeddings': text_embeddings,
        'first_token_embeddings': first_token_embeddings,
        'image_paths': [item['image_path'] for item in batch],
        'texts': [item['text'] for item in batch],
        'questions': [item.get('question', "") for item in batch],
        'answers': [item.get('answer', "") for item in batch]
    }

class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    def __init__(self, temperature=0.1):  # 使用更高的温度参数
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, image_embeddings, text_embeddings):
        # 确保数据类型一致 - 修复数据类型不匹配问题
        if image_embeddings.dtype != text_embeddings.dtype:
            logging.info(f"Converting tensors to compatible dtype: image={image_embeddings.dtype}, text={text_embeddings.dtype}")
            # 通常使用两者中精度较高的类型
            if torch.finfo(image_embeddings.dtype).bits > torch.finfo(text_embeddings.dtype).bits:
                target_dtype = image_embeddings.dtype
            else:
                target_dtype = text_embeddings.dtype
            
            image_embeddings = image_embeddings.to(target_dtype)
            text_embeddings = text_embeddings.to(target_dtype)
        
        # 计算批次内所有图像-文本对的相似度
        batch_size = image_embeddings.size(0)
        
        # 标准化嵌入
        image_embeddings = nn.functional.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        # 创建标签: 对角线为正例(相似的图像-文本对)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # 计算图像到文本和文本到图像的损失
        i2t_loss = self.cross_entropy(similarity_matrix, labels)
        t2i_loss = self.cross_entropy(similarity_matrix.t(), labels)
        
        # 综合损失 (双向)
        loss = (i2t_loss + t2i_loss) / 2
        return loss

def train_model(model, train_loader, optimizer, criterion, contrastive_criterion, device, epoch, log_interval, writer):
    """训练模型一个轮次"""
    model.train()
    running_loss = 0.0
    running_mse_loss = 0.0
    running_contrastive_loss = 0.0
    
    # 梯度累积参数
    accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    optimizer.zero_grad()  # 开始前清零梯度
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}") as pbar:
        for batch_idx, batch in enumerate(pbar):
            # 获取输入数据并确保数据类型匹配 - 修复数据类型不匹配问题
            target_dtype = model.fc1.weight.dtype
            image_features = batch['image_features'].to(device, dtype=target_dtype)
            text_embeddings = batch['text_embeddings'].to(device, dtype=target_dtype)
            
            # 为了调试，打印批次的形状
            if batch_idx == 0:
                logging.debug(f"Batch image features shape: {image_features.shape}, dtype: {image_features.dtype}")
                logging.debug(f"Batch text embeddings shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")
            
            # 前向传播
            outputs = model(image_features)
            
            # 计算MSE损失（与平均文本嵌入的对齐）
            if isinstance(criterion, nn.CosineEmbeddingLoss):
                # 为每个样本创建全1标签（表示正相似度）
                target = torch.ones(image_features.size(0), device=device)
                mse_loss = criterion(outputs, text_embeddings, target)
            else:
                mse_loss = criterion(outputs, text_embeddings)
            
            # 计算对比损失
            contrastive_loss = contrastive_criterion(outputs, text_embeddings)
            
            # 组合损失 - 使用加权组合
            loss = MSE_LOSS_WEIGHT * mse_loss + CONTRASTIVE_LOSS_WEIGHT * contrastive_loss
            
            # 根据梯度累积步数缩放损失
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 更新统计信息 (按原始未缩放的损失)
            running_loss += (MSE_LOSS_WEIGHT * mse_loss.item() + CONTRASTIVE_LOSS_WEIGHT * contrastive_loss.item())
            running_mse_loss += mse_loss.item()
            running_contrastive_loss += contrastive_loss.item()
            
            # 梯度累积 - 每accumulation_steps步执行一次优化器步骤
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新进度条
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_mse_loss = running_mse_loss / log_interval
                avg_contrastive_loss = running_contrastive_loss / log_interval
                
                pbar.set_postfix(
                    loss=avg_loss, 
                    mse=avg_mse_loss, 
                    contrastive=avg_contrastive_loss
                )
                
                logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, "
                           f"Loss: {avg_loss:.4f}, MSE: {avg_mse_loss:.4f}, "
                           f"Contrastive: {avg_contrastive_loss:.4f}")
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), global_step)
                
                running_loss = 0.0
                running_mse_loss = 0.0
                running_contrastive_loss = 0.0

def validate_model(model, val_loader, criterion, contrastive_criterion, device, writer):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_contrastive_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            # 获取输入数据并确保数据类型匹配
            target_dtype = model.fc1.weight.dtype
            image_features = batch['image_features'].to(device, dtype=target_dtype)
            text_embeddings = batch['text_embeddings'].to(device, dtype=target_dtype)
            
            # 前向传播
            outputs = model(image_features)
            
            # 计算MSE损失
            if isinstance(criterion, nn.CosineEmbeddingLoss):
                target = torch.ones(image_features.size(0), device=device)
                mse_loss = criterion(outputs, text_embeddings, target)
            else:
                mse_loss = criterion(outputs, text_embeddings)
            
            # 计算对比损失
            contrastive_loss = contrastive_criterion(outputs, text_embeddings)
            
            # 组合损失 - 使用加权组合
            loss = MSE_LOSS_WEIGHT * mse_loss + CONTRASTIVE_LOSS_WEIGHT * contrastive_loss
            
            # 更新统计信息
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            writer.add_scalar('Loss/val', loss.item(), batch_idx)
    
    # 计算平均损失
    num_batches = len(val_loader)
    if num_batches > 0:  # 防止除零错误
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
    else:
        avg_loss = total_loss
        avg_mse_loss = total_mse_loss
        avg_contrastive_loss = total_contrastive_loss
    
    logging.info(f"Validation Loss: {avg_loss:.4f}, MSE: {avg_mse_loss:.4f}, "
               f"Contrastive: {avg_contrastive_loss:.4f}")
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Train MLP Projector for multimodal fusion")
    parser.add_argument("--image_path", type=str, default=IMAGE_PATH, help="Path to image directory")
    parser.add_argument("--text_json", type=str, default=TEXT_JSON, help="Path to text JSON file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=EPOCH, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--log_interval", type=int, default=LOG_INTERVAL, help="Log interval")
    parser.add_argument("--save_dir", type=str, default=os.path.dirname(Weights_path), help="Directory to save model weights")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--contrastive_temp", type=float, default=TrainingSettings.TEMPERATURE, help="Temperature for contrastive loss")
    parser.add_argument("--early_stopping", type=int, default=TrainingSettings.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--grad_accum_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--mse_weight", type=float, default=MSE_LOSS_WEIGHT, help="Weight for MSE/cosine loss")
    parser.add_argument("--contrastive_weight", type=float, default=CONTRASTIVE_LOSS_WEIGHT, help="Weight for contrastive loss")
    
    args = parser.parse_args()
    global token_model, llama_model
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 创建保存目录（如果不存在）
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载ViT模型和预处理器
    logging.info("Loading ViT model and preprocessor...")
    vit_model, preprocess = load_vit_and_preprocessor()
    
    # 初始化数据集
    logging.info("Creating dataset...")
    full_dataset = ImageTextDataset(
        image_path=args.image_path,
        text_json_path=args.text_json,
        vit_model=vit_model,
        preprocess_fn=preprocess,
        token_model=token_model,
        llama_model = llama_model
    )
    
    # 获取检测到的图像特征维度
    actual_img_feature_dim = full_dataset.actual_img_feature_dim
    
    # 将数据集拆分为训练集和验证集
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=True
    )
    
    logging.info(f"Dataset created with {train_size} training samples and {val_size} validation samples")
    logging.info(f"Using gradient accumulation with {args.grad_accum_steps} steps (effective batch size: {args.batch_size * args.grad_accum_steps})")
    
    # 使用检测到的特征维度初始化模型
    model = MLPProjector(input_dim=actual_img_feature_dim).to(DEVICE)
    logging.info(f"Initialized MLPProjector with detected input dimension: {actual_img_feature_dim}")
    
    # 初始化损失函数
    criterion = nn.CosineEmbeddingLoss()
    contrastive_criterion = ContrastiveLoss(temperature=args.contrastive_temp)
    
    logging.info(f"Using loss functions: {criterion.__class__.__name__} (weight: {args.mse_weight}) and "
                f"ContrastiveLoss (weight: {args.contrastive_weight}, temp: {args.contrastive_temp})")
    
    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=TrainingSettings.LR_FACTOR,
        patience=TrainingSettings.LR_PATIENCE, 
        verbose=True
    )
    
    # 训练模型
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # 添加异常处理，避免因小错误终止整个训练
    try:
        for epoch in range(args.epochs):
            # 训练
            try:
                train_model(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    contrastive_criterion=contrastive_criterion,
                    device=DEVICE,
                    epoch=epoch,
                    log_interval=args.log_interval,
                    writer=writer
                )
            except Exception as e:
                logging.error(f"Error during training epoch {epoch+1}: {str(e)}")
                continue
            
            # 验证
            try:
                val_loss = validate_model(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    contrastive_criterion=contrastive_criterion,
                    device=DEVICE, 
                    writer = writer
                )
            except Exception as e:
                logging.error(f"Error during validation for epoch {epoch+1}: {str(e)}")
                val_loss = float('inf')  # 使用无穷大作为默认验证损失
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存模型（如果是目前最好的）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.save_dir, f"mlp_projector_best.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_dim': actual_img_feature_dim,
                    'hidden_dim_1': HIDDEN_DIM_1,
                    'hidden_dim_2': HIDDEN_DIM_2,
                    'hidden_dim_3': HIDDEN_DIM_3,
                    'hidden_dim_4': HIDDEN_DIM_4,
                    'output_dim': OUTPUT_DIM,
                    'val_loss': best_val_loss,
                    'epoch': epoch + 1,
                    'architecture': 'five_layer_mlp'
                }, save_path)
                logging.info(f"Model saved to {save_path} with validation loss {val_loss:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                logging.info(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{args.early_stopping}")
                
                if early_stopping_counter >= args.early_stopping:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # 保存检查点
            checkpoint_path = os.path.join(args.save_dir, f"mlp_projector_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'input_dim': actual_img_feature_dim,
                'hidden_dim_1': HIDDEN_DIM_1,
                'hidden_dim_2': HIDDEN_DIM_2,
                'hidden_dim_3': HIDDEN_DIM_3, 
                'hidden_dim_4': HIDDEN_DIM_4,
                'output_dim': OUTPUT_DIM,
                'architecture': 'five_layer_mlp'
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")

    except Exception as e:
        logging.error(f"Unexpected error during training: {str(e)}")
    finally:
        # 保存最终模型（无论训练如何结束）
        try:
            final_path = os.path.join(args.save_dir, "mlp_projector_final.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': actual_img_feature_dim,
                'hidden_dim_1': HIDDEN_DIM_1,
                'hidden_dim_2': HIDDEN_DIM_2,
                'hidden_dim_3': HIDDEN_DIM_3, 
                'hidden_dim_4': HIDDEN_DIM_4,
                'output_dim': OUTPUT_DIM,
                'architecture': 'five_layer_mlp'
            }, final_path)
            logging.info(f"Final model saved to {final_path}")
        except Exception as e:
            logging.error(f"Error saving final model: {str(e)}")
    
    logging.info("Training process completed!")

if __name__ == "__main__":
    main()