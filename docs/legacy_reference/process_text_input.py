import torch
import torch.nn as nn
import logging
from config import LLAMA_MODEL_NAME, DEVICE
from llama_wrapper import load_llama_model, load_llama_tokenizer


def process_text_input(token_model, llama_model, text_input):
    """
    Process text input and convert to embeddings for fusion with image features.
    
    Args:
        token_model: LLaMA tokenizer
        llama_model: LLaMA model
        text_input (str): The input text to process
        
    Returns:
        tuple: (text_embeddings, input_ids, attention_mask, position_ids)
            text_embeddings: Embedding vectors from LLaMA's embedding layer
            input_ids: Token IDs from tokenizer
            attention_mask: Mask indicating valid tokens (1) vs padding (0)
            position_ids: Position IDs for tokens
    """
    try:
        if token_model is None:
            token_model = load_llama_tokenizer()
        if llama_model is None:
            llama_model = load_llama_model()
        
        # Process text with tokenizer
        encoded_input = token_model(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,  #这里是截断的意思，超过max_length的token会被截断
            max_length=512,
        ).to(DEVICE)

        input_ids = encoded_input["input_ids"] #这个是token的数字编号
        attention_mask = encoded_input["attention_mask"] #这里是注意力掩码，主要是来区分有效的token和无效的token
        
        # Get embeddings from model
        with torch.no_grad():
            text_embeddings = llama_model.get_input_embeddings()(input_ids)
        
        # Create position IDs
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=DEVICE).unsqueeze(0).repeat(batch_size, 1)
            
        logging.info(f"Text input processed successfully: {text_input[:30]}... translated to shape= {text_embeddings.shape} embedding vector")
        
        return text_embeddings, input_ids, attention_mask, position_ids
    except Exception as e:
        logging.error(f"Error processing text input: {str(e)}")
        raise


def process_text_input_llava(token_model, llama_model, text_input, image_token="<image>"):
    """
    Process text input with image token for LLaVA-style fusion.
    
    Args:
        token_model: LLaMA tokenizer
        llama_model: LLaMA model
        text_input (str): The input text to process
        image_token (str): The token string to mark image position
        
    Returns:
        tuple: (text_embeddings, input_ids, attention_mask, position_ids, image_positions)
            text_embeddings: Embedding vectors from LLaMA's embedding layer
            input_ids: Token IDs from tokenizer
            attention_mask: Mask indicating valid tokens (1) vs padding (0)
            position_ids: Position IDs for tokens
            image_positions: List of positions where image tokens are located
    """
    try:
        if token_model is None:
            token_model = load_llama_tokenizer()
        if llama_model is None:
            llama_model = load_llama_model()
        
        # 确保文本中有图像标记
        if image_token not in text_input:
            text_input = f"{image_token} {text_input}"
        
        # 使用分词器处理文本
        encoded_input = token_model(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        
        # 获取嵌入
        with torch.no_grad():
            text_embeddings = llama_model.get_input_embeddings()(input_ids)
        
        # 创建位置ID
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=DEVICE).unsqueeze(0).repeat(batch_size, 1)
        
        # 查找图像标记的位置
        image_positions = []
        
        # 先编码图像标记以获取其ID
        image_token_ids = token_model.encode(image_token, add_special_tokens=False)
        
        for b in range(batch_size): #############################滑动窗口还不熟练，自己无法使用
            positions = []
            i = 0
            while i < seq_length - len(image_token_ids) + 1:
                # 检查当前位置是否匹配图像标记序列
                match = True
                for j in range(len(image_token_ids)):
                    if i+j >= seq_length or input_ids[b, i+j].item() != image_token_ids[j]:
                        match = False
                        break
                
                if match:
                    # 如果匹配，添加所有图像标记位置
                    positions.extend(range(i, i + len(image_token_ids)))
                    i += len(image_token_ids)
                else:
                    i += 1
            
            image_positions.append(positions)
        
        logging.info(f"Text with image token processed successfully: {text_input[:30]}...")
        logging.info(f"Image token positions: {image_positions}")
        
        return text_embeddings, input_ids, attention_mask, position_ids, image_positions
    except Exception as e:
        logging.error(f"Error processing text input with image token: {str(e)}")
        raise

    """
    这个文件的主要作用是：
    1.处理普通文本提示词，使用llama模型处理用户输入提示词逻辑
    2.处理问题附带<image>标签的提示词，
    3.使用滑动窗口来处理<image>标签位置
    4.返回处理后的文本嵌入，token编号，掩码还有位置id
    """