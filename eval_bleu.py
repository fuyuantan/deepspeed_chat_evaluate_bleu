import argparse
import logging
import torch
import math
import os
from tqdm import tqdm
# 新增 Imports for BLEU
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoConfig, # 可选，用于获取模型配置
)

# 假设这些来自你的环境并且可用
from dschat.utils.model.model_utils import create_hf_model
from dschat.utils.utils import load_hf_tokenizer
from deepspeed import get_accelerator

import evaluate

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    # 更新描述
    parser = argparse.ArgumentParser(description="Evaluate BLEU score for baseline and finetuned models using MS MARCO")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path or identifier of the BASELINE model (also used for tokenizer)",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path or identifier of the FINETUNED model",
        required=True,
    )

    # --- Arguments for BLEU Evaluation (using MS MARCO defaults) ---
    parser.add_argument(
        "--generation_dataset_name_or_path",
        type=str,
        default="ms_marco", # 默认使用 MS MARCO
        help="Name or path to the dataset for generation/BLEU evaluation.",
    )
    parser.add_argument(
        "--generation_dataset_config_name",
        type=str,
        default="v1.1", # MS MARCO v1.1 常用配置, 如需更改请指定
        help="Configuration name for the generation dataset (e.g., 'v1.1' for MS MARCO).",
    )
    parser.add_argument(
        "--generation_dataset_split",
        type=str,
        default="validation", # MS MARCO 通常用 validation 或 dev
        help="Which split of the generation dataset to use (e.g., 'validation', 'test', 'dev').",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="query", # MS MARCO 问题列名
        help="Column name for the input prompt/query in the generation dataset.",
    )
    parser.add_argument(
        "--reference_column",
        type=str,
        default="answers", # MS MARCO 答案列名 (包含一个答案列表)
        help="Column name for the reference answers in the generation dataset.",
    )
    parser.add_argument(
        "--max_generation_samples",
        type=int,
        default=1000, # 默认评估一部分样本以提高速度
        help="Maximum number of samples to evaluate for BLEU. Set to -1 or None for all samples.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=128, # 限制输入 Prompt 的最大长度
        help="Maximum length of the input prompt tokens for truncation.",
    )

    # --- Generation Parameters ---
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64, # 生成答案的最大长度
        help="Maximum new tokens to generate for BLEU evaluation.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1, # 默认 greedy search
        help="Number of beams for beam search generation.",
    )
    parser.add_argument(
        "--do_sample",
        action='store_true',
        help="Whether to use sampling during generation; overrides num_beams > 1 if set.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter.",
    )

    # --- General Arguments ---
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16, # 适用于生成的批处理大小
        help="Batch size for BLEU generation evaluation.",
    )
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer.")

    args = parser.parse_args()
    return args

# =========================================================================
# BLEU Evaluation Functions (New)
# =========================================================================
def calculate_bleu_for_model(model, tokenizer, device, args):
    """ Calculates BLEU score for a given model using arguments. """
    model.eval()
    logger.info(f"--- Starting BLEU Calculation ---")
    logger.info(f"Loading BLEU dataset '{args.generation_dataset_name_or_path}' config '{args.generation_dataset_config_name}' split '{args.generation_dataset_split}'...")

    try:
        # 加载数据集
        dataset_args = {}
        if args.generation_dataset_config_name:
            dataset_args['name'] = args.generation_dataset_config_name
        gen_dataset = load_dataset(args.generation_dataset_name_or_path, split=args.generation_dataset_split, **dataset_args)

    except Exception as e:
        logger.error(f"Failed to load generation dataset: {e}", exc_info=True)
        raise

    # 验证列名
    if args.prompt_column not in gen_dataset.column_names:
        raise ValueError(f"Prompt column '{args.prompt_column}' not found in dataset. Available: {gen_dataset.column_names}")
    if args.reference_column not in gen_dataset.column_names:
        raise ValueError(f"Reference column '{args.reference_column}' not found in dataset. Available: {gen_dataset.column_names}")

    # 限制样本数量
    num_samples = len(gen_dataset)
    if args.max_generation_samples is not None and args.max_generation_samples > 0 and args.max_generation_samples < num_samples:
        logger.info(f"Evaluating BLEU on a subset of {args.max_generation_samples} samples.")
        gen_dataset = gen_dataset.select(range(args.max_generation_samples))
    else:
         logger.info(f"Evaluating BLEU on all {num_samples} samples.")


    # 加载 BLEU 指标 (使用 sacrebleu)
    try:
        bleu_metric = evaluate.load("sacrebleu")
    except Exception as e:
        logger.error(f"Failed to load sacrebleu metric: {e}. Make sure 'sacrebleu' and 'datasets' are installed (`pip install sacrebleu datasets`).")
        raise

    predictions = []
    references_for_bleu = []

    # 定义整理函数 (collate_fn)
    def collate_fn(batch):
        prompts = [item[args.prompt_column] for item in batch]
        # MS MARCO 的 'answers' 列本身就是答案字符串列表
        refs_list_of_lists = [item[args.reference_column] for item in batch]

        # 过滤掉没有有效参考答案的样本
        valid_indices = []
        processed_refs_for_sacrebleu = []
        for i, ref_list in enumerate(refs_list_of_lists):
            if isinstance(ref_list, list) and any(isinstance(r, str) and r.strip() for r in ref_list):
                # 清理内部列表，去除空字符串
                cleaned_ref_list = [r for r in ref_list if isinstance(r, str) and r.strip()]
                if cleaned_ref_list: # 确保清理后列表不为空
                    # 只取第一个有效的参考答案，并放入一个新的列表
                    processed_refs_for_sacrebleu.append([cleaned_ref_list[0]])
                    valid_indices.append(i)
                else:
                    logger.debug(f"Sample {i} skipped, references became empty after cleaning: {ref_list}")
            else:
                logger.debug(f"Sample {i} skipped due to invalid references format or content: {ref_list}")

        # 根据有效参考过滤 prompts
        valid_prompts = [prompts[i] for i in valid_indices]

        if not valid_prompts:
            return None # 如果整个批次都无效，返回 None

        return {"prompts": valid_prompts, "references": processed_refs_for_sacrebleu}

    # 创建 DataLoader
    data_loader = DataLoader(gen_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    logger.info(f"Generating predictions for BLEU evaluation (Batch Size: {args.eval_batch_size})...")
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "top_k": args.top_k if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    logger.info(f"Generation config: {generation_config}")

    # 开始生成
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating BLEU"):
            if batch is None: # 跳过无效批次
                continue

            prompts = batch["prompts"]
            batch_references = batch["references"]

            # Tokenize prompts (带 padding 和 truncation)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_prompt_length).to(device)

            # 生成答案 ID
            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask, # 传入 attention_mask
                **generation_config
            )

            # 解码生成的 token (跳过 prompt 部分和特殊 token)
            prompt_lengths = inputs.input_ids.shape[1]
            decoded_preds = tokenizer.batch_decode(generated_ids[:, prompt_lengths:], skip_special_tokens=True)

            # 清理生成的文本（例如去除首尾空格）
            cleaned_preds = [pred.strip() for pred in decoded_preds]

            predictions.extend(cleaned_preds)
            references_for_bleu.extend(batch_references)

            # 清理显存
            del inputs, generated_ids
            if device.type == 'cuda': torch.cuda.empty_cache()

    # 计算 BLEU 分数
    if not predictions:
        logger.error("No valid predictions were generated. Cannot calculate BLEU.")
        return 0.0

    logger.info(f"Calculating BLEU score using sacrebleu on {len(predictions)} samples...")
    try:
        # sacrebleu 需要 predictions 是字符串列表, references 是列表的列表
        results = bleu_metric.compute(predictions=predictions, references=references_for_bleu)
        bleu_score = results["score"]
        logger.info(f"BLEU Score: {bleu_score:.4f}")
        logger.info(f"Sacrebleu details: {results}") # 打印详细信息
        return bleu_score
    except Exception as e:
        logger.error(f"Failed to compute BLEU score: {e}", exc_info=True)
        # 打印一些样本帮助调试
        idx_to_log = min(5, len(predictions))
        for i in range(idx_to_log):
            logger.error(f"  Sample {i} Prediction: {predictions[i]}")
            logger.error(f"  Sample {i} Reference: {references_for_bleu[i]}")
        return 0.0


def eval_BLEU(args, model_baseline, model_fintuned, tokenizer, device):
    """ Evaluates and prints BLEU score for both models. """
    # 检查依赖
    try:
        logger.debug("正在检查 'datasets' 库...")
        from datasets import load_dataset # 需要 datasets 来加载数据
        logger.debug("'datasets' 导入成功。")

        logger.debug("正在检查 'evaluate' 库...")
        import evaluate # 需要 evaluate 来加载指标
        logger.debug("'evaluate' 导入成功。")

        logger.debug("正在检查 'sacrebleu' 库 (可能是隐式依赖)...")
        import sacrebleu # 明确检查 sacrebleu 是否能导入
        logger.debug("'sacrebleu' 导入成功。")

        logger.debug("尝试通过 evaluate.load() 加载 'sacrebleu' 指标...")
        # 这行会实际尝试加载指标脚本
        evaluate.load("sacrebleu")
        logger.debug("通过 evaluate 加载 'sacrebleu' 指标成功。")

    except ImportError as e:
        logger.error(f"依赖检查失败 (ImportError): {e}。可能是 datasets, evaluate 或 sacrebleu 库缺失或不在当前 Python 环境中。")
        logger.error("请确保已正确安装 'datasets', 'evaluate', 和 'sacrebleu': pip install datasets evaluate sacrebleu")
        return # 如果依赖缺失则退出
    except Exception as e:
        # 捕捉加载指标时可能发生的其他错误 (例如网络问题、缓存问题)
        logger.error(f"检查/加载依赖时发生意外错误: {e}", exc_info=True) # 打印完整错误信息
        logger.error("请确保已正确安装 'datasets', 'evaluate', 和 'sacrebleu': pip install datasets evaluate sacrebleu")
        return

    logger.info("=" * 30)
    logger.info("Starting BLEU Evaluation")
    logger.info(f"Dataset: {args.generation_dataset_name_or_path}, Config: {args.generation_dataset_config_name}, Split: {args.generation_dataset_split}")
    logger.info(f"Prompt: '{args.prompt_column}', Reference: '{args.reference_column}'")
    logger.info("=" * 30)

    logger.info("Evaluating Baseline Model BLEU")
    bleu_baseline = calculate_bleu_for_model(model_baseline, tokenizer, device, args)
    logger.info(f"\nBaseline Model BLEU Score: {bleu_baseline:.4f}\n")

    logger.info("Evaluating Fine-tuned Model BLEU")
    bleu_finetuned = calculate_bleu_for_model(model_fintuned, tokenizer, device, args)
    logger.info(f"\nFine-tuned Model BLEU Score: {bleu_finetuned:.4f}\n")

    return bleu_baseline, bleu_finetuned

# =========================================================================
# Main Function
# =========================================================================
def main():
    args = parse_args()

    # 确定设备
    if torch.cuda.is_available():
        device = torch.device(get_accelerator().device_name(0))
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")

    # --- 加载 Tokenizer ---
    logger.info(f"Loading tokenizer from {args.model_name_or_path_baseline}...")
    additional_special_tokens = ["<|endoftext|>"] if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    if tokenizer.pad_token is None:
        # 对于生成任务，pad token 必须设置
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(f"Tokenizer lacks a pad token. Setting pad_token to eos_token ({tokenizer.eos_token}). This is crucial for generation.")
    logger.info(f"Tokenizer Pad Token ID: {tokenizer.pad_token_id}, EOS Token ID: {tokenizer.eos_token_id}")


    # --- 加载模型 ---
    logger.info(f"Loading baseline model from {args.model_name_or_path_baseline}...")
    # 尝试使用半精度加载
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    try:
        model_baseline = create_hf_model(AutoModelForCausalLM, args.model_name_or_path_baseline, tokenizer, ds_config=None, torch_dtype=dtype)
        logger.info(f"Loaded baseline model with dtype: {dtype}")
    except Exception as e:
         logger.warning(f"Failed to load baseline model with dtype {dtype}: {e}. Trying default dtype.")
         model_baseline = create_hf_model(AutoModelForCausalLM, args.model_name_or_path_baseline, tokenizer, ds_config=None)

    logger.info(f"Loading fine-tuned model from {args.model_name_or_path_finetune}...")
    try:
        model_fintuned = create_hf_model(AutoModelForCausalLM, args.model_name_or_path_finetune, tokenizer, ds_config=None, torch_dtype=dtype)
        logger.info(f"Loaded fine-tuned model with dtype: {dtype}")
    except Exception as e:
         logger.warning(f"Failed to load fine-tuned model with dtype {dtype}: {e}. Trying default dtype.")
         model_fintuned = create_hf_model(AutoModelForCausalLM, args.model_name_or_path_finetune, tokenizer, ds_config=None)

    # --- 移动模型到设备 ---
    logger.info(f"Ensuring models are on device {device}")
    model_baseline.to(device)
    model_fintuned.to(device)

    # --- 运行 BLEU 评估 ---
    eval_BLEU(args, model_baseline, model_fintuned, tokenizer, device)

    logger.info("BLEU evaluation finished.")


if __name__ == "__main__":
    main()