The eval_bleu.py aims to evaluate **bleu scores** of baseline model and finetune model from [https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/ ](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

1.Add eval_bleu.py to DeepSpeedExamples/applications/DeepSpeed-Chat/. You should cloned [the](https://github.com/deepspeedai/DeepSpeedExamples.git) before.
```
cd DeepSpeedExamples/applications/DeepSpeed-Chat/
git clone https://github.com/fuyuantan/deepspeed_chat_evaluate_bleu.git
```
2.Install dependencies
```
pip install datasets sacrebleu torch transformers tqdm deepspeed accelerate
```

3.Run
```
python eval_bleu.py --model_name_or_path_baseline facebook/opt-1.3b --model_name_or_path_finetune /root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/output/actor/ --generation_dataset_config_name v1.1 --generation_dataset_split validation --max_new_tokens 64 --eval_batch_size 32 --num_beams 4 --max_generation_samples 2000 --do_sample --top_k 50 --top_p 0.95
```

Directory structure:
![微信截图_20250504233700](https://github.com/user-attachments/assets/5198a749-75c6-4a60-8970-285b5aeec17c)
