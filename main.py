from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import pipeline
import evaluate
import numpy as np


# 步骤一：加载模型和分词器
print("Step1: Loading model and tokenizer...")
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 步骤二：数据预处理
print("Step2: Loading and preprocessing dataset...")
# 假设你的CPI数据存储在名为your_cpi_data.jsonl的文件中
dataset = load_dataset('json', data_files='langchain-handbook.jsonl')
# 划分训练集和验证集，这里简单按9:1划分
dataset = dataset["train"].train_test_split(test_size=0.1)

def preprocess_function(examples):
    inputs = [f"问题：{q} 答案：" for q in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["answer"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 步骤三：配置LoRA
print("Step3: Configuring LoRA...")
# 不同的模型，注意力模块命名不相同也就是target_modules参数不同
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 步骤四：设置训练参数
# 在设置训练的时候save strategy和evaluation_strategy都设置为epoch
# 这样每个epoch都会保存模型和评估模型
# 这样可以避免每个step都保存模型和评估模型，节省时间
print("Step4: Setting training arguments...")
training_args = TrainingArguments(
    output_dir="fine_tuned_gpt2",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",  # 每个epoch保存一次模型
    eval_strategy="epoch",  # 每个epoch进行一次评估
    load_best_model_at_end=True
)

# 步骤五：定义评估指标
print("Step5: Defining evaluation metrics...")
metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # 将预测结果和标签转换为文本
    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 计算BLEU分数
    bleu_results = metric_bleu.compute(predictions=[p.split() for p in pred_texts], references=[[l.split()] for l in label_texts])
    # 计算ROUGE分数
    rouge_results = metric_rouge.compute(predictions=pred_texts, references=label_texts)

    return {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"].mid.fmeasure,
        "rouge2": rouge_results["rouge2"].mid.fmeasure,
        "rougeL": rouge_results["rougeL"].mid.fmeasure
    }

# 步骤六：初始化Trainer并训练
print("Step6: Initializing Trainer and training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()

# 步骤七：保存微调后的模型
print("Step7: Saving the fine-tuned model...")
model.save_pretrained("fine_tuned_gpt2")

# 步骤八：模型部署和使用
print("Step8: Deploying and using the model...")
model = AutoModelForCausalLM.from_pretrained("fine_tuned_gpt2")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_gpt2")

qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
question = "软件的卸载步骤是什么？"
input_text = f"问题：{question} 答案："
answer = qa_pipeline(input_text, max_length=200, num_return_sequences=1)[0]['generated_text']
print(answer.replace(input_text, ""))
    