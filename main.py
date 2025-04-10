# import nltk
# nltk.download('punkt')

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoConfig, GenerationConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import pipeline
import evaluate
import numpy as np

from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 步骤一：加载模型和分词器
model_name = "gpt2"
# 检查配置
config = AutoConfig.from_pretrained(model_name)
# print(config)  # 打印配置查看是否有loss_type
model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 步骤二：数据预处理
# 假设你的CPI数据存储在名为your_cpi_data.jsonl的文件中
dataset = load_dataset('json', data_files='langchain-handbook.jsonl')
# 划分训练集和验证集，这里简单按9:1划分
dataset = dataset["train"].train_test_split(test_size=0.1)


def preprocess_function(examples):
    inputs = [f"question: {q} answer:" for q in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["answer"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 步骤三：配置LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 步骤四：设置训练参数
training_args = TrainingArguments(
    output_dir="fine_tuned_gpt2",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy='epoch',  # 每轮次结束时保存模型
    eval_strategy="epoch",  # 每轮次结束时进行评估
    load_best_model_at_end=True,
    label_names=["labels"],  # 指定标签名称
)

# 步骤五：定义评估指标
metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 解码前替换 labels 中的 -100 为 pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 解码为文本
    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 过滤空文本并去除首尾空格
    pred_texts = [text.strip() for text in pred_texts if text.strip()]
    label_texts = [text.strip() for text in label_texts if text.strip()]
    
    # 处理空数据
    if not pred_texts or not label_texts:
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    # 计算 ROUGE（新版库直接返回数值）
    rouge_results = metric_rouge.compute(
        predictions=pred_texts,
        references=label_texts
    )
    
    # 计算 BLEU
    try:
        bleu_results = metric_bleu.compute(
            predictions=pred_texts,
            references=[[ref] for ref in label_texts],  # 嵌套列表格式
            max_order=4
        )
        bleu_score = bleu_results["bleu"]
    except:
        bleu_score = 0.0
    
    # 兼容 ROUGE 新旧版本
    def get_rouge_score(key):
        value = rouge_results[key]
        return value.mid.fmeasure if hasattr(value, 'mid') else value
    
    return {
        "bleu": bleu_score,
        "rouge1": get_rouge_score("rouge1"),
        "rouge2": get_rouge_score("rouge2"),
        "rougeL": get_rouge_score("rougeL")
    }


# 步骤六：初始化Trainer并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()

# 步骤七：保存微调后的模型
# 保存模型和分词器
model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

import json
documents = []
with open("langchain-handbook.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            answer = data.get("answer", "")
            documents.append(Document(page_content=answer))
        except json.JSONDecodeError:
            print(f"Error decoding line: {line}")

# loader = JSONLoader(
#     file_path="langchain-handbook.jsonl",  # 替换为你的数据路径
#     jq_schema=".[]",
#     text_content=False
# )
# raw_docs = loader.load()

# documents = []
# for raw_doc in raw_docs:
#     answer = raw_doc.metadata.get("answer", "")
#     documents.append(Document(page_content=answer))  # 构造 LangChain 文档对象

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
# retriever = FAISSRetriever.from_documents(documents, embeddings)
# retriever.search_kwargs["k"] = 1  # 检索最相关的1个文档

def generate_answer_with_rag(question):
    relevant_docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in relevant_docs])
    input_text = f"follow the knowledge below: {context} question: {question} answer: "
    
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")
    outputs = model.generate(
        inputs,
        generation_config=GenerationConfig(
            temperature=0.2,
            max_new_tokens=200,
            repetition_penalty=1.5
        )
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("answer：")[-1].strip()

# 步骤八：模型部署和使用

model = AutoModelForCausalLM.from_pretrained("fine_tuned_gpt2")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_gpt2")

qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


print("模型加载完成！")
# 测试模型
# question = "What is LangChain?"
# input_text = f"question：{question} answer："
# answer = qa_pipeline(input_text, max_length=200, num_return_sequences=1)[0]['generated_text']
# print(answer.replace(input_text, ""))

question = "What is LangChain?"
print("Question:", question)
print("Answer:", generate_answer_with_rag(question))