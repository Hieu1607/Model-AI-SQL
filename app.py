import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load và xử lý dữ liệu
train0_data = pd.read_json('data/train_0.json', lines=True)
train1_data = pd.read_json('data/train_1.json', lines=True)
train2_data = pd.read_json('data/train_2.json', lines=True)
train3_data = pd.read_json('data/train_3.json', lines=True)
train4_data = pd.read_json('data/train_4.json', lines=True)
train_data = pd.concat([train0_data, train1_data, train2_data, train3_data, train4_data])

# Loại bỏ các hàng chứa giá trị thiếu
train_data = train_data.dropna(subset=['sql_prompt', 'sql_context', 'sql'])

# Tạo cột input_text kết hợp câu hỏi và ngữ cảnh
train_data['input_text'] = train_data.apply(
    lambda row: f"Translate SQL: {row['sql_prompt']} Context: {row['sql_context']}", axis=1
)

# 2. Mã hóa dữ liệu
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

train_texts = train_data['input_text'].tolist()
train_labels = train_data['sql'].tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
train_labels_enc = tokenizer(train_labels, truncation=True, padding=True, max_length=512)

# Thêm nhãn vào dữ liệu mã hóa
train_encodings['labels'] = train_labels_enc['input_ids']

# Chuyển đổi thành tập Dataset
train_dataset = Dataset.from_dict(train_encodings)

# 3. Khởi tạo mô hình
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 4. Huấn luyện mô hình
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()







import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json



# 1. Tạo input_text cho dữ liệu test
test_data['input_text'] = test_data.apply(
    lambda row: f"Translate SQL: {row['sql_prompt']} Context: {row['sql_context']}", axis=1
)

# 2. Mã hóa dữ liệu test
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Tải mô hình đã huấn luyện
model = T5ForConditionalGeneration.from_pretrained('./results')

# 3. Dự đoán với mô hình cho tập test
predictions = []
total_correct = 0
total_samples = len(test_data)
for idx, text in enumerate(test_data['input_text']):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(**inputs)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if(prediction==test_data['sql']):
        total_correct += 1
        print('Test ' + idx + ' correct')

accuracy = total_correct/total_samples       

print('Accuracy: ' + accuracy)
