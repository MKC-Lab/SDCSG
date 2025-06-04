import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
from transformers import AdamW, AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import json
# Import configuration variables
from train_config import embedding_model_path, IMDB_data_path

# Set hyperparameters and paths
DATASET_PATH = "/root/autodl-tmp/DATG/data/train_dataset.json"  # 替换为您的数据文件路径
EMBEDDING_MODEL_NAME = '/root/autodl-tmp/model/bge-large-en-v1.5'
SAVE_PATH = '/root/autodl-tmp/DATG/model/internal_classifier/stance_classifier'  # 修改保存路径名称
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
MAX_ITER = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
TRAIN_SIZE = 0.9
GRADIENT_ACCUMULATION_STEPS = 8

# 加载立场数据
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    stance_data = json.load(f)

# 处理数据：从JSON结构中提取文本和标签
data_rows = []
for item in stance_data:
    # 提取对话文本
    conversations = item["conversations"][1]
    text = item["conversations"][1]["value"]
    
    # 二分类标签映射: against->0, favor/neutral->1
    label = 0 if item["stance"] == "against" else 1
    
    data_rows.append({"text": text, "label": label})

# 创建DataFrame
df_train = pd.DataFrame(data_rows)

# 查看数据分布
print("标签分布:")
print(df_train['label'].value_counts())

# 平衡数据集（与原代码一致）
min_count = min(df_train['label'].value_counts())
df_balanced = pd.concat([
    df_train[df_train['label'] == 1].sample(n=min_count, random_state=SEED),
    df_train[df_train['label'] == 0].sample(n=min_count, random_state=SEED)
]).sample(frac=1, random_state=SEED)

# 查看平衡后的数据分布
print("平衡后的标签分布:")
print(df_balanced['label'].value_counts())

# 分割训练集和验证集
df_train, df_val = train_test_split(df_balanced, test_size=1 - TRAIN_SIZE, random_state=SEED)
print(f"训练集大小: {len(df_train)}, 验证集大小: {len(df_val)}")

# 查看一个样本示例
print("\n样本示例:")
sample_idx = 0
print(f"文本: {df_train.iloc[sample_idx]['text'][:100]}...")  # 只显示前100个字符
print(f"标签: {df_train.iloc[sample_idx]['label']} ({'against' if df_train.iloc[sample_idx]['label'] == 0 else 'favor/neutral'})")

# Define a custom Dataset class for the sentiment analysis task
class CommentsDataset(Dataset):
    """Custom Dataset class for loading data into the PyTorch model."""
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoded_input = self.tokenizer(
            row['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=512,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }
    
    
# Define the classifier model
class GuideClassifier(PreTrainedModel):
    """Classifier model extending a base transformer model for sentiment analysis."""
    config_class = PretrainedConfig
    
    def __init__(self, base_model):
        super(GuideClassifier, self).__init__(base_model.config)
        self.sentence_transformer = base_model
        self.classification_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        model_output = self.sentence_transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        logits = self.classification_head(sentence_embeddings)
        return logits
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    

    
# Initialize tokenizer, model, datasets, and DataLoader
base_model_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
base_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
train_dataset = CommentsDataset(df_train, base_model_tokenizer)
val_dataset = CommentsDataset(df_val, base_model_tokenizer)

# Initialize model, loss function, and optimizer
model = GuideClassifier(base_model=base_model)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = BCEWithLogitsLoss()

# Prepare DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Training and validation loop
train_losses, val_losses, val_accuracies = [], [], []


# 修改训练循环
for epoch in range(MAX_ITER):
    model.train()
    total_loss, total_eval_accuracy, total_eval_loss = 0, 0, 0
    for i, batch_data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{MAX_ITER}')):
        outputs = model(
            input_ids=batch_data['input_ids'].to(DEVICE),
            attention_mask=batch_data['attention_mask'].to(DEVICE)
        )
        loss = criterion(outputs, batch_data['labels'].to(DEVICE).float().unsqueeze(1))
        
        # 缩放损失值
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        # 每积累N个批次才更新一次参数
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    # Validation step
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc='Validating'):
            outputs = model(
                input_ids=batch_data['input_ids'].to(DEVICE),
                attention_mask=batch_data['attention_mask'].to(DEVICE)
            )
            loss = criterion(outputs, batch_data['labels'].to(DEVICE).float().unsqueeze(1))
            total_eval_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            total_eval_accuracy += accuracy_score(batch_data['labels'].cpu(), preds.cpu())

    # Calculate and print average losses and accuracy
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = total_eval_loss / len(val_loader)
    avg_val_accuracy = total_eval_accuracy / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy * 100)
    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.2f}%')

# Save the model and tokenizer
def save_model(model, tokenizer, save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_config = model.module.config if hasattr(model, 'module') else model.config
    model_config.save_pretrained(save_path)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(model_state_dict, os.path.join(save_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(save_path)

save_model(model, base_model_tokenizer, SAVE_PATH)

# Plot training and validation losses, and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()