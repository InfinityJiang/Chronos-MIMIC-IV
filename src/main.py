# -*- coding: utf-8 -*-
# ============== 模块导入 ==============
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chronos import ChronosPipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import time
import types
import seaborn as sns
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']

# ============== 配置参数 ==============
# 数据路径和保存目录
SAVE_DIR = "./saved"

# 选择要使用的标签类型（readmission: 再入院, mor: 是否死亡, los: 住院时长是否超过七天）
DATA_NAME = ("labels_los", "labels_readmission", "labels_mor")[1]

DATA_PATH = f"patient_data/data.pt"
os.makedirs(SAVE_DIR, exist_ok=True)

# 训练参数
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
SEED = 24

# 运行控制
RUN_TRAIN = False  # 是否训练模型
RUN_EVAL = True  # 是否评估模型
CONTINUE_TRAIN = False
MODEL_PATH = os.path.join(SAVE_DIR, f'classifier_{DATA_NAME}.pt')

# Chronos模型配置
CHRONOS_MODEL_NAME = "amazon/chronos-t5-small"
CHRONOS_ON_CPU = False  # 如果GPU内存不足可以设为True
FREEZE_CHRONOS = True  # 是否冻结Chronos模型的参数

# 设备配置
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# 随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)

# ============== 数据加载 ==============
print("加载数据：", DATA_PATH)
data = torch.load(DATA_PATH)

# 加载各种医疗数据
meds = data["meds"].float()  # [N, T, D_meds]
out = data["out"].float()  # [N, T, D_out]
proc = data["proc"].float()  # [N, T, D_proc]
stat = data["stat_df"].float()  # [N, D_stat]
demo = data["demo_df"].float()  # [N, D_demo]
y_df = data[DATA_NAME]
y = y_df.long()

print("数据形状:", meds.shape, out.shape, proc.shape, stat.shape, demo.shape, y.shape)


# ============== 数据集类 ==============
class PatientDataset(Dataset):
    """自定义患者数据集类，封装所有医疗数据和标签"""

    def __init__(self, meds, out, proc, stat, demo, y):
        self.meds = meds
        self.out = out
        self.proc = proc
        self.stat = stat
        self.demo = demo
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.meds[idx],
            self.out[idx],
            self.proc[idx],
            self.stat[idx],
            self.demo[idx],
            self.y[idx],
        )


# ============== 数据划分：70/20/10 (train/val/test) ==============
n_samples = len(y)
idx = np.arange(n_samples)

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)  # train vs temp
train_idx, temp_idx = next(sss1.split(idx, y.numpy()))

val_fraction_of_temp = 2.0 / 3.0
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_fraction_of_temp, random_state=SEED)
val_idx_rel, test_idx_rel = next(sss2.split(temp_idx, y[temp_idx].numpy()))
val_idx = temp_idx[val_idx_rel]
test_idx = temp_idx[test_idx_rel]

print(f"划分后样本数: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
print(
    f"标签正样本比例: 全量={y.float().mean():.4f}, train={y[train_idx].float().mean():.4f}, val={y[val_idx].float().mean():.4f}, test={y[test_idx].float().mean():.4f}"
)


# ============== 训练集 SMOTE 过采样（在 pooled 特征空间上） ==============
def smote_oversample_build(meds, out, proc, stat, demo, y, random_state=SEED):
    """
    对训练数据做 SMOTE：
    - 在 pooled 特征空间（对序列取时间维均值）上执行 SMOTE，得到合成的 pooled 特征 + stat/demo。
    - 对于 SMOTE 生成的“新行”，通过在原始 pooled 空间中寻找最近邻，将该最近邻样本的原始时间序列 (meds/out/proc) 复制过来；
      并将 stat/demo 替换为 SMOTE 生成的 stat/demo（使静态特征具有合成特性）。
    """
    X_meds = meds.mean(axis=1).numpy()  # [n, D_meds]
    X_out = out.mean(axis=1).numpy()
    X_proc = proc.mean(axis=1).numpy()
    X_stat = stat.numpy()
    X_demo = demo.numpy()
    X_pool = np.concatenate([X_meds, X_out, X_proc, X_stat, X_demo], axis=1)
    y_np = y.numpy()

    positive_count = sum(y_np == 1)
    sampling_strategy = {1: positive_count * 2}

    sm = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_res, y_res = sm.fit_resample(X_pool, y_np)

    # 如果 X_res 中的行与原始池中完全相同，则直接使用对应原始样本
    # 否则认为该行为 SMOTE 生成的合成样本，需要构造其完整模态数据
    # 构造近邻搜索器以找到原始样本的最近邻（用于拷贝时间序列）

    nbrs = NearestNeighbors(n_neighbors=1).fit(X_pool)
    meds_new = []
    out_new = []
    proc_new = []
    stat_new = []
    demo_new = []
    y_new = []

    for i, row in enumerate(X_res):
        # 尝试匹配原始行
        matches = np.all(np.isclose(X_pool, row, atol=1e-8), axis=1)
        if matches.any():
            idx0 = np.where(matches)[0][0]
            meds_new.append(meds[idx0].numpy())
            out_new.append(out[idx0].numpy())
            proc_new.append(proc[idx0].numpy())
            stat_new.append(X_stat[idx0])  # 或者用 row 中的 stat 段
            demo_new.append(X_demo[idx0])
            y_new.append(y_res[i])
        else:
            # 合成行：用最近邻的时间序列作为代理，但用合成的 stat/demo（row 的末端）
            _, nn_idx = nbrs.kneighbors(row.reshape(1, -1), n_neighbors=1, return_distance=True)
            nn = int(nn_idx[0, 0])
            meds_new.append(meds[nn].numpy().copy())
            out_new.append(out[nn].numpy().copy())
            proc_new.append(proc[nn].numpy().copy())
            # 从 row 中拆出 stat/demo（假定其在拼接的后端）
            # 先计算各段长度
            D_meds = X_meds.shape[1]
            D_out = X_out.shape[1]
            D_proc = X_proc.shape[1]
            D_stat = X_stat.shape[1]
            D_demo = X_demo.shape[1]
            start = D_meds + D_out + D_proc
            stat_from_row = row[start:start + D_stat]
            demo_from_row = row[start + D_stat:start + D_stat + D_demo]
            stat_new.append(stat_from_row)
            demo_new.append(demo_from_row)
            y_new.append(y_res[i])

    # 转回 torch.tensor
    meds_res_t = torch.tensor(np.stack(meds_new), dtype=meds.dtype)
    out_res_t = torch.tensor(np.stack(out_new), dtype=out.dtype)
    proc_res_t = torch.tensor(np.stack(proc_new), dtype=proc.dtype)
    stat_res_t = torch.tensor(np.stack(stat_new), dtype=stat.dtype)
    demo_res_t = torch.tensor(np.stack(demo_new), dtype=demo.dtype)
    y_res_t = torch.tensor(np.array(y_new), dtype=y.dtype)

    return meds_res_t, out_res_t, proc_res_t, stat_res_t, demo_res_t, y_res_t


# ============== 创建训练/验证/测试 Dataset 与 DataLoader ==============
# 对训练集做 SMOTE / 过采样
meds_train, out_train, proc_train, stat_train, demo_train, y_train = (
    meds[train_idx], out[train_idx], proc[train_idx], stat[train_idx], demo[train_idx], y[train_idx]
)

meds_train, out_train, proc_train, stat_train, demo_train, y_train = smote_oversample_build(
    meds_train, out_train, proc_train, stat_train, demo_train, y_train, random_state=SEED
)  # 过采样

# 构建 Dataset / DataLoader
train_dataset = PatientDataset(meds_train, out_train, proc_train, stat_train, demo_train, y_train)
val_dataset = PatientDataset(meds[val_idx], out[val_idx], proc[val_idx], stat[val_idx], demo[val_idx], y[val_idx])
test_dataset = PatientDataset(meds[test_idx], out[test_idx], proc[test_idx], stat[test_idx], demo[test_idx], y[test_idx])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


class ChronosClassifier(nn.Module):

    def __init__(
        self,
        chronos_model_name=CHRONOS_MODEL_NAME,
        bottleneck_dim=128,
        dropout=0.3,
        chronos_on_cpu=CHRONOS_ON_CPU,
        freeze_chronos=FREEZE_CHRONOS,
    ):
        super().__init__()
        # self 现在是 ChronosClassifier 实例

        self.chronos_model_name = chronos_model_name

        # 确定 Chronos Pipeline 的设备配置
        self.pipeline_device = torch.device("cpu" if chronos_on_cpu or not torch.cuda.is_available() else DEVICE)
        device_map = "cpu" if self.pipeline_device.type == "cpu" else self.pipeline_device
        print(f"加载ChronosPipeline {chronos_model_name} 到 {device_map}...")

        # 加载 Pipeline
        self.pipeline = ChronosPipeline.from_pretrained(
            chronos_model_name,
            device_map=device_map,
        )

        # 获取 EOS token ID
        self.eos_token_id = self.pipeline.model.config.eos_token_id
        if self.eos_token_id is None:
            print("警告：未能从模型配置中获取 eos_token_id，使用默认值 2。")
            self.eos_token_id = 2

        # 将数据移动到GPU上
        if hasattr(self.pipeline, 'tokenizer'):
            tk = self.pipeline.tokenizer

            # 将 tokenizer 内部的辅助张量移动到正确设备
            if hasattr(tk, 'centers') and isinstance(tk.centers, torch.Tensor):
                tk.centers = tk.centers.to(self.pipeline_device)
            if hasattr(tk, 'boundaries') and isinstance(tk.boundaries, torch.Tensor):
                tk.boundaries = tk.boundaries.to(self.pipeline_device)

            # 替换 tokenizer 的 _append_eos_token 方法
            # 使用闭包特性，让内部函数能够访问外部 __init__ 中的 self.eos_token_id
            def _patched_append_eos_token_closure(tk_instance, token_ids, attention_mask):
                eos_id = self.eos_token_id

                device = token_ids.device
                batch_size = token_ids.shape[0]

                # 确保新创建的张量和输入数据在同一设备
                eos = torch.tensor([eos_id], dtype=token_ids.dtype, device=device)
                eos_tokens = eos.repeat(batch_size, 1)

                token_ids = torch.cat((token_ids, eos_tokens), dim=1)

                if attention_mask is not None:
                    ones = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                    attention_mask = torch.cat((attention_mask, ones), dim=1)

                return token_ids, attention_mask

            # 应用补丁
            tk._append_eos_token = types.MethodType(_patched_append_eos_token_closure, tk)

        # 冻结 Chronos 参数
        if freeze_chronos:
            for p in self.pipeline.model.parameters():
                p.requires_grad = False

        # 定义池化层 (用于将多变量序列转换为 Chronos 期望的单变量序列)
        # 维度计算使用占位符张量
        D_meds = meds.shape[2]
        D_out = out.shape[2]
        D_proc = proc.shape[2]

        # 将池化层放到主设备 (DEVICE) 上
        self.pool_meds = nn.Conv1d(in_channels=D_meds, out_channels=1, kernel_size=1).to(DEVICE)
        self.pool_out = nn.Conv1d(in_channels=D_out, out_channels=1, kernel_size=1).to(DEVICE)
        self.pool_proc = nn.Conv1d(in_channels=D_proc, out_channels=1, kernel_size=1).to(DEVICE)

        # 检测嵌入维度
        dummy_len = meds.shape[1]
        # dummy_input 必须在 pipeline_device 上
        dummy_input = torch.ones(1, dummy_len).to(self.pipeline_device)

        with torch.no_grad():
            emb_dummy, _ = self.pipeline.embed(dummy_input)

        chronos_dim = emb_dummy.shape[-1]
        print("检测到的Chronos嵌入维度:", chronos_dim)

        # 定义分类头 (FC layers)
        non_seq_dim = stat.shape[1] + demo.shape[1]
        total_dim = chronos_dim*3 + non_seq_dim  # 3 个 Chronos 嵌入 + 静态/人口特征

        self.bottleneck = nn.Sequential(
            nn.Linear(total_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(bottleneck_dim, 1)

    # 辅助函数：将 Chronos 输出 (可能为 3D) 转换为 2D
    def _embed_to_2d(self, emb):
        if emb.dim() == 3:
            # (Batch, Time, Dim) -> (Batch, Dim)
            return emb.mean(dim=1)
        return emb

    def forward(self, meds, out, proc, stat, demo):
        # 1. 预处理：多变量 -> 单变量序列 (N, T)
        # 步骤: (N, T, D) -> (N, D, T) -> Conv1d(N, 1, T) -> squeeze(1) -> (N, T)
        meds_1d = self.pool_meds(meds.transpose(1, 2)).squeeze(1)
        out_1d = self.pool_out(out.transpose(1, 2)).squeeze(1)
        proc_1d = self.pool_proc(proc.transpose(1, 2)).squeeze(1)

        # 2. 转移到 Pipeline 设备进行特征提取
        meds_for_embed = meds_1d.to(self.pipeline_device)
        out_for_embed = out_1d.to(self.pipeline_device)
        proc_for_embed = proc_1d.to(self.pipeline_device)

        # 3. Chronos 嵌入
        with torch.no_grad():
            meds_emb, _ = self.pipeline.embed(meds_for_embed)
            out_emb, _ = self.pipeline.embed(out_for_embed)
            proc_emb, _ = self.pipeline.embed(proc_for_embed)

        # 4. 转移回主设备，并转换为 2D 特征
        meds_emb = self._embed_to_2d(meds_emb).to(stat.device)
        out_emb = self._embed_to_2d(out_emb).to(stat.device)
        proc_emb = self._embed_to_2d(proc_emb).to(stat.device)

        # 5. 拼接特征并分类
        x = torch.cat([meds_emb, out_emb, proc_emb, stat, demo], dim=-1)
        x = self.bottleneck(x)
        logits = self.fc(x).squeeze(-1)
        probs = torch.sigmoid(logits)

        return probs, logits


# ============== 评估/训练/保存/绘图函数 ==============
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    criterion = nn.BCEWithLogitsLoss()
    y_trues = []
    y_scores = []

    with torch.no_grad():
        for meds_b, out_b, proc_b, stat_b, demo_b, y_b in loader:
            meds_b = meds_b.to(device)
            out_b = out_b.to(device)
            proc_b = proc_b.to(device)
            stat_b = stat_b.to(device)
            demo_b = demo_b.to(device)
            y_b = y_b.float().to(device)

            probs, logits = model(meds_b, out_b, proc_b, stat_b, demo_b)
            loss = criterion(logits, y_b)

            preds = (probs > 0.5).long()
            correct += (preds == y_b.long()).sum().item()
            total += y_b.size(0)
            loss_sum += loss.item() * y_b.size(0)

            y_trues.append(y_b.cpu().numpy())
            y_scores.append(probs.cpu().numpy())

    y_trues = np.concatenate(y_trues) if len(y_trues) > 0 else np.array([])
    y_scores = np.concatenate(y_scores) if len(y_scores) > 0 else np.array([])
    return (loss_sum / total if total > 0 else float('nan'), correct / total if total > 0 else float('nan'), y_trues, y_scores)


def save_classifier(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'chronos_model_name': model.chronos_model_name,
    }, path)
    print("模型已保存到", path)


def load_classifier(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ChronosClassifier(chronos_model_name=ckpt.get('chronos_model_name', CHRONOS_MODEL_NAME))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print("从", path, "加载模型")
    return model


def train(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR, save_path=None):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 早停相关参数
    patience = 5  # 允许性能不提升的epoch数
    best_auc = -1.0
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        model.train()
        total_loss = 0.0
        for meds_b, out_b, proc_b, stat_b, demo_b, y_b in train_loader:
            meds_b = meds_b.to(device)
            out_b = out_b.to(device)
            proc_b = proc_b.to(device)
            stat_b = stat_b.to(device)
            demo_b = demo_b.to(device)
            y_b = y_b.float().to(device)

            probs, logits = model(meds_b, out_b, proc_b, stat_b, demo_b)
            loss = criterion(logits, y_b)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            total_loss += loss.item() * y_b.size(0)

        # 训练/验证评估
        train_loss, train_acc, _, _ = evaluate(model, train_loader, device)
        val_loss, val_acc, y_val_trues, y_val_scores = evaluate(model, val_loader, device)
        try:
            val_auc = roc_auc_score(y_val_trues, y_val_scores) if len(y_val_trues) > 0 else float('nan')
        except Exception:
            val_auc = float('nan')

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}"
        )

        # 早停逻辑
        if not np.isnan(val_auc):
            if val_auc > best_auc:  # 若验证集上达到历史最佳，则保存最佳模型
                best_auc = val_auc
                epochs_no_improve = 0
                if save_path is not None:
                    save_classifier(model, save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:  # 超过一定代数不提升则终止
                    early_stop = True
        else:
            print("Warning: Validation AUC is NaN, skipping early stopping check")

    return model, best_auc


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])

    plt.figure(figsize=(8, 6))

    # 使用seaborn绘制热力图
    sns.heatmap(
        cm_df,
        annot=True,  # 显示数值
        fmt='d',  # 整数格式
        cmap='Blues',  # 蓝色渐变
        cbar=True,  # 显示颜色条
        linewidths=0.5,  # 单元格边框宽度
        linecolor='lightgray',  # 单元格边框颜色
        annot_kws={'size': 16}  # 调整注释字体大小
    )

    plt.title(f'混淆矩阵-{DATA_NAME}', fontsize=18)
    plt.xlabel('预测', fontsize=16)
    plt.ylabel('真实', fontsize=16)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_roc(y_true, y_scores, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f'ROC曲线 (AUC = {roc_auc:.4f})', fontsize=18)
    plt.xlabel('假阳性率', fontsize=16)
    plt.ylabel('真阳性率', fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_probability_histogram(y_true, y_scores, bins=30):
    plt.figure(figsize=(8, 6))

    # 分别提取正类和负类的预测概率
    pos_probs = y_scores[y_true == 1]
    neg_probs = y_scores[y_true == 0]

    plt.hist(pos_probs, bins=bins, alpha=0.7, color='green', label='正类 (y_true=1)')
    plt.hist(neg_probs, bins=bins, alpha=0.7, color='red', label='负类 (y_true=0)')

    plt.title('预测概率分布直方图', fontsize=15)
    plt.xlabel('预测为正类的概率', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_calibration_curve(y_true, y_scores, n_bins=10, strategy='uniform'):
    # 计算校准曲线
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy=strategy)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label='模型', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='完美校准')

    plt.title('校准曲线', fontsize=15)
    plt.xlabel('预测概率 (分箱均值)', fontsize=12)
    plt.ylabel('真实正类比例', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 确保坐标轴范围是0到1
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# ============== 主程序 ==============
if __name__ == "__main__":
    if CONTINUE_TRAIN:
        print("继续训练模型...")
        model = load_classifier(MODEL_PATH, DEVICE)
    elif RUN_TRAIN:
        print("开始训练模型...")
        model = ChronosClassifier().to(DEVICE)

    if RUN_TRAIN:
        train(model, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR, save_path=MODEL_PATH)

    if RUN_EVAL:
        if os.path.exists(MODEL_PATH):
            model = load_classifier(MODEL_PATH, DEVICE)
        else:
            print("警告: 模型文件不存在，使用当前内存中的模型进行评估。")

        test_loss, test_acc, y_trues, y_scores = evaluate(model, test_loader, DEVICE)
        print(f"测试损失={test_loss:.4f}  测试准确率={test_acc:.4f}")

        if len(y_trues) > 0:
            y_preds = (y_scores > 0.5).astype(int)
            plot_confusion_matrix(y_trues, y_preds)
            plot_roc(y_trues, y_scores)
            plot_probability_histogram(y_trues, y_scores)
            plot_calibration_curve(y_trues, y_scores)
        else:
            print("评估集为空或未产生预测结果。")
