# nanoGPT 教程

nanoGPT 是 Andrej Karpathy 编写的一个极简 GPT 训练/微调/推理代码库。整个项目的核心代码只有两个文件：`train.py`（约 300 行训练循环）和 `model.py`（约 300 行 GPT 模型定义）。代码短小精悍、可读性极高，非常适合学习和快速实验。

---

## 1. 环境安装

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

**Python 版本要求：**

仓库未显式声明最低 Python 版本限制。从依赖推断：
- PyTorch 2.0+ 要求 Python **≥ 3.8**；建议使用 **Python 3.10 或 3.11**，兼容性最好
- 如果不用 `torch.compile`（加 `--compile=False`），PyTorch 1.x + Python 3.7 也能运行，但官方已不再测试此类旧组合
- Python 3.12+ 在较旧版本的 PyTorch 上可能存在兼容问题，请留意

依赖说明：

| 包名 | 用途 |
|------|------|
| `torch` | 深度学习框架（推荐 2.0+，支持 `torch.compile` 加速） |
| `numpy` | 数值计算，用于内存映射（memmap）读取数据 |
| `transformers` | HuggingFace 库，用于加载 OpenAI 官方 GPT-2 权重 |
| `datasets` | HuggingFace 数据集库，用于下载 OpenWebText |
| `tiktoken` | OpenAI 的快速 BPE 分词器 |
| `wandb` | 实验日志记录（可选） |
| `tqdm` | 进度条 |

---

## 2. 数据集准备

nanoGPT 的数据格式非常简单：将分词后的 token ID 序列以 `uint16` 格式存入 `.bin` 文件，训练时通过 numpy 的 `memmap` 零拷贝读取。

仓库提供了三套数据准备脚本：

### 2.1 字符级莎士比亚（适合入门调试）

```bash
python data/shakespeare_char/prepare.py
```

整体流程：**下载原始文本 → 统计字符集 → 建立字符↔整数映射 → 编码 → 切分训练/验证集 → 保存为二进制文件**。

**逐步分析：**

1. **下载数据** — 从 GitHub 下载约 1.1MB 的莎士比亚文本到本地 `input.txt`，如果文件已存在则跳过。
2. **统计字符集** — 遍历全文本，找出所有出现过的唯一字符，共 **65 个**（含大小写字母、标点、换行符等）：
   ```
    !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
   ```
3. **建立映射** — `stoi`（字符 → 整数索引）和 `itos`（整数索引 → 字符）。这就是该数据集的"分词器"——每个字符直接映射为一个整数，不使用 BPE 等子词切分方案。
4. **切分训练/验证集** — 按 **9:1** 分割，前 90% 为训练集（约 100 万 token），后 10% 为验证集（约 11 万 token）。
5. **编码并保存** — 将整数序列以 `uint16` 写入 `train.bin` 和 `val.bin`；同时将 `vocab_size`、`stoi`、`itos` 存入 `meta.pkl`，供后续采样时解码字符。

最终得到一个词表仅 65、无外部依赖的迷你语言模型数据集。

### 2.2 BPE 分词版莎士比亚（适合微调实验）

```bash
python data/shakespeare/prepare.py
```

- 使用 GPT-2 的 BPE 分词器（tiktoken）编码
- 训练集约 30 万 token，验证集约 3.6 万 token

**字符级 vs BPE 对比：**

| | `shakespeare_char/` | `shakespeare/` |
|---|---|---|
| 分词方式 | 逐字符 | GPT-2 BPE |
| 词表大小 | 65 | 50257 |
| 是否需要 tiktoken | 否 | 是 |
| meta.pkl | 有（含编解码映射） | 无（直接用 tiktoken） |

字符级模型的优势是简单直观、无外部依赖；缺点是序列更长、语义颗粒度较粗，生成质量不如 BPE 模型。

### 2.3 OpenWebText（复现 GPT-2 用）

```bash
python data/openwebtext/prepare.py
```

- 从 HuggingFace 下载 OpenWebText 数据集
- 使用 GPT-2 BPE 分词
- 训练集约 90 亿 token，约 17GB；验证集约 400 万 token

---

## 3. 训练

`train.py` 是整个 nanoGPT 的核心训练脚本（约 330 行），支持单 GPU、多 GPU 分布式训练（DDP）、CPU 甚至 Apple Silicon（MPS）。以下按执行顺序分析其内部机制。

### 3.0 train.py 内部机制

**配置系统（第 33–78 行）**：所有以大写字母开头的全局变量都是可配置项。第 77 行通过 `exec(open('configurator.py').read())` 执行配置覆写——要么从配置文件（如 `config/train_gpt2.py`）读入，要么从命令行 `--key=value` 直接覆盖。

**DDP 分布式初始化（第 82–101 行）**：通过环境变量 `RANK` 判断是否为 `torchrun` 启动的 DDP 模式。DDP 模式下调用 `init_process_group`，梯度累积步数按进程数等比例缩小，`master_process` 标志位确保只有 rank 0 做日志和 checkpoint 保存。

**数据加载器（第 115–131 行）**：没有使用 PyTorch 的 DataLoader，而是手工实现。用 numpy `memmap` 读取 `.bin` 文件（uint16 格式的 token 序列），`get_batch(split)` 每次随机截取 `block_size` 长度的连续 token 作为 `x`，偏移一位作为 `y`（预测下一个 token）。每次调用重建 memmap 对象防止内存泄漏。CUDA 下使用 `pin_memory()` + `non_blocking=True` 实现异步传输。

**模型初始化（第 146–202 行）**，三种模式由 `init_from` 控制：

| 模式 | 行为 |
|------|------|
| `'scratch'` | 随机初始化，vocab_size 取自 `meta.pkl` 或默认 50304 |
| `'resume'` | 从 `out_dir/ckpt.pt` 恢复模型 + 优化器状态 |
| `'gpt2'` / `'gpt2-xl'` 等 | 通过 HuggingFace transformers 加载 OpenAI 权重 |

加载后若 `block_size` 小于模型原本的上下文长度，调用 `crop_block_size()` 裁切位置嵌入和注意力 mask。

**混合精度（第 109–112、196 行）**：默认 `bfloat16`（GPU 支持时），否则 `float16` + `GradScaler`（自动梯度缩放防溢出），也可强制 `float32`。`torch.amp.autocast` 包裹前向传播，`GradScaler` 包裹反向传播。

**优化器（第 199 行）**：调用 `model.configure_optimizers()` 返回 AdamW。关键设计是**选择性权重衰减**——2D+ 参数（Linear 权重、Embedding）施加 weight decay，1D 参数（bias、LayerNorm）不施加。优先使用 PyTorch fused AdamW（CUDA 下更快）。

**torch.compile（第 205–208 行）**：默认启用 PyTorch 2.0 的 `torch.compile`，可将迭代时间从 ~250ms 降至 ~135ms。通过 `--compile=False` 可禁用。

**学习率调度（第 231–242 行）**，三段式：
```
warmup_iters 步:     线性从 0 升到 learning_rate
中间阶段:             余弦衰减
lr_decay_iters 步后:  保持 min_lr（通常为 learning_rate / 10）
```
第 233 行的 `(it + 1) / (warmup_iters + 1)` 确保第 0 步学习率非零（PR #578 修复的 bug）。

**训练循环（第 255–333 行）**：每 `eval_interval` 步评估 loss 并保存 checkpoint。内部梯度累积循环中，forward 后异步预取下一批数据，backward 由 GradScaler 缩放，DDP 下只最后一次 micro-step 同步梯度（通过 `model.require_backward_grad_sync`）。之后梯度裁剪（默认 1.0）、optimizer.step()、计算 MFU（Model FLOPS Utilization，以 A100 的 312 TFLOPS 为基准衡量训练效率）。

### 3.1 训练流程全景分析：`python train.py config/train_shakespeare_char.py`

以下以 `python train.py config/train_shakespeare_char.py` 这条命令为例，逐步分析从启动到训练结束的完整过程。

#### 第 1 步：配置合并

`train.py` 开头定义了一套默认超参（针对 OpenWebText 上复现 GPT-2 124M），然后第 77 行执行：

```python
exec(open('configurator.py').read())
```

`configurator.py` 解析命令行参数。由于传入了 `config/train_shakespeare_char.py`（不含 `=` 号），它直接 `exec()` 该配置文件，用莎士比亚专用值覆盖默认全局变量：

| 参数 | 默认值 (GPT-2 124M) | 覆盖后的值 (莎士比亚) | 说明 |
|---|---|---|---|
| `dataset` | `'openwebtext'` | `'shakespeare_char'` | 数据子目录 |
| `batch_size` | 12 | **64** | 微批次大小 |
| `block_size` | 1024 | **256** | 上下文长度（最多看 256 个字符） |
| `n_layer` | 12 | **6** | Transformer 层数 |
| `n_head` | 12 | **6** | 注意力头数 |
| `n_embd` | 768 | **384** | 嵌入维度 |
| `dropout` | 0.0 | **0.2** | Dropout 概率 |
| `learning_rate` | 6e-4 | **1e-3** | 最大学习率 |
| `max_iters` | 600000 | **5000** | 总训练迭代数 |
| `gradient_accumulation_steps` | 40 | **1** | 梯度累积步数（这里不做累积） |
| `warmup_iters` | 2000 | **100** | 学习率预热步数 |
| `lr_decay_iters` | 600000 | **5000** | 学习率衰减步数（= max_iters） |
| `min_lr` | 6e-5 | **1e-4** | 最小学习率 |
| `beta2` | 0.95 | **0.99** | Adam β₂（增大以应对小 batch） |

最终模型约 **10.6M 参数**——一个真正的"迷你"GPT，适合单 GPU 快速迭代。

#### 第 2 步：数据加载（"穷人的数据加载器"）

数据集由 `data/shakespeare_char/prepare.py` 预先准备：

**准备阶段**（训练前一次性执行）：
1. 下载 Tiny Shakespeare 文本（约 111 万字符）
2. 遍历全文本，提取 65 个唯一字符：`!$&',-.3:;?A-Za-z` 等
3. 构建字符级映射：`stoi`（字符 → 整数）、`itos`（整数 → 字符）——这就是该数据集的"分词器"
4. 按 9:1 切分训练/验证集
5. 将整数序列以 `uint16` 写入 `train.bin` 和 `val.bin`，同时将 `vocab_size=65`、`stoi`、`itos` 存入 `meta.pkl`

**训练时** — `get_batch('train')`（`train.py:116-131`）的每一步：

```python
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
```

- 用 **`np.memmap`** 打开 `.bin` 文件——操作系统将文件直接映射到虚拟内存，按需读取，不会把整个 2MB 文件加载到 RAM
- 在 `[0, len(data) - 256]` 范围内随机采样 **64 个起始索引**
- 每个样本截取 256 个连续 token：`x` = `data[i : i+256]`，`y` = `data[i+1 : i+257]`（标准"预测下一个 token"）
- 最终 batch 形状：**(64, 256)**，每个迭代步处理 64 × 256 = **16,384 个 token**
- 每条 CUDA 张量通过 `pin_memory()` + `non_blocking=True` **异步传输**到 GPU，不阻塞 CPU
- **每个 batch 都重新创建 memmap**，避免 NumPy 的 memmap 内存泄漏问题（见 [StackOverflow #45132940](https://stackoverflow.com/questions/45132940)）

#### 第 3 步：模型初始化

因 `init_from='scratch'`（默认值），走从头训练路径（`train.py:149-157`）：

```python
model_args = dict(n_layer=6, n_head=6, n_embd=384, block_size=256,
                  bias=False, vocab_size=65, dropout=0.2)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
```

`model.__init__()` 中构建的完整模型结构：

```
GPT (10.6M 参数)
├── transformer.wte       Embedding(65, 384)         词元嵌入（查表）
├── transformer.wpe       Embedding(256, 384)        位置嵌入（可学习）
├── transformer.drop      Dropout(0.2)
├── transformer.h         ModuleList([               ×6 个 Transformer 块
│   └── Block
│       ├── ln_1          LayerNorm(384, bias=False)  → 预归一化
│       ├── attn          CausalSelfAttention         → 因果自注意力
│       │   ├── c_attn    Linear(384, 1152)           Q/K/V 联合投影
│       │   └── c_proj    Linear(384, 384)            输出投影
│       ├── ln_2          LayerNorm(384, bias=False)  → 预归一化
│       └── mlp           MLP(384 → 1536 → 384)      前馈网络 (4× 扩展比)
│           ├── c_fc      Linear(384, 1536)
│           ├── gelu      GELU()
│           └── c_proj    Linear(1536, 384)
├── transformer.ln_f      LayerNorm(384, bias=False) 最终层归一化
└── lm_head               Linear(384, 65, bias=False) 输出投影 → logits
```

核心设计要点：

- **权重绑定**（`model.py:138`）：`lm_head.weight` 与 `transformer.wte.weight` **共享同一张量**（`self.transformer.wte.weight = self.lm_head.weight`），节省参数并提升性能
- **无偏置**（`bias=False`）：所有 Linear 层和 LayerNorm 均不加偏置，比 GPT-2 更简洁快速
- **Pre-LayerNorm**（`model.py:103-106`）：每个 Block 在 Attention/MLP **之前**做 LayerNorm，然后残差连接——现代 Transformer 的标准做法
- **权重初始化**：Linear 层用 `N(0, 0.02)`；残差投影 (`c_proj`) 额外按 `1/√(2·n_layer)` 缩放，防止残差信号随深度膨胀
- **Flash Attention**：PyTorch ≥ 2.0 时自动使用 `torch.nn.functional.scaled_dot_product_attention`，CUDA 内核级融合，不需要具体化 `(T, T)` 注意力矩阵

#### 第 4 步：优化器设置

`model.configure_optimizers()`（`model.py:263-287`）返回 **AdamW**，采用了**选择性权重衰减**的策略：

```
decay_params    → 所有 ≥2D 的参数（Linear 权重、Embedding）→ weight_decay = 0.1
nodecay_params  → 所有 <2D 的参数（LayerNorm 权重、bias）  → weight_decay = 0.0
```

- `betas = (0.9, 0.99)` — β₂ 是 0.99 而非默认的 0.95，因为每个迭代步只有 ~16K token，更大的 β₂ 使二阶矩估计更平滑
- 若 CUDA 设备支持，自动使用 **fused AdamW**（`fused=True`），将优化器核心逻辑融合为单个 CUDA kernel，减少多次内存读写

混合精度策略（`train.py:109-112`）：

- 默认 `bfloat16`（若 GPU 支持 bf16）——动态范围大，不易溢出
- 否则 `float16` + `GradScaler`（自动缩放 loss 防止梯度下溢，`train.py:196`）
- 也可通过 `--dtype=float32` 强制全精度

#### 第 5 步：学习率调度

`get_lr(it)`（`train.py:231-242`）实现**线性预热 + 余弦衰减**的三段式调度：

```
迭代步     0 → 100  (warmup):     从 ~0 线性上升至 1e-3
迭代步   100 → 5000 (cosine):     余弦衰减至 1e-4 (min_lr)
迭代步  5000+            :        保持 1e-4
```

第 233 行的具体实现：

```python
if it < warmup_iters:
    return learning_rate * (it + 1) / (warmup_iters + 1)
```

使用 `(it + 1)` 而非 `it`，确保第 0 步学习率为 `1e-3 / 101 ≈ 9.9e-6`（非零）。这是 PR #578 修复的一个关键 bug——warmup 从零开始时，Adam 的动量项在第一步无法建立有效的梯度方向。

余弦衰减阶段：

```python
decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 从 1.0 平滑降到 0.0
return min_lr + coeff * (learning_rate - min_lr)
```

#### 第 6 步：训练循环（逐迭代详解）

主循环位于 `train.py:255-333`。以下是**单次迭代**的完整执行路径：

```
┌─────────────────────────────────────────────────────────┐
│  iter_num = 0, 1, 2, ..., 5000                         │
│                                                         │
│  ① 计算并设置当前 LR                                       │
│     lr = get_lr(iter_num)                               │
│                                                         │
│  ② 每 250 步评估 + 保存 checkpoint（仅 master_process）       │
│     if iter_num % 250 == 0:                             │
│         losses = estimate_loss()  # 取 200 个 batch 平均   │
│         若 val_loss 改善 → 保存 ckpt.pt                   │
│                                                         │
│  ③ 梯度累积循环（此处 gradient_accumulation_steps=1，仅一次）  │
│     with autocast(bfloat16):                            │
│         logits, loss = model(X, Y)  # 前向传播             │
│     scaler.scale(loss).backward()   # 反向传播             │
│     X, Y = get_batch('train')       # 异步预取下一批        │
│                                                         │
│  ④ 梯度裁剪 + 优化器更新                                    │
│     clip_grad_norm_(max_norm=1.0)                       │
│     scaler.step(optimizer)                              │
│     scaler.update()                                     │
│     optimizer.zero_grad(set_to_none=True)               │
│                                                         │
│  ⑤ 每 10 步打印日志                                       │
│     iter {n}: loss {val}, time {ms}ms, mfu {pct}%       │
└─────────────────────────────────────────────────────────┘
```

**前向传播详解**（`model.py:170-193`）：

1. **嵌入**：token → `wte`（查表），位置 → `wpe`（可学习嵌入），相加后过 Dropout
2. **×6 个 Block**，每个：
   - `x = x + CausalSelfAttention(LayerNorm(x))` — 因果自注意力（只看到当前及之前的 token）
   - `x = x + MLP(LayerNorm(x))` — GELU 前馈网络
3. **最终 LayerNorm** → `lm_head` 投影得到 logits：**(64, 256, 65)** — 每个位置在每个词表词条上的分数
4. **损失计算**：`cross_entropy(logits.view(-1, 65), targets.view(-1))` — 沿词表维度展开，逐 token 标准分类损失

**异步预取的时序技巧**（`train.py:303`）：

```
           GPU                           CPU
     ┌─────────────┐           ┌─────────────────┐
     │ forward(X,Y) │◄──────────│ X,Y = 当前 batch  │
     │    loss      │           └─────────────────┘
     │ backward()   │
     └──────┬───────┘
            │ GPU 忙于 backward
            │ 同时 CPU 异步传输下一批数据
            ▼
     ┌─────────────┐           ┌─────────────────┐
     │  scaler.step │           │ X,Y = get_batch │ ← 预取下一批
     │  zero_grad() │           │ (pin_memory +   │
     └─────────────┘           │  non_blocking)   │
                               └─────────────────┘
```

前向传播之后立即调用 `get_batch('train')`，在 GPU 执行反向传播的同时，CPU 端异步准备下一批数据，隐藏了数据传输延迟。

**评估**（`train.py:215-228`）：

`estimate_loss()` 将模型切换到 `eval()` 模式，从训练集和验证集各采样 200 个 batch，计算平均交叉熵损失，然后恢复 `train()` 模式。这是对训练/验证困惑度的无偏估计。

**检查点**（`train.py:274-286`）：

每 250 步，若验证损失改善则保存 `out-shakespeare-char/ckpt.pt`，包含：
- `model` state_dict（原始模型，非 DDP wrapper）
- `optimizer` state_dict（可断点续训）
- `model_args`（可重建模型架构）
- `iter_num`、`best_val_loss`、`config`

#### 整体数据流总结

```
Tiny Shakespeare 文本 (1.1M 字符, 65 个唯一字符)
        │
        ▼
  prepare.py → train.bin (1,003,854 uint16 token id)
               val.bin   (111,540 uint16 token id)
               meta.pkl  (vocab_size=65, stoi/itos 映射)
        │
        ▼
  get_batch() → 随机选 64 个连续 chunk，各 256 token
                X: (64, 256) 输入 id, Y: (64, 256) 目标 id (偏移 1)
        │
        ▼
  GPT.forward() → logits (64, 256, 65), loss (标量)
        │                         │
        ▼                         ▼
  loss.backward()            cross_entropy(预测, 目标)
        │
        ▼
  AdamW step (lr=1e-3, wd=0.1, β=(0.9, 0.99))
        │
        ▼
  重复 5,000 步，余弦衰减学习率至 1e-4
```

### 3.2 其他训练场景

**仅用 CPU / MacBook 时：**

```bash
python train.py config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --eval_iters=20 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0
```

**Apple Silicon Mac 上：**

将 `--device=cpu` 替换为 `--device=mps`，可以调用 M 系列芯片的 GPU 加速训练（通常能快 2-3 倍）。

### 3.3 复现 GPT-2 (124M)

需要 8 张 A100 40GB GPU，训练约 4 天：

```bash
# 先准备 OpenWebText 数据
python data/openwebtext/prepare.py

# 8 GPU 分布式训练
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

跨节点训练（例如 2 台机器各 8 卡）：

```bash
# 主节点（假设 IP 为 123.456.123.456）
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=123.456.123.456 --master_port=1234 train.py config/train_gpt2.py

# 工作节点
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=123.456.123.456 --master_port=1234 train.py config/train_gpt2.py
```

如果没有 InfiniBand 高速互联，需加上环境变量：`NCCL_IB_DISABLE=1`。

### 3.4 微调 GPT-2

微调与从头训练的区别仅在于：从预训练权重初始化，并使用更小的学习率。

```bash
# 先准备 BPE 版莎士比亚数据
python data/shakespeare/prepare.py

# 在 GPT-2 XL 上微调
python train.py config/finetune_shakespeare.py
```

配置文件 `config/finetune_shakespeare.py` 中：
- `init_from = 'gpt2-xl'` — 从最大 GPT-2 模型初始化
- `learning_rate = 3e-5` — 学习率远小于从头训练的 `6e-4`
- `decay_lr = False` — 微调阶段使用恒定学习率

### 3.5 关键配置参数速查

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `init_from` | 初始化方式 | `'scratch'` / `'resume'` / `'gpt2'` / `'gpt2-xl'` |
| `dataset` | 数据集子目录名 | `'shakespeare_char'` / `'openwebtext'` |
| `n_layer` | Transformer 层数 | GPT-2 124M: 12 |
| `n_head` | 注意力头数 | GPT-2 124M: 12 |
| `n_embd` | 嵌入维度 | GPT-2 124M: 768 |
| `block_size` | 上下文长度 | GPT-2: 1024 |
| `batch_size` | 微批次大小 | 12 |
| `gradient_accumulation_steps` | 梯度累积步数 | 5 × GPU 数量 |
| `max_iters` | 总训练迭代数 | GPT-2 124M: 600000 |
| `learning_rate` | 最大学习率 | 从头训练: 6e-4；微调: 3e-5 |
| `dropout` | Dropout 概率 | 预训练: 0.0；微调: 0.1+ |
| `compile` | 是否使用 torch.compile | 默认 True（需 PyTorch 2.0） |
| `device` | 计算设备 | `'cuda'` / `'cpu'` / `'mps'` |

---

## 4. 采样 / 推理

使用 `sample.py` 从训练好的模型或 OpenAI 官方 GPT-2 生成文本。

### 4.1 从自己训练的模型采样

```bash
python sample.py --out_dir=out-shakespeare-char
```

### 4.2 从预训练 GPT-2 采样

```bash
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 \
    --max_new_tokens=100
```

关键采样参数：
- `--temperature=0.8` — 温度（< 1 更保守，> 1 更随机）
- `--top_k=200` — 只保留概率最高的 top-k 个 token
- `--start="FILE:prompt.txt"` — 从文件读取提示词

### 4.3 评估预训练模型基线

```bash
python train.py config/eval_gpt2.py          # GPT-2 124M
python train.py config/eval_gpt2_medium.py   # GPT-2 350M
python train.py config/eval_gpt2_large.py    # GPT-2 774M
python train.py config/eval_gpt2_xl.py       # GPT-2 1.5B
```

各个模型的参数量和在 OpenWebText 上的 loss：

| 模型 | 参数量 | 训练 loss | 验证 loss |
|------|--------|-----------|------------|
| GPT-2 | 124M | 3.11 | 3.12 |
| GPT-2 Medium | 350M | 2.85 | 2.84 |
| GPT-2 Large | 774M | 2.66 | 2.67 |
| GPT-2 XL | 1.5B | 2.56 | 2.54 |

---

## 5. 代码架构

### 5.1 整体文件结构

```
nanoGPT/
├── train.py              # 训练脚本（核心）
├── model.py              # GPT 模型定义（核心）
├── sample.py             # 推理/采样脚本
├── bench.py              # 基准测试脚本
├── configurator.py       # 配置系统
├── config/               # 配置文件目录
│   ├── train_gpt2.py           # 复现 GPT-2 124M 的配置
│   ├── train_shakespeare_char.py # 字符级莎士比亚训练配置
│   ├── finetune_shakespeare.py  # 微调莎士比亚的配置
│   └── eval_gpt2*.py           # 评估各种 GPT-2 大小的配置
├── data/                 # 数据集目录
│   ├── shakespeare_char/ # 字符级莎士比亚（含 prepare.py）
│   ├── shakespeare/      # BPE 版莎士比亚（含 prepare.py）
│   └── openwebtext/      # OpenWebText（含 prepare.py）
├── scaling_laws.ipynb    # 缩放定律分析 Notebook
└── transformer_sizing.ipynb  # Transformer 规模估算 Notebook
```

### 5.2 配置系统（`configurator.py`）

nanoGPT 使用一种非传统的配置方式：`configurator.py` 被 `train.py`、`sample.py` 和 `bench.py` 通过 `exec()` 直接执行，从而覆写调用脚本中的全局变量。

```python
# 在 train.py 的顶层代码中：
exec(open('configurator.py').read())
```

两种覆写方式：
1. **配置文件**：`python train.py config/xxx.py` — 执行配置文件中的赋值语句
2. **命令行参数**：`python train.py --batch_size=32 --compile=False` — 直接覆写变量

**注意：** `configurator.py` 不是一个普通的 Python 模块，不能 `import`。它利用 `globals()` 直接修改调用脚本的全局命名空间。

### 5.3 模型结构（`model.py`）

```
GPT
├── transformer.wte      # Token 嵌入 (vocab_size → n_embd)
├── transformer.wpe      # 位置嵌入 (block_size → n_embd)
├── transformer.drop     # Dropout
├── transformer.h        # N 个 Block 的列表
│   └── Block
│       ├── ln_1         # LayerNorm（pre-norm）
│       ├── attn         # CausalSelfAttention
│       ├── ln_2         # LayerNorm（pre-norm）
│       └── mlp          # MLP (n_embd → 4*n_embd → n_embd)
├── transformer.ln_f     # 最终 LayerNorm
└── lm_head              # 输出投影 (n_embd → vocab_size)
```

关键设计要点：

- **权重绑定**：`wte`（token 嵌入）和 `lm_head`（输出层）共享权重，减少参数量
- **Flash Attention**：优先使用 PyTorch 的 `scaled_dot_product_attention`（PyTorch 2.0+），否则回退到手动实现
- **Pre-LayerNorm**：每个 Block 先做 LayerNorm 再做 Attention/MLP，然后残差连接
- **GELU 激活**：MLP 使用 GELU 激活函数，4 倍扩展比
- **自定义 LayerNorm**：支持 `bias=False` 参数（PyTorch 原生 LayerNorm 不支持关闭 bias）

### 5.4 训练循环（`train.py`）

```
获取第一批数据
for iter in range(max_iters):
    计算当前学习率（warmup → cosine decay）
    每 eval_interval 步评估 train/val loss 并保存 checkpoint
    
    for micro_step in range(gradient_accumulation_steps):
        forward pass（带自动混合精度）
        异步预取下一批数据
        backward pass（带 gradient scaling）
    
    梯度裁剪
    optimizer.step()
    optimizer.zero_grad()
    打印日志
```

训练循环的重要机制：

- **数据加载**：用 numpy `memmap` 读取 `.bin` 文件，每次随机取 `block_size` 长度的连续 token 序列。每个 batch 都会重新创建 memmap 以避免内存泄漏。
- **混合精度**：默认 `bfloat16`；如果硬件不支持则用 `float16` + `GradScaler`；也可以强制 `float32`
- **学习率调度**：线性 warmup（`warmup_iters` 步）→ 余弦衰减（最后降到 `min_lr`，通常为最大学习率的 1/10）
- **梯度累积**：通过 `gradient_accumulation_steps` 模拟更大的 batch size。DDP 模式下只最后一次 micro-step 同步梯度
- **MFU 估算**：`estimate_mfu()` 计算模型浮点运算利用率（相对于 A100 的 312 TFLOPS bfloat16 峰值）
- **Checkpoint**：保存模型权重、优化器状态、模型参数和配置，支持断点续训（`init_from='resume'`）

### 5.5 从 HuggingFace 加载权重（`GPT.from_pretrained`）

`model.py` 中的 `from_pretrained` 类方法可以加载 OpenAI 官方 GPT-2 权重。由于 OpenAI 的原始实现使用了 `Conv1D` 而非标准的 `nn.Linear`，加载时需要将以下层的权重进行**转置**：

- `attn.c_attn.weight`
- `attn.c_proj.weight`
- `mlp.c_fc.weight`
- `mlp.c_proj.weight`

---

## 6. 基准测试

```bash
python bench.py
```

`bench.py` 是 `train.py` 的精简版，只保留核心的训练循环（forward + backward + update），去掉评估、checkpoint、日志等额外逻辑，适合做模型速度的快速基准测试。支持 `--profile=True` 来使用 PyTorch Profiler 做更详细的性能分析。

---

## 7. 常见问题

**Q: 报错 `torch.compile` 相关错误？**

加上 `--compile=False`。`torch.compile` 是 PyTorch 2.0 的新特性，在某些平台（如 Windows）上可能不可用。

**Q: 运行时的 dtype 提示？**

默认使用 `bfloat16`，如果 GPU 不支持则自动回退到 `float16`。你也可以通过 `--dtype=float32` 强制使用全精度。

**Q: 如何查看所有可用的命令行参数？**

打开 `train.py` 文件，所有以大写字母开头的全局变量（如 `out_dir`、`batch_size`、`learning_rate` 等）都是可以通过命令行 `--key=value` 覆写的配置项。

---

## 8. 延伸阅读

- [Zero To Hero 系列教程](https://karpathy.ai/zero-to-hero.html) — Karpathy 的深度学习教学视频
- [GPT 讲解视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 从零实现 GPT 的视频教程
- [minGPT](https://github.com/karpathy/minGPT) — nanoGPT 的前身，更注重教学可读性
- [nanochat](https://github.com/karpathy/nanochat) — nanoGPT 的进化版（2025 年发布，推荐新项目使用）
