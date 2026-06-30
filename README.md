# Kaggle Disaster Tweets NLP

本项目是一个基于 Kaggle Disaster Tweets 数据集的 NLP 练习项目。

任务目标是根据一条 tweet 的内容，判断它是否在描述真实灾害。本项目的主要目的是搭建一个可复现、可扩展的 NLP 实验流程：先从传统的 TF-IDF baseline 开始，再逐步扩展到 BERT 等 Transformer 模型。

## 当前进度

- 完成了 `TF-IDF + Logistic Regression` baseline。
- 对比了多个特征与参数变体，包括：
  - 是否加入 `keyword`
  - 是否移除英文 stop words
  - Logistic Regression 的正则化参数 `C`
  - F1 指标下的 threshold tuning
- 将原本分散的 TF-IDF 实验整理为统一的训练脚本。
- 记录了本地交叉验证结果和 Kaggle public score，用于后续实验对比。

## 项目结构

```text
.
├── data/
├── outputs/
│   ├── submissions/
│   └── archive/
├── scripts/
│   ├── train_tfidf.py
│   └── archive/
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   └── model.py
├── README.md
├── pyproject.toml
└── .gitignore
```

## 运行方式

运行一个 TF-IDF 实验：

```bash
python scripts/train_tfidf.py --experiment v1_baseline
```

生成提交文件：

```bash
python scripts/train_tfidf.py --experiment v1_baseline --make-submission
```

进行 threshold tuning：

```bash
python scripts/train_tfidf.py --experiment v1_baseline --tune-threshold
```

## 当前实验结论

目前在已提交的传统 NLP 模型中，public score 最好的版本是 text-only 的 `TF-IDF + Logistic Regression` baseline。

一些本地 CV 分数更高的变体，例如加入 `keyword`、不移除 stop words、调整 `C`、以及 threshold tuning，并没有带来更高的 Kaggle public score。因此，这些实验被记录为有价值的 ablation 结果，但暂时不作为当前最优提交方案。

这也说明，本地交叉验证和 Kaggle public leaderboard 之间可能存在一定分布差异。后续实验会继续保留本地 CV 作为主要离线评估方式，同时谨慎使用 public score 作为外部验证。

## 后续计划

下一阶段计划构建 Transformer-based baseline，例如 DistilBERT 或 BERT，并与当前 TF-IDF baseline 进行对比。

后续重点包括：

1. 搭建 tokenizer、dataset 和 dataloader 流程。
2. Fine-tune 一个轻量级 Transformer 分类模型。
3. 比较 Transformer 模型与传统 TF-IDF 模型在 CV F1 和 Kaggle public score 上的差异。
4. 进行错误分析，观察模型在哪些类型的 tweet 上容易误判。
5. 将实验结果整理进 README，形成完整的 NLP Kaggle 项目记录。