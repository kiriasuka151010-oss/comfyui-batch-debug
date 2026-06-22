# ComfyUI Batch Debug

> 批量参数扫描插件 — 多 LoRA 链式笛卡尔积扫描 + 自适应评分查看器

[![version](https://img.shields.io/badge/version-2.1-green)]()

## 一句话

把 AI 绘图调参从"手动改参数 → 点 Queue → 记笔记"变成"配置一次 → 批量扫描 → HTML 打分筛选"。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kiriasuka151010-oss/comfyui-batch-debug.git
```

重启 ComfyUI，在节点菜单中找到 `batch_debug` 分类。

## 三个节点

| 节点 | 作用 |
|------|------|
| **BatchDebugConfig** | 6 槽 LoRA 配置（下拉选取 + 开关 + 权重范围）+ CFG + Steps 扫描范围 |
| **BatchDebugExecute** | 笛卡尔积执行引擎：链式多 LoRA → conditioning 直通/重编码 → KSampler → VAE |
| **BatchDebugGridSave** | 结构化输出：单张图 + 拼图 + CSV + report.json + 评分查看器 |

## 快速开始

1. 拖入 `workflows/anima-danbooru-batch.json`（D 站 Pipeline 工作流）
2. 在 AnimaDexBrowser 选 artist → DanbooruBrowser 选参考图
3. BatchDebugConfig 里开 1-2 个 LoRA 槽，设权重范围
4. 点 Queue，跑完去 `output/batch_debug/` 找到 sweep 文件夹
5. 打开 `viewer.html` → 拖入 `report.json` → 打分筛选

## 工作流文件

| 文件 | 用途 |
|------|------|
| `workflows/anima-danbooru-batch.json` | D 站 Pipeline + 批量扫描（推荐） |
| `workflows/anima-smoke-test.json` | 最简测试工作流 |

## 评分查看器

`viewer.html` 自适应变量数自动排版：
- **1 变量** → 分组排列
- **2 变量** → 矩阵对比
- **3+ 变量** → 嵌套分组

支持 ⭐ 评分 / 🔍 筛选 / 📥 导出 CSV / 🖼 双击放大。

## 技术要点

- 多 LoRA 链式叠加（`load_lora_for_models` 自动 clone，不污染原始模型）
- Conditioning 直通模式（上游 AnimaPromptConverter 的 conditioning 原样传给 KSampler）
- 去重保护（min==max 自动去重）
- 中断安全（try/finally + LoRA 缓存清理）
- 零额外 pip 依赖（仅 ComfyUI 内置 torch/numpy/PIL）

## 文档

`docs.html` — 浏览器打开，含完整参数表和技术说明。
