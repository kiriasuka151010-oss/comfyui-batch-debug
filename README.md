# ComfyUI Batch Debug Plugin

批量参数扫描插件 — 笛卡尔积扫描 LoRA 权重 / CFG / Seed / Steps，自动生成对比网格图 + CSV 元数据。

## 文件

```
comfyui-batch-debug/
├── __init__.py              节点注册
├── nodes.py                 三个节点实现
├── utils.py                 工具函数
├── requirements.txt         依赖（零额外依赖）
├── docs.html                完整文档（浏览器打开）
├── smoke_test_workflow.json 烟雾测试工作流
└── README.md                本文件
```

## 三个节点

| 节点 | 作用 |
|------|------|
| BatchDebugConfig | 参数扫描范围配置 |
| BatchDebugExecute | 批量执行引擎（conditioning 直通 / reencode 双模式） |
| BatchDebugGridSave | 结果保存（单张图 + 网格图 + CSV） |

## 安装

复制整个文件夹到 `ComfyUI/custom_nodes/comfyui-batch-debug/`，重启 ComfyUI。

## 快速测试

1. 拖入 `smoke_test_workflow.json`
2. 点 Queue
3. 查看 `ComfyUI/output/batch_debug/smoke_test/`

## 文档

`docs.html` — 浏览器打开，含完整参数表和 Anima 模型推荐工作流。
