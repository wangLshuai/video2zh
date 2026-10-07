# video2zh — 视频自动转中文配音

一条命令把视频变成中文配音版：

```
视频 → 提取音频(ffmpeg) → 转写(faster-whisper, GPU) → 翻译(claude -p 无头模式) → 配音(edge-tts) → 合成视频
```

## 依赖

- `ffmpeg`（系统已装）
- venv Python `~/python-venv/python`，需安装：`faster-whisper numpy edge-tts`
- `claude` CLI 已登录（翻译步骤走你的 Claude Code 订阅，消耗少量配额）

## 用法

```bash
~/python-venv/python/bin/python ~/video2zh/video2zh.py <视频文件>

# 常用选项
--outdir ~/video/out      # 输出目录（默认与视频同目录）
--voice zh-CN-XiaoxiaoNeural   # 换女声（默认 zh-CN-YunyangNeural 男声）
--model small              # 转写更快、精度略低（默认 medium；large-v3 最准）
--no-mux                   # 只出音频+字幕，不合成视频
```

## 产物（与视频同目录）

| 文件 | 说明 |
|------|------|
| `<名称> 中文配音.mp4` | 成品：原视频轨 + 按原时间点对齐的中文配音 |
| `<名称>_zh_narration.mp3` | 连续版纯配音（通读，无停顿） |
| `<名称>_zh.srt` | 中文字幕 |
| `<名称>_transcript_en.tsv` | 原文转写（含时间戳） |
| `<名称>_translation_zh.tsv` | 中文翻译（含时间戳） |

## 说明

- 首次运行会下载 Whisper 模型（约 1.5GB）到 `~/.cache/huggingface`
- 本机 CUDA 13 与 ctranslate2 的兼容问题（libcublas.so.12）由脚本自动处理（符号链接 shim + 自重启）
- edge-tts 调用微软免费语音服务，文本会发送到其服务器
- 配音超出原始时间槽时会自动微加速（上限 1.35x）以保持与画面同步
