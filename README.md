# video2zh — 视频自动转中文配音

一条命令把视频变成中文配音版：

```
视频 → 提取音频(ffmpeg) → 转写(faster-whisper, GPU) → 翻译(claude -p 无头模式) → 配音(edge-tts)或者加入软字幕 → 合成视频
```

## 依赖

- `ffmpeg’
- 需安装python pip：`faster-whisper numpy edge-tts demucs`
- `claude` 翻译步骤走 Claude Code 订阅，消耗少量配额）

## 用法

```bash
python ~/video2zh/video2zh.py <视频文件>

# 常用选项
--outdir ~/video/out      # 输出目录（默认与视频同目录）
--voice zh-CN-XiaoxiaoNeural   # 换女声（默认 zh-CN-YunyangNeural 男声）
--model small              # 转写更快、精度略低（默认 medium；large-v3 最准）
--tts                      # 合成语音，默认不合成
```

## 产物（与视频同目录）

| 文件 | 说明 |
|------|------|
| `<名称>_zh.mp4` | 成品
| `<名称>_zh.srt` | 中文字幕 |
| `<名称>_transcript.tsv` | 原文转写（含时间戳） |
| `<名称>_translation_zh.tsv` | 中文翻译（含时间戳） |
| `separated/htdemucs/<名称>` | --tts 时，demucs 分离的背景音和人声

## 说明

- 首次运行会下载 Whisper 模型（约 1.5GB）到 `~/.cache/huggingface`
- edge-tts 调用微软免费语音服务，文本会发送到其服务器
- 配音超出原始时间槽时会自动微加速（上限 1.35x）以保持与画面同步

如果播放器不支持软字幕，可以考虑 ffmpeg 加入硬字幕，这种方式需要重新编码视频流，速度比较慢。

```bash
ffmpeg -i input.mp4 -c copy -sn input2.mp4  # 复制但是排除软字幕流
ffmpeg -i input2.mp4 -vf "subtitles=sub.srt" -c:v libx264 -preset fast -crf 22 -c:a copy -movflags +faststart output.mp4
```
