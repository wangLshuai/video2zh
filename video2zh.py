#!/home/wang/python-venv/python/bin/python
"""video2zh — 视频自动转中文配音流水线

用法:
  video2zh.py <视频文件> [--outdir 输出目录] [--model medium] [--voice zh-CN-YunyangNeural]
             [--batch 60] [--no-mux] [--keep-temp]

步骤:
  1. ffmpeg 提取 16kHz 单声道音频
  2. faster-whisper (GPU) 转写, 输出 <名称>_transcript_en.tsv
  3. claude -p 无头模式逐批翻译, 输出 <名称>_translation_zh.tsv (消耗 Claude 配额)
  4. edge-tts 逐句合成中文配音, 按原时间点对齐 (超长句自动微加速)
  5. 合成 <名称> 中文配音.mp4 (视频轨原样保留, 音频轨换成中文)

产物: <名称> 中文配音.mp4 / <名称>_zh_narration.mp3 (连续版配音) / <名称>_zh.srt (中文字幕)
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import wave


def ensure_cuda_shim():
    """本机是 CUDA 13 而 ctranslate2 按 CUDA 12 编译: 缺 libcublas.so.12 时建符号链接
    shim 并重启自身 (动态链接器只在进程启动时读 LD_LIBRARY_PATH)。"""
    cuda = "/opt/cuda/lib64"
    if not os.path.exists(f"{cuda}/libcublas.so.13"):
        return
    try:
        ld = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
        if "libcublas.so.12" in ld:
            return
    except FileNotFoundError:
        pass
    shim = "/tmp/cuda12-shim"
    os.makedirs(shim, exist_ok=True)
    for lib in ("cublas", "cublasLt", "cudart"):
        dst = f"{shim}/lib{lib}.so.12"
        if not os.path.lexists(dst):
            try:
                os.symlink(f"{cuda}/lib{lib}.so.13", dst)
            except OSError:
                pass
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if shim not in cur.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(x for x in (shim, cur) if x)
        os.execv(sys.executable, [sys.executable] + sys.argv)


ensure_cuda_shim()

import numpy as np

SR = 16000
TRANSLATE_PROMPT = (
    '把下面的每一行翻译成简体中文。每行格式是"起始秒\\t结束秒\\t文本"，'
    "必须保持三列制表符分隔的格式完全不变，只翻译第三列的文本内容。要求："
    "1) 只输出翻译后的行，不要任何解释、标题或代码块；"
    "2) 输出行数必须与输入行数完全相同；"
    '3) 数字写成中文读法（如 72 写成"七十二"）；'
    "4) 专业术语使用标准译法（如 bevel gear=锥齿轮，pinion=小齿轮，module=模数，"
    "sketch=草图，boolean=布尔，workbench=工作台，pitch circle=节圆，backlash=侧隙）。"
)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def extract_audio(video, wav):
    p = run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            video,
            "-map",
            "0:a:0",
            "-vn",
            "-ar",
            str(SR),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            wav,
        ]
    )
    if p.returncode != 0:
        raise SystemExit(f"无法提取音频（视频里没有音轨？）: {p.stderr.strip()[:300]}")
    return wav


def video_duration(video):
    p = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video,
        ]
    )
    return float(p.stdout.strip())


def transcribe(wav, model_name):
    with wave.open(wav, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    data = data.astype(np.float32) / 32768.0

    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
    except RuntimeError as e:
        print(f"警告: GPU 初始化失败({e})，回退到 CPU int8", file=sys.stderr)
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count() or 4,
        )
    segments, info = model.transcribe(data, beam_size=5, vad_filter=True)
    print(
        f"转写完成: 语言={info.language} (置信度 {info.language_probability:.2f})",
        flush=True,
    )
    return [f"{s.start:.2f}\t{s.end:.2f}\t{s.text.strip()}" for s in segments]


def sanitize_tsv(line):
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    try:
        float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return parts[0] + "\t" + parts[1] + "\t" + "\t".join(parts[2:]).strip()


def translate(lines, batch):
    if not shutil.which("claude"):
        raise SystemExit("找不到 claude 命令，请先安装并登录 Claude Code")
    out = []
    n_batches = (len(lines) + batch - 1) // batch
    for i in range(0, len(lines), batch):
        chunk = lines[i : i + batch]
        done = False
        for attempt in range(3):
            try:
                p = run(
                    ["claude", "-p", "--output-format", "text", TRANSLATE_PROMPT],
                    input="\n".join(chunk) + "\n",
                    timeout=600,
                )
            except subprocess.TimeoutExpired:
                print(f"翻译批 {i // batch + 1} 超时，重试", file=sys.stderr)
                continue
            if p.returncode != 0:
                print(
                    f"翻译批 {i // batch + 1} 第{attempt + 1}次失败: "
                    f"{p.stderr.strip()[:200]}",
                    file=sys.stderr,
                )
                continue
            got = [x for x in (sanitize_tsv(l) for l in p.stdout.splitlines()) if x]
            if len(got) == len(chunk):
                out.extend(got)
                done = True
                break
            print(
                f"翻译批 {i // batch + 1} 行数不符({len(got)}/{len(chunk)})，重试",
                file=sys.stderr,
            )
        if not done:
            raise SystemExit(
                f"翻译失败（第 {i // batch + 1} 批，3 次重试后放弃）。"
                f"转写文件已保留，可单独重跑翻译。"
            )
        print(f"翻译批 {i // batch + 1}/{n_batches} 完成", flush=True)
    return out


async def synth_all(segs, segdir, voice):
    import edge_tts

    sem = asyncio.Semaphore(3)

    async def one(i, text, out):
        async with sem:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(text, voice).save(out)
                    return
                except Exception as e:
                    print(f"片段 {i} 合成第{attempt + 1}次失败: {e}", file=sys.stderr)
                    await asyncio.sleep(2)
            raise RuntimeError(f"片段 {i} 合成失败")

    await asyncio.gather(
        *(one(i, t, f"{segdir}/{i:04d}.mp3") for i, (_, _, t) in enumerate(segs))
    )
    print(f"配音合成完成: {len(segs)} 段", flush=True)


def decode_mp3(path, tempo=None):
    args = ["ffmpeg", "-v", "error", "-i", path]
    if tempo:
        args += ["-af", f"atempo={tempo:.4f}"]
    args += ["-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    p = subprocess.run(args, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 解码失败: {p.stderr.decode(errors='replace')[:300]}"
        )
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def save_wav(path, arr):
    pcm = (np.clip(arr, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def srt_time(t):
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_and_encode(segs, segdir, dur, outdir, stem, tmp):
    synced = np.zeros(int((dur + 2) * SR), dtype=np.float32)
    parts = []
    for i, (a, b, _) in enumerate(segs):
        mp3 = f"{segdir}/{i:04d}.mp3"
        slot = b - a
        x = decode_mp3(mp3)
        if len(x) > slot * SR:  # 超出时间槽: 加速(上限1.35x), 仍超则截断
            tempo = min(len(x) / (slot * SR), 1.35)
            x = decode_mp3(mp3, tempo)
            x = x[: int(slot * SR)]
        start = int(a * SR)
        synced[start : start + len(x)] = x
        parts.append(x)
    cont = np.concatenate(parts)
    print(f"配音对齐完成: 连续版 {len(cont) / SR:.1f}s / 同步版 {dur:.1f}s", flush=True)

    synced_wav = f"{tmp}/synced.wav"
    narration_wav = f"{tmp}/narration.wav"
    save_wav(synced_wav, synced)
    save_wav(narration_wav, cont)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            narration_wav,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            f"{outdir}/{stem}_zh_narration.mp3",
        ],
        check=True,
    )
    synced_m4a = f"{outdir}/{stem}_zh_synced.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            synced_wav,
            "-codec:a",
            "aac",
            "-b:a",
            "128k",
            synced_m4a,
        ],
        check=True,
    )
    return synced_m4a


def write_srt(lines, path):
    with open(
        path, "w", encoding="utf-8-sig"
    ) as f:  # 带 BOM, 避免 VLC 等播放器编码检测失败
        for i, line in enumerate(lines, 1):
            a, b, text = line.split("\t", 2)
            f.write(f"{i}\n{srt_time(float(a))} --> {srt_time(float(b))}\n{text}\n\n")


def main():
    ap = argparse.ArgumentParser(description="视频自动转中文配音流水线")
    ap.add_argument("video", help="输入视频文件")
    ap.add_argument("--outdir", default=None, help="输出目录（默认与视频同目录）")
    ap.add_argument(
        "--model",
        default="medium",
        help="Whisper 模型 (tiny/base/small/medium/large-v3)",
    )
    ap.add_argument("--voice", default="zh-CN-YunyangNeural", help="edge-tts 音色")
    ap.add_argument("--batch", type=int, default=60, help="每次 claude 翻译的行数")
    ap.add_argument(
        "--no-mux", action="store_true", help="只生成音频和字幕，不合成视频"
    )
    ap.add_argument("--keep-temp", action="store_true", help="保留中间文件")
    ap.add_argument("--no-tts", action="store_true", help="不进行tts 步骤")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        raise SystemExit(f"文件不存在: {video}")
    outdir = args.outdir or os.path.dirname(video)
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    tmp = tempfile.mkdtemp(prefix="video2zh-")
    try:
        print(f"输入: {video}", flush=True)
        print("1/5 提取音频 ...", flush=True)
        wav = f"{tmp}/audio.wav"
        extract_audio(video, wav)

        print("2/5 语音转写 ...", flush=True)
        lines = transcribe(wav, args.model)
        with open(f"{outdir}/{stem}_transcript.tsv", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print("3/5 claude 翻译 ...", flush=True)
        zh_lines = translate(lines, args.batch)
        with open(f"{outdir}/{stem}_translation_zh.tsv", "w", encoding="utf-8") as f:
            f.write("\n".join(zh_lines) + "\n")
        write_srt(zh_lines, f"{outdir}/{stem}_zh.srt")
        out_video = f"{outdir}/{stem}_zh.mp4"
        if not args.no_tts:
            print("4/5 edge-tts 配音合成与对齐 ...", flush=True)
            segs = []
            for l in zh_lines:
                a, b, text = l.split("\t", 2)
                segs.append((float(a), float(b), text))
            segdir = f"{tmp}/segs"
            os.makedirs(segdir, exist_ok=True)
            asyncio.run(synth_all(segs, segdir, args.voice))
            dur = video_duration(video)
            synced_m4a = build_and_encode(segs, segdir, dur, outdir, stem, tmp)
            print("5/5 合成视频 ...", flush=True)
            if not args.no_mux:

                p = run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        video,
                        "-i",
                        synced_m4a,
                        "-map",
                        "0:v",
                        "-map",
                        "1:a",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "copy",
                        "-shortest",
                        out_video,
                    ]
                )
                if p.returncode != 0:
                    raise SystemExit(f"合成视频失败: {p.stderr.strip()[:300]}")
        else:
            print("加入字幕")
            srt = f"{outdir}/{stem}_zh.srt"
            vf = "subtitles='" + srt.replace("'", r"'\''") + "'"
            p = run(
                # ["ffmpeg", "-i", video, "-vf", vf, "-c:a", "copy", out_video]
                [
                    "ffmpeg",
                    "-i",
                    video,
                    "-i",
                    srt,
                    "-map",
                    "0:v",
                    "-map",
                    "0:a",
                    "-map",
                    "1:s:0",
                    "-c",
                    "copy",
                    "-c:s",
                    "mov_text",
                    "-disposition:s:0",
                    "default",
                    "-metadata:s:s:0",
                    "language=chi",
                    out_video,
                ]
            )
            if p.returncode != 0:
                raise SystemExit(f"合成视频失败: {p.stderr}")

        print("\n完成。产物:", flush=True)
        for name in (["_zh.mp4"] if not args.no_mux else []) + [
            "_zh.srt",
            "_transcript.tsv",
            "_translation_zh.tsv",
        ]:
            path = (
                f"{outdir}/{stem} {name}"
                if name == "中文配音.mp4"
                else f"{outdir}/{stem}{name}"
            )
            size = os.path.getsize(path) / 1048576
            print(f"  {path}  ({size:.1f} MB)", flush=True)
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
