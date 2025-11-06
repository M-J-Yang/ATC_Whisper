"""
FastAPI 后端服务
提供语音识别推理接口
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import subprocess
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 添加父目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.inference_service import get_service

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 查找 FFmpeg 路径
def find_ffmpeg():
    """
    查找 FFmpeg 可执行文件路径
    """
    # 尝试在 PATH 中查找
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.info(f"在 PATH 中找到 FFmpeg: {ffmpeg_path}")
        return ffmpeg_path

    # 尝试在 Windows 的 WinGet 安装位置查找
    winget_path = Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")).glob("Gyan.FFmpeg*")
    for package_dir in winget_path:
        ffmpeg_exe = package_dir / "ffmpeg-*-full_build" / "bin" / "ffmpeg.exe"
        for match in package_dir.glob("**/ffmpeg.exe"):
            if match.exists():
                logger.info(f"在 WinGet 中找到 FFmpeg: {match}")
                return str(match)

    logger.error("未找到 FFmpeg")
    return "ffmpeg"  # 返回默认值，让 subprocess 尝试从 PATH 中找

FFMPEG_PATH = find_ffmpeg()

# 创建 FastAPI 应用
app = FastAPI(title="语音识别服务", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
inference_service = None
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)


class ConfigResponse(BaseModel):
    """配置响应模型"""
    status: str
    message: str
    config: dict


class InferenceResponse(BaseModel):
    """推理响应模型"""
    text: str
    duration: float
    inference_time: float
    total_time: float
    rtf: Optional[float]
    timestamp: str
    audio_path: str


@app.on_event("startup")
async def startup_event():
    """启动时的初始化"""
    logger.info("后端服务启动中...")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "语音识别服务运行中",
        "version": "1.0.0",
        "status": "ok"
    }


@app.post("/api/config/load", response_model=ConfigResponse)
async def load_model():
    """加载模型配置"""
    global inference_service

    try:
        logger.info("开始加载模型...")
        inference_service = get_service()
        config = inference_service.get_config()

        return ConfigResponse(
            status="success",
            message="模型加载成功",
            config={
                "model_path": inference_service.model_path,
                "language": inference_service.language,
                "device": inference_service.device,
                "vocab_constraint": inference_service.vocab_constraint is not None,
                "vocab_size": len(inference_service.vocab_constraint) if inference_service.vocab_constraint else 0
            }
        )
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@app.get("/api/config/status")
async def get_config_status():
    """获取配置状态"""
    if inference_service is None:
        return {
            "loaded": False,
            "message": "模型未加载"
        }

    return {
        "loaded": True,
        "model_path": inference_service.model_path,
        "language": inference_service.language,
        "device": inference_service.device,
        "vocab_constraint": inference_service.vocab_constraint is not None,
    }


@app.post("/api/inference/single", response_model=InferenceResponse)
async def inference_single(file: UploadFile = File(...)):
    """单条语音推理"""
    if inference_service is None:
        raise HTTPException(status_code=400, detail="模型未加载，请先加载模型")

    try:
        # 保存上传的文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_ext = Path(file.filename).suffix
        save_path = upload_dir / f"{timestamp}{file_ext}"

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"接收到文件: {file.filename}, 保存到: {save_path}")

        # 验证音频文件，如果格式不支持则尝试转换
        try:
            import torchaudio
            # 先尝试直接加载
            waveform, sr = torchaudio.load(str(save_path))
            logger.info(f"音频格式验证成功: {sr}Hz")
            audio_path = str(save_path)
        except Exception as load_error:
            logger.warning(f"原始文件格式不支持，尝试用 FFmpeg 转换: {load_error}")
            # 使用 FFmpeg 转换为标准 WAV 格式
            converted_path = str(save_path).replace(file_ext, ".wav")
            try:
                subprocess.run(
                    [
                        FFMPEG_PATH,
                        "-i", str(save_path),
                        "-ar", "16000",
                        "-ac", "1",
                        "-acodec", "pcm_s16le",
                        "-y",
                        converted_path
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30
                )
                logger.info(f"音频格式转换成功: {converted_path}")
                audio_path = converted_path
                # 删除原始文件
                Path(save_path).unlink(missing_ok=True)
            except Exception as ffmpeg_error:
                logger.error(f"FFmpeg 转换失败: {ffmpeg_error}")
                Path(save_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"无法处理该音频格式: {str(load_error)}"
                )

        # 执行推理
        result = inference_service.transcribe(
            audio_path,
            use_vocab_constraint=True
        )

        # 添加时间戳
        result["timestamp"] = datetime.now().isoformat()
        result["audio_path"] = file.filename

        logger.info(f"推理完成: {result['text']}")

        return InferenceResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


def convert_webm_to_wav(webm_data: bytes) -> str:
    """
    将 WebM 音频数据转换为 WAV 格式

    Args:
        webm_data: WebM 格式的音频数据

    Returns:
        str: 转换后的 WAV 文件路径
    """
    # 创建临时文件保存 WebM 数据
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
        webm_file.write(webm_data)
        webm_path = webm_file.name

    # 创建输出 WAV 文件路径
    wav_path = webm_path.replace(".webm", ".wav")

    try:
        # 使用 ffmpeg 转换格式
        # -i: 输入文件
        # -ar 16000: 采样率设置为 16kHz (Whisper 标准)
        # -ac 1: 单声道
        # -y: 覆盖输出文件
        subprocess.run(
            [
                FFMPEG_PATH,
                "-i", webm_path,
                "-ar", "16000",
                "-ac", "1",
                "-y",
                wav_path
            ],
            check=True,
            capture_output=True,
            timeout=10
        )

        # 删除临时 WebM 文件
        Path(webm_path).unlink()

        return wav_path

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 转换失败: {e.stderr.decode()}")
        # 清理临时文件
        Path(webm_path).unlink(missing_ok=True)
        Path(wav_path).unlink(missing_ok=True)
        raise Exception(f"音频格式转换失败: {e.stderr.decode()}")
    except FileNotFoundError:
        logger.error("未找到 ffmpeg，请确保已安装 ffmpeg")
        Path(webm_path).unlink(missing_ok=True)
        raise Exception("未找到 ffmpeg，请安装 ffmpeg 后重试")
    except Exception as e:
        logger.error(f"转换过程出错: {e}")
        Path(webm_path).unlink(missing_ok=True)
        Path(wav_path).unlink(missing_ok=True)
        raise


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """实时语音识别 WebSocket 端点"""
    await websocket.accept()
    logger.info("WebSocket 连接已建立")

    if inference_service is None:
        await websocket.send_json({
            "error": "模型未加载，请先加载模型"
        })
        await websocket.close()
        return

    try:
        chunk_count = 0
        while True:
            # 接收音频数据 (WebM 格式)
            data = await websocket.receive_bytes()
            chunk_count += 1

            logger.info(f"接收到音频数据块 {chunk_count}, 大小: {len(data)} bytes")

            # 转换 WebM 到 WAV
            try:
                wav_path = convert_webm_to_wav(data)
                logger.info(f"音频格式转换成功: {wav_path}")
            except Exception as e:
                logger.error(f"音频转换失败: {e}")
                await websocket.send_json({
                    "error": f"音频转换失败: {str(e)}",
                    "chunk_id": chunk_count
                })
                continue

            # 执行推理
            try:
                result = inference_service.transcribe(
                    wav_path,
                    use_vocab_constraint=True
                )

                # 发送结果
                response = {
                    "text": result["text"],
                    "duration": result["duration"],
                    "inference_time": result["inference_time"],
                    "timestamp": datetime.now().isoformat(),
                    "chunk_id": chunk_count
                }

                await websocket.send_json(response)
                logger.info(f"发送识别结果: {result['text']}")

            except Exception as e:
                logger.error(f"推理失败: {e}")
                await websocket.send_json({
                    "error": f"推理失败: {str(e)}",
                    "chunk_id": chunk_count
                })
            finally:
                # 删除临时文件
                Path(wav_path).unlink(missing_ok=True)

    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except:
            pass


@app.delete("/api/uploads/clear")
async def clear_uploads():
    """清理上传文件"""
    try:
        count = 0
        for file in upload_dir.glob("*"):
            if file.is_file():
                file.unlink()
                count += 1

        return {
            "status": "success",
            "message": f"已清理 {count} 个文件"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
