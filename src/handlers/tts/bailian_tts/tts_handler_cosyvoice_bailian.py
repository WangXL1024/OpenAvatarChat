import io
import os
import re
import time
from typing import Dict, Optional, cast
import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.directory_info import DirectoryInfo
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
import dashscope


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    ref_audio_path: str = Field(default=None)
    ref_audio_text: str = Field(default=None)
    voice: str = Field(default=None)
    sample_rate: int = Field(default=24000)
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    model_name: str = Field(default="cosyvoice-1")


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        self.synthesizer = None


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

        self.ref_audio_path = None
        self.ref_audio_text = None
        self.voice = None
        self.ref_audio_buffer = None
        self.sample_rate = None
        self.model_name = None
        self.api_key = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        config = cast(TTSConfig, handler_config)
        print(">>>>>>>>>>>>>>>>>>Bailian TTS Handler loaded(Just one time)<<<<<<<<<<<<<<<<<<")
        self.voice = config.voice
        self.sample_rate = config.sample_rate
        self.ref_audio_path = config.ref_audio_path
        self.ref_audio_text = config.ref_audio_text
        self.model_name = config.model_name
        if 'DASHSCOPE_API_KEY' in os.environ:
            # load API-key from environment variable DASHSCOPE_API_KEY
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            dashscope.api_key = config.api_key  # set API-key manually

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.input_text = ''
        if context.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp',
                                          f"dump_avatar_audio_{context.session_id}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pcm")
            context.audio_dump_file = open(dump_file_path, "wb")
        return context

    def start_context(self, session_context, context: HandlerContext):
        context = cast(TTSContext, context)

    def filter_text(self, text):
        pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?，。！？ ]"  # 匹配不在范围内的字符
        filtered_text = re.sub(pattern, "", text)
        return filtered_text

    # add by wangxl@20251203 for get user voice setting from user_settings.json
    @staticmethod
    def get_user_setting(
        user_id: str = "default_user",
        field: str = "sound_id",  # 新增参数：指定要获取的字段（sound_id或video_id）
        settings_file: str = "user_settings.json"
    ) -> str | None:
        import json
        import os
        """
        从 user_settings.json 中读取指定用户的配置字段（支持sound_id和video_id）
        
        Args:
            user_id: 要查询的用户ID（如 "default_user"）
            field: 要获取的字段名（支持 "sound_id" 或 "video_id"）
            settings_file: 配置文件路径
        
        Returns:
            成功：返回用户对应的字段值（字符串）
            失败：返回 None（文件不存在、用户不存在、字段不存在等情况）
        """
        # 固定配置文件路径
        settings_file = "/core/dt_avatar/code/OpenAvatarSetting/user_settings.json"

        # 检查文件是否存在
        if not os.path.exists(settings_file):
            print(f"警告：配置文件 {settings_file} 不存在")
            return None
        
        try:
            # 读取并解析JSON文件
            with open(settings_file, "r", encoding="utf-8") as f:
                try:
                    settings_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"错误：{settings_file} 是无效的JSON文件")
                    return None
            
            # 检查用户是否存在
            if user_id not in settings_data:
                print(f"警告：用户 {user_id} 不存在于配置文件中")
                return None
            
            user_info = settings_data[user_id]
            # 检查用户信息是否为字典且包含目标字段
            if not isinstance(user_info, dict):
                print(f"警告：用户 {user_id} 的配置格式错误（非字典类型）")
                return None
            if field not in user_info:
                print(f"警告：用户 {user_id} 的配置中不包含 {field} 字段")
                return None
            
            # 返回字段值（确保为字符串）
            return str(user_info[field])
        
        except PermissionError:
            print(f"错误：没有读取 {settings_file} 的权限")
            return None
        except Exception as e:
            print(f"错误：读取用户配置失败 - {str(e)}")
            return None
        
    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        user_voice = self.get_user_setting("default_user", "sound_id")
        self.voice = user_voice if user_voice is not None else self.voice
        print(f"use the voice id{self.voice}")
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)

        text_end = inputs.data.get_meta("avatar_text_end", False)
        try:
            if not text_end:
                if context.synthesizer is None:
                    callback = CosyvoiceCallBack(
                        context=context, output_definition=output_definition, speech_id=speech_id)
                    context.synthesizer = SpeechSynthesizer(
                        model=self.model_name, voice=self.voice, callback=callback, format=AudioFormat.PCM_24000HZ_MONO_16BIT)
                logger.info(f'streaming_call {text}')
                context.synthesizer.streaming_call(text)
            else:
                logger.info(f'streaming_call last {text}')
                context.synthesizer.streaming_call(text)
                context.synthesizer.streaming_complete()
                context.synthesizer = None
                context.input_text = ''
        except Exception as e:
            logger.error(e)
            context.synthesizer.streaming_complete()
            context.synthesizer = None

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        logger.info('destroy context')


class CosyvoiceCallBack(ResultCallback):
    def __init__(self, context: TTSContext, output_definition, speech_id):
        super().__init__()
        self.context = context
        self.output_definition = output_definition
        self.speech_id = speech_id
        self.temp_bytes = b''

    def on_open(self) -> None:
        logger.info('连接成功')

    def on_event(self, message) -> None:
        # 实现接收合成结果的逻辑
        # logger.info(message)
        pass

    def on_data(self, data: bytes) -> None:
        self.temp_bytes += data
        if len(self.temp_bytes) > 24000:
            # 实现接收合成二进制音频结果的逻辑
            output_audio = np.array(np.frombuffer(self.temp_bytes, dtype=np.int16)).astype(
                np.float32)/32767  # librosa.load(io.BytesIO(self.temp_bytes), sr=None)[0]
            output_audio = output_audio[np.newaxis, ...]
            output = DataBundle(self.output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", False)
            output.add_meta("speech_id", self.speech_id)
            self.context.submit_data(output)
            self.temp_bytes = b''

    def on_complete(self) -> None:
        if len(self.temp_bytes) > 0:
            output_audio = np.array(np.frombuffer(self.temp_bytes, dtype=np.int16)).astype(np.float32)/32767
            output_audio = output_audio[np.newaxis, ...]
            output = DataBundle(self.output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", False)
            output.add_meta("speech_id", self.speech_id)
            self.context.submit_data(output)
            self.temp_bytes = b''
        output = DataBundle(self.output_definition)
        output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
        output.add_meta("avatar_speech_end", True)
        output.add_meta("speech_id", self.speech_id)
        self.context.submit_data(output)
        logger.info(f"speech end")
        logger.info('合成完成')

    def on_error(self, message) -> None:
        logger.error(f'bailian tts 服务出现异常,请确保参数正确：${message}')
        output = DataBundle(self.output_definition)
        output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
        output.add_meta("avatar_speech_end", True)
        output.add_meta("speech_id", self.speech_id)
        self.context.submit_data(output)
        logger.info(f"speech end")

    def on_close(self) -> None:
        logger.info('连接关闭')
