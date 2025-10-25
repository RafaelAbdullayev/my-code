import whisperx
import os
import noisereduce as nr
import numpy as np
import logging
import json
import io
import sys
import sqlite3
import aiohttp
import uuid
import pytz
import locale
import asyncio
import aiofiles
import requests
import re
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from calendar_client import add_event_to_calendar
# 📌 ИСПРАВЛЕНО: импортируем get_prompts_with_dates из llm_client для админки
from llm_client import query_llm, set_model, get_model, get_prompts_with_dates as get_llm_prompts
from event_parser import (
    analyze_transcript,
    format_plan_response,
    format_today_response,
    format_tomorrow_response,
    format_last_week_response,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pydub.silence import split_on_silence
from pydub import AudioSegment, effects
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from aiogram.utils.keyboard import InlineKeyboardBuilder
from language_tool_python import LanguageTool

# === WhisperX Settings ===
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

device = "cuda"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка WhisperX модели
whisper_model = whisperx.load_model(
    "large-v3",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "float32",
    language="ru"
)

logger.info("✅ Модель WhisperX загружена!")

# === Функция для транскрипции и выравнивания ===
tool = LanguageTool('ru-RU')

# === Функция для исправления грамматики ===
def correct_grammar(text: str) -> str:
    """
    Исправляет грамматические и пунктуационные ошибки в тексте
    """
    try:
        # Исправляем ошибки с помощью LanguageTool
        matches = tool.check(text)
        corrected_text = tool.correct(text)
        
        # Дополнительные ручные исправления для специфических случаев
        corrections = {
            r'\bс (\w+[ое]м)\b': r'со \1',  # с стасом -> со стасом
            r'\bс (\w+[ие]й)\b': r'с \1',   # сохраняем правильные формы
            r'\bв (\w+[ое]м)\b': r'во \1',  # в всем -> во всем
            r'(\w),(\w)': r'\1, \2',        # добавляем пробелы после запятых
            r'(\w)\.(\w)': r'\1. \2',       # добавляем пробелы после точек
            r'\s+': ' ',                     # убираем лишние пробелы
            r'\s([.,!?;:])': r'\1',          # убираем пробелы перед знаками препинания
            r'([a-zA-Zа-яА-Я])([A-ZА-Я])': r'\1. \2'  # добавляем точки между инициалами
        }
        
        for pattern, replacement in corrections.items():
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        return corrected_text.strip()
    
    except Exception as e:
        logger.error(f"❌ Ошибка при исправлении грамматики: {e}")
        return text  # Возвращаем оригинальный текст в случае ошибки

# === Удалена дублированная функция process_audio ===

# === Timezone, Locale ===
# === Timezone, Locale ===
# 📌 ИСПРАВЛЕНО: Удалены глобальные даты, вычислявшиеся один раз при импорте
# Теперь используем get_current_dates() для динамического получения актуальных дат
tz = pytz.timezone("Europe/Moscow")
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


# === Load config ===
load_dotenv()
MODEL_NAME = "llama3:8b-instruct-q4_K_M"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

TOKEN = config.get("telegram_token")
UPLOAD_DIR = Path(config.get("upload_dir", "upload"))
OUTPUT_DIR = Path(config.get("output_dir", "output"))
ALLOWED_USERS = set(config.get("allowed_users", []))
ADMINS = set(config.get("admins", []))

def is_admin(user_id: int) -> bool:
    return user_id in ADMINS

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Database ===
db_path = "bot_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, full_name TEXT);''')
cursor.execute('''CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, user_id INTEGER, file_name TEXT, file_path TEXT, transcribed_text TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (id));''')
conn.commit()

# === Bot Setup ===
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())
admin_state = {}  # Хранилище действий администратора

MAX_CHUNK_LENGTH_MS = 60 * 1000  # 1 минута

def is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USERS

def save_user_info(user_id: int, username: str, full_name: str):
    cursor.execute(
        'INSERT OR IGNORE INTO users (id, username, full_name) VALUES (?, ?, ?)',
        (user_id, username, full_name)
    )
    conn.commit()

def save_file_info(user_id: int, file_name: str, file_path: str, transcribed_text: str):
    cursor.execute(
        'INSERT INTO files (user_id, file_name, file_path, transcribed_text) VALUES (?, ?, ?, ?)',
        (user_id, file_name, file_path, transcribed_text)
    )
    conn.commit()

def save_transcription_to_file(transcribed_text: str, file_name: str):
    output_file = OUTPUT_DIR / f"{file_name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcribed_text)
    logger.info(f"Транскрипция сохранена в файл: {output_file}")
    
def convert_to_wav(input_path: str) -> str:
    ext = Path(input_path).suffix.lower()
    if ext == ".wav":
        return input_path
    
    audio = AudioSegment.from_file(input_path)
    wav_path = str(Path(input_path).with_suffix(".wav"))
    audio.export(wav_path, format="wav")
    audio = audio.set_frame_rate(16000)
    return wav_path

def normalize_and_amplify(audio: AudioSegment) -> AudioSegment:
    audio = effects.normalize(audio)
    target_dBFS = -18.0
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)

def analyze_audio(audio: AudioSegment):
    rms = audio.rms
    if rms == 0:
        snr = float('-inf')
    else:
        snr = 20 * np.log10(rms / 1e-6)
    logger.info(f"[Аудио] RMS: {rms}, SNR: {snr:.2f} dB")

def remove_silence(audio: AudioSegment, keep_silence=500):
    dBFS = audio.dBFS
    silence_thresh = dBFS - 14
    return split_on_silence(audio, min_silence_len=700, silence_thresh=silence_thresh, keep_silence=keep_silence)

def split_long_chunk(chunk: AudioSegment, max_length: int = MAX_CHUNK_LENGTH_MS):
    return [chunk[i:i + max_length] for i in range(0, len(chunk), max_length)]

def reduce_noise(audio: AudioSegment) -> AudioSegment:
    try:
        samples = np.array(audio.get_array_of_samples())
        
        if len(samples) == 0:
            return audio
            
        # Более агрессивное шумоподавление
        reduced = nr.reduce_noise(
            y=samples, 
            sr=audio.frame_rate, 
            prop_decrease=0.9,  # более сильное подавление
            time_constant_s=2.0,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )
        
        # Ограничение чтобы не было клиппинга
        reduced = np.clip(reduced, -32768, 32767)
        reduced = reduced.astype(np.int16)
        
        return AudioSegment(
            reduced.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=1
        )
    except Exception as e:
        logger.warning(f"⚠️ Шумоподавление не сработало: {e}")
        return audio

# Удалена неиспользуемая функция process_audio_async
async def process_audio(file_path: str) -> str:
    try:
        wav_file_path = convert_to_wav(file_path)
        audio = AudioSegment.from_file(wav_file_path)
        
        # === УЛУЧШЕННАЯ ПРЕОБРАБОТКА АУДИО ===
        # Конвертация в моно для лучшего качества
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        # Ресемплинг до 16kHz (оптимально для Whisper)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # Нормализация громкости
        audio = effects.normalize(audio)
        
        # Увеличение громкости если слишком тихо
        if audio.dBFS < -20:
            gain = min(10, -20 - audio.dBFS)  # максимум +10dB
            audio = audio.apply_gain(gain)
        
        # Сохраняем обработанное аудио
        processed_path = f"{wav_file_path}_processed.wav"
        audio.export(processed_path, format="wav")
        
        # === УЛУЧШЕННОЕ РАЗДЕЛЕНИЕ НА ЧАНКИ ===
        chunks = split_on_silence(
            audio,
            min_silence_len=700,  # длина тишины для разделения
            silence_thresh=audio.dBFS - 16,  # порог тишины
            keep_silence=300  # оставляем немного тишины для контекста
        )
        
        # Если чанков слишком много, объединяем мелкие
        if len(chunks) > 15:
            combined_chunks = []
            current_chunk = AudioSegment.empty()
            for chunk in chunks:
                if len(current_chunk) + len(chunk) < 45000:  # 45 секунд макс
                    current_chunk += chunk
                else:
                    if len(current_chunk) > 3000:  # минимум 3 секунды
                        combined_chunks.append(current_chunk)
                    current_chunk = chunk
            if len(current_chunk) > 3000:
                combined_chunks.append(current_chunk)
            chunks = combined_chunks

        full_transcription = ""

        for idx, chunk in enumerate(chunks):
            chunk_path = f"{wav_file_path}_part{idx}.wav"
            
            # Дополнительная обработка каждого чанка
            chunk = effects.normalize(chunk)
            
            # Сохраняем чанк
            chunk.export(chunk_path, format="wav")

            try:
                # === ИСПРАВЛЕННЫЕ ПАРАМЕТРЫ WHISPER ===
                # Для FasterWhisper убираем неподдерживаемые параметры
                result = whisper_model.transcribe(
                    chunk_path, 
                    batch_size=8,
                    language='ru'
                    # temperature и condition_on_previous_text не поддерживаются
                )

                # === ВЫРАВНИВАНИЕ ===
                try:
                    align_model, metadata = whisperx.load_align_model(
                        language_code="ru", 
                        device=device
                    )
                    result_aligned = whisperx.align(
                        result["segments"],
                        align_model,
                        metadata,
                        chunk_path,
                        device=device
                    )
                    segments = result_aligned["segments"]
                except Exception as align_error:
                    logger.warning(f"⚠️ Выравнивание не удалось: {align_error}")
                    segments = result["segments"]

                for segment in segments:
                    text = segment["text"].strip()
                    if text and len(text) > 1:  # пропускаем пустые и очень короткие сегменты
                        full_transcription += text + " "

            except Exception as whisper_err:
                logger.error(f"❌ Ошибка распознавания {chunk_path}: {whisper_err}")
                # Fallback: простое распознавание без выравнивания
                try:
                    result = whisper_model.transcribe(chunk_path, language='ru')
                    for segment in result["segments"]:
                        text = segment["text"].strip()
                        if text:
                            full_transcription += text + " "
                except Exception as fallback_err:
                    logger.error(f"❌ Fallback тоже не сработал: {fallback_err}")

            finally:
                # Очистка временных файлов
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

        # Очистка основного обработанного файла
        if os.path.exists(processed_path):
            os.remove(processed_path)

        # === УЛУЧШЕННАЯ ПОСТОБРАБОТКА ТЕКСТА ===
        if not full_transcription.strip():
            logger.warning("⚠️ Транскрипция пустая, пробуем полное аудио...")
            # Fallback: пробуем распознать всё аудио целиком
            try:
                result = whisper_model.transcribe(wav_file_path, language='ru')
                for segment in result["segments"]:
                    text = segment["text"].strip()
                    if text:
                        full_transcription += text + " "
            except Exception as full_err:
                logger.error(f"❌ Полное распознавание тоже не сработало: {full_err}")
                return "❌ Не удалось распознать аудио"

        full_transcription = re.sub(r'\s+', ' ', full_transcription).strip()
        
        # Исправление распространенных ошибок распознавания
        corrections = {
            r'\bпитей\b': 'Петей',
            r'\bпитя\b': 'Петя',
            r'\bдулмой\b': 'Дулмой',
            r'\bдима\b': 'Дулма',
            r'\bмарией\b': 'Мария',
            r'\bмакс\b': 'Максом',
            r'\bанжелика\b': 'Анжеликой',
            r'\bгеншин\b': 'Genshin Impact',
            r'\bдота\b': 'Dota',
            r'\bс руководителм\b': 'с руководителем',
            r'\bс тасом\b': 'со Стасом',
            r'\bсо стасам\b': 'со Стасом',
            r'\bяндекс\b': 'Яндекс',
            r'\bсбером\b': 'Сбером',
            r'\bгазпром\b': 'Газпром',
            r'\bhr\b': 'HR',
            r'\bcrm\b': 'CRM',
            r'\bтз\b': 'ТЗ',
            r'\.(\d{1,2})': r':\1',  # 9.30 -> 9:30
            r'(\d):(\d{2})': r'\1:\2',  # 9:30 -> 9:30
        }

        for wrong, right in corrections.items():
            full_transcription = re.sub(wrong, right, full_transcription, flags=re.IGNORECASE)

        # === ГРАММАТИЧЕСКАЯ КОРРЕКЦИЯ ===
        logger.info("🔧 Исправляю грамматические ошибки...")
        try:
            full_transcription = correct_grammar(full_transcription)
        except Exception as grammar_err:
            logger.warning(f"⚠️ Ошибка грамматической коррекции: {grammar_err}")
            # Продолжаем без грамматической коррекции

        # Улучшение пунктуации
        full_transcription = re.sub(r'(\w)([А-ЯA-Z])', r'\1. \2', full_transcription)
        full_transcription = re.sub(r'\s+\.', '.', full_transcription)
        full_transcription = re.sub(r',\s*,', ',', full_transcription)

        logger.info(f"✅ Транскрипция завершена. Длина: {len(full_transcription)} символов")
        return full_transcription.strip()

    except Exception as e:
        logger.error(f"❌ Ошибка в process_audio: {e}", exc_info=True)
        raise

def convert_google_drive_link(url: str) -> str:
    match = re.search(r'd/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    else:
        return url

# === Отправка длинного текста частями ===
async def send_long_text(message: Message, text: str, chunk_size: int = 4096):
    for i in range(0, len(text), chunk_size):
        await message.answer(text[i:i + chunk_size])

@dp.message(lambda message: message.text and message.text.startswith("http"))
async def handle_audio_link(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    url = convert_google_drive_link(message.text.strip())
    await message.answer("🔗 Получена ссылка. Скачиваю аудиофайл...")

    user_id = message.from_user.id
    save_user_info(user_id, message.from_user.username or "", message.from_user.full_name or "")
    filename = f"{uuid.uuid4().hex}.mp3"
    local_path = UPLOAD_DIR / filename

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    await message.answer("❌ Не удалось скачать файл. Проверьте ссылку.")
                    return
                async with aiofiles.open(local_path, 'wb') as f:
                    await f.write(await resp.read())

        await message.answer("🔊 Аудиофайл скачан. Обрабатываю...")

        transcription = await process_audio(str(local_path))
        save_file_info(user_id, filename, str(local_path), transcription)
        save_transcription_to_file(transcription, filename)
        await message.answer(f"✅ Транскрипция завершена.\nКоличество слов: {len(transcription.split())}")
        await send_long_text(message, f"<b>Транскрипция:</b>\n\n{transcription}")

    except Exception as e:
        logger.error(f"Ошибка обработки аудио: {e}")
        await message.answer("❌ Ошибка при обработке аудиофайла.")

# === Обработка голосовых сообщений ===
@dp.message(F.voice)
async def handle_voice_message(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    try:
        # Скачиваем файл
        voice = message.voice
        file = await bot.get_file(voice.file_id)
        file_path = file.file_path
        file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"

        filename = f"{uuid.uuid4().hex}.ogg"
        local_path = UPLOAD_DIR / filename

        await message.answer("🔊 Голосовое сообщение получено. Скачиваю...")

        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                if resp.status == 200:
                    async with aiofiles.open(local_path, 'wb') as f:
                        await f.write(await resp.read())

        await message.answer("⏳ Обрабатываю аудио...")

        transcription = await process_audio(str(local_path))
        save_file_info(message.from_user.id, filename, str(local_path), transcription)
        save_transcription_to_file(transcription, filename)

        await message.answer(f"✅ Транскрипция завершена.\nКоличество слов: {len(transcription.split())}")

        if len(transcription) <= 4000:
            await message.answer(transcription)
        else:
            await message.answer("📋 Текст длинный, отправляю частями...")
            await send_long_text(message, f"<b>Транскрипция:</b>\n\n{transcription}")

        # Кнопка
        keyboard = ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="📋 Покажи всю транскрипцию")]],
            resize_keyboard=True
        )
        await message.answer("👆 Нажмите, чтобы повторно открыть всю транскрипцию:", reply_markup=keyboard)

    except Exception as e:
        logger.exception("Ошибка обработки голосового")
        await message.answer("❌ Ошибка при обработке голосового сообщения.")

# === Кнопка "Покажи всю транскрипцию" ===
@dp.message(lambda message: message.text == "📋 Покажи всю транскрипцию")
async def handle_show_transcription(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    cursor.execute(
        'SELECT transcribed_text FROM files WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1',
        (message.from_user.id,)
    )
    row = cursor.fetchone()

    # ИСПРАВЛЕНО: Улучшена проверка пустых результатов
    if not row or not row[0] or not row[0].strip():
        await message.answer("📭 Нет доступной транскрипции. Сначала отправьте голосовое сообщение или ссылку на аудиофайл.")
        return

    await message.answer("<b>Вот последняя транскрипция:</b>")
    await send_long_text(message, row[0])

    transcribed_text = row[0]

    try:
        analysis = analyze_transcript(transcribed_text, tz)
    except Exception as e:
        logger.error(f"Ошибка анализа для повторного просмотра транскрипции: {e}", exc_info=True)
        await message.answer("❌ Не удалось проанализировать текст. Попробуйте отправить аудио ещё раз.")
        return

    await message.answer("⏳ Повторный анализ задач...")
    plan_text = format_plan_response(analysis)
    if len(plan_text) > 4000:
        await send_long_text(message, plan_text)
    else:
        await message.answer(plan_text)

    await message.answer("⏳ Повторный анализ встреч сегодня...")
    today_text = format_today_response(analysis)
    if len(today_text) > 4000:
        await send_long_text(message, today_text)
    else:
        await message.answer(today_text)

    await message.answer("⏳ Повторный анализ встреч завтра...")
    tomorrow_text = format_tomorrow_response(analysis)
    if len(tomorrow_text) > 4000:
        await send_long_text(message, tomorrow_text)
    else:
        await message.answer(tomorrow_text)

    await message.answer("⏳ Повторный анализ за прошлую неделю...")
    last_week_text = format_last_week_response(analysis)
    if len(last_week_text) > 4000:
        await send_long_text(message, last_week_text)
    else:
        await message.answer(last_week_text)

@dp.message(lambda message: message.text == "📝 Что мне нужно запланировать?")
async def handle_plan(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    cursor.execute('SELECT transcribed_text FROM files WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (message.from_user.id,))
    row = cursor.fetchone()
    if row is None or not row[0].strip():
        await message.answer("❌ Нет доступных транскрипций для анализа.")
        return

    transcribed_text = row[0]
    await message.answer("⏳ Собираю события и задачи для планирования...")

    try:
        # ✅ Новый надёжный анализ: используем локальный парсер вместо LLM
        analysis = analyze_transcript(transcribed_text, tz)
        response_text = format_plan_response(analysis)
        if len(response_text) > 4000:
            await send_long_text(message, response_text)
        else:
            await message.answer(response_text)

    except Exception as e:
        logger.error(f"Ошибка локального анализа задач: {e}", exc_info=True)
        await message.answer(
            "❌ Не удалось разобрать текст. Попробуйте уточнить формулировки или отправьте запись ещё раз."
        )

@dp.message(lambda message: message.text == "👥 С кем я сегодня встречался?")
async def handle_meetings_today(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    cursor.execute('SELECT transcribed_text FROM files WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (message.from_user.id,))
    row = cursor.fetchone()
    if row is None or not row[0].strip():
        await message.answer("❌ Нет доступных транскрипций для анализа.")
        return

    transcribed_text = row[0]
    await message.answer("⏳ Ищу встречи за сегодняшний день...")

    try:
        analysis = analyze_transcript(transcribed_text, tz)
        response_text = format_today_response(analysis)
        if len(response_text) > 4000:
            await send_long_text(message, response_text)
        else:
            await message.answer(response_text)
    except Exception as e:
        logger.error(f"Ошибка локального анализа встреч сегодня: {e}", exc_info=True)
        await message.answer(
            "❌ Не удалось выделить встречи за сегодня. Попробуйте указать дату или время в тексте."
        )

@dp.message(lambda message: message.text == "📅 Какой план встреч у меня может быть завтра?")
async def handle_meetings_tomorrow(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    cursor.execute('SELECT transcribed_text FROM files WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (message.from_user.id,))
    row = cursor.fetchone()
    if not row or not row[0].strip():
        await message.answer("📭 Нет текста для анализа. Сначала отправьте аудио.")
        return

    transcribed_text = row[0]
    await message.answer("⏳ Собираю встречи, запланированные на завтра...")

    try:
        analysis = analyze_transcript(transcribed_text, tz)
        response_text = format_tomorrow_response(analysis)
        if len(response_text) > 4000:
            await send_long_text(message, response_text)
        else:
            await message.answer(response_text)
    except Exception as e:
        logger.error(f"Ошибка локального анализа встреч завтра: {e}", exc_info=True)
        await message.answer(
            "❌ Не удалось определить планы на завтра. Проверьте, что в тексте есть слова «завтра» или конкретная дата."
        )

@dp.message(lambda message: message.text == "🕒 Какие встречи были на прошлой неделе?")
async def handle_meetings_last_week(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    cursor.execute('SELECT transcribed_text FROM files WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (message.from_user.id,))
    row = cursor.fetchone()
    if not row or not row[0].strip():
        await message.answer("❌ Нет текста для анализа.")
        return

    transcribed_text = row[0]
    await message.answer("⏳ Ищу события за прошлую неделю...")

    try:
        analysis = analyze_transcript(transcribed_text, tz)
        response_text = format_last_week_response(analysis)
        if len(response_text) > 4000:
            await send_long_text(message, response_text)
        else:
            await message.answer(response_text)

    except Exception as e:
        logger.error(f"Ошибка локального анализа прошлой недели: {e}", exc_info=True)
        await message.answer(
            "❌ Не удалось собрать встречи за прошлую неделю. Убедитесь, что в тексте указаны даты предыдущей недели."
        )

@dp.message(lambda message: message.text == "🔁 Повторить весь анализ")
async def handle_repeat_full_analysis(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    cursor.execute('SELECT transcribed_text FROM files WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (message.from_user.id,))
    row = cursor.fetchone()
    if not row or not row[0].strip():
        await message.answer("❌ Нет доступной транскрипции.")
        return

    transcribed_text = row[0]

    try:
        analysis = analyze_transcript(transcribed_text, tz)
    except Exception as e:
        logger.error(f"Ошибка повторного анализа текста: {e}", exc_info=True)
        await message.answer("❌ Не удалось повторить анализ. Попробуйте позже или обновите запись.")
        return

    await message.answer("⏳ Повторный анализ задач...")
    plan_text = format_plan_response(analysis)
    if len(plan_text) > 4000:
        await send_long_text(message, plan_text)
    else:
        await message.answer(plan_text)

    await message.answer("⏳ Повторный анализ встреч сегодня...")
    today_text = format_today_response(analysis)
    if len(today_text) > 4000:
        await send_long_text(message, today_text)
    else:
        await message.answer(today_text)

    await message.answer("⏳ Повторный анализ встреч завтра...")
    tomorrow_text = format_tomorrow_response(analysis)
    if len(tomorrow_text) > 4000:
        await send_long_text(message, tomorrow_text)
    else:
        await message.answer(tomorrow_text)

    await message.answer("⏳ Повторный анализ за прошлую неделю...")
    last_week_text = format_last_week_response(analysis)
    if len(last_week_text) > 4000:
        await send_long_text(message, last_week_text)
    else:
        await message.answer(last_week_text)

@dp.message(lambda message: message.text == "📌 Добавь в календарь")
async def handle_add_to_calendar(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    cursor.execute('SELECT transcribed_text FROM files WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (message.from_user.id,))
    row = cursor.fetchone()
    if not row or not row[0].strip():
        await message.answer("❌ Нет текста для анализа.")
        return

    transcribed_text = row[0]
    await message.answer("📆 Анализирую задачи для добавления в календарь...")

    try:
        tasks = query_llm(transcribed_text, mode="todo")
        # ИСПРАВЛЕНО: Улучшена проверка пустых результатов
        if not tasks or not tasks.strip() or any(phrase in tasks.lower() for phrase in ["нет задач", "нет событий", "событий не найдено", "задач не найдено", "планирования"]):
            await message.answer("📭 Нет задач для добавления в календарь.")
            return

        now = datetime.now(tz)
        confirmations = []
        task_count = 0
        
        for i, line in enumerate(tasks.splitlines()):
            task = line.strip("-• ").strip()
            if not task or len(task) < 3:  # Пропускаем слишком короткие задачи
                continue
            start_time = now + timedelta(hours=i+1)
            try:
                url = add_event_to_calendar(task, start_time)
                confirmations.append(f"✅ <b>{task}</b> — <a href='{url}'>в календаре</a>")
                task_count += 1
            except Exception as calendar_err:
                logger.error(f"Ошибка добавления задачи в календарь: {calendar_err}")
                confirmations.append(f"❌ <b>{task}</b> — ошибка добавления")

        if task_count == 0:
            await message.answer("📭 Не удалось добавить ни одной задачи в календарь.")
        else:
            await message.answer("\n".join(confirmations), parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Ошибка при добавлении задач в календарь: {e}")
        await message.answer("❌ Ошибка при добавлении задач.")

# === Команда /start ===
@dp.message(F.text == "/start")
async def cmd_start(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    # Создаём основную клавиатуру с кнопками
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📝 Что мне нужно запланировать?")],
            [KeyboardButton(text="👥 С кем я сегодня встречался?")],
            [KeyboardButton(text="📅 Какой план встреч у меня может быть завтра?")],
            [KeyboardButton(text="🕒 Какие встречи были на прошлой неделе?")],
            [KeyboardButton(text="📌 Добавь в календарь")],
            [KeyboardButton(text="📋 Покажи всю транскрипцию")],
            [KeyboardButton(text="🔁 Повторить весь анализ")],
            [KeyboardButton(text="🛠 Админ-панель")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите действие",
    )

    # Отправляем приветственное сообщение с основной клавиатурой
    await message.answer(
        f"👋 Привет, {message.from_user.full_name}!\n"
        f"Я помогу тебе с анализом встреч и задач по аудио.",
        reply_markup=keyboard
    )

# === Команда /help ===
@dp.message(F.text == "/help")
async def cmd_help(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    help_text = (
        "🤖 **Я — ваш голосовой помощник!**\n\n"
        "**Как я работаю:**\n"
        "1. Отправьте мне голосовое сообщение или ссылку на аудиофайл.\n"
        "2. Я расшифрую его и предложу варианты анализа.\n\n"
        "**Что я умею:**\n"
        "• **📝 Что мне нужно запланировать?** — Нахожу все задачи из аудиозаписи.\n"
        "• **👥 С кем я сегодня встречался?** — Вывожу встречи на сегодня.\n"
        "• **📅 Какой план встреч у меня может быть завтра?** — Показываю встречи на завтра.\n"
        "• **🕒 Какие встречи были на прошлой неделе?** — Анализирую встречи за прошедшую неделю.\n"
        "• **📌 Добавь в календарь** — Создаю события в Google Календаре (требуется настройка).\n"
        "• **📋 Покажи всю транскрипцию** — Отправляю полный текст последней расшифровки.\n"
        "• **🔁 Повторить весь анализ** — Повторяю анализ по всем пунктам.\n"
    )
    await message.answer(help_text, parse_mode=ParseMode.MARKDOWN)

# === Команда /support ===
@dp.message(F.text == "/support")
async def cmd_support(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    support_text = (
    "🆘 **Служба Разработки Бота**\n\n"
    "Возникли вопросы или проблемы? Пишите прямо мне, разработчику! 😄\n\n"
    "📱 **WhatsApp:** +994 55 434 50 06\n"
    "📱 **MAX:** +7 996 687 24 20\n"
    "✉️ **Email:** djigit1220@gmail.com\n"
    "💬 **Telegram:** @Rafael005t"
    )
    await message.answer(support_text, parse_mode=ParseMode.MARKDOWN)

@dp.message(F.text.in_({"/admin", "🛠 Админ-панель"}))
async def admin_panel(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("⛔️ У вас нет доступа к админ-панели.")
        return

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Добавить пользователя", callback_data="admin_add")],
        [InlineKeyboardButton(text="➖ Удалить пользователя", callback_data="admin_remove")],
        [InlineKeyboardButton(text="📋 Список пользователей", callback_data="admin_list")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="🗑 Очистить кэш файлов", callback_data="admin_clear_cache")],
        [InlineKeyboardButton(text="📃 Посмотреть bot.log", callback_data="admin_view_log_bot")],
        [InlineKeyboardButton(text="❌ Посмотреть error.log", callback_data="admin_view_log_error")],
        [InlineKeyboardButton(text="♻ Перезагрузить бота", callback_data="admin_restart")],
        [InlineKeyboardButton(text="✏ Редактировать промпт", callback_data="admin_edit_prompt")],
    ])
    await message.answer("🛠 <b>Админ-панель:</b>", reply_markup=keyboard, parse_mode="HTML")

# --- ОБРАБОТЧИКИ ДЛЯ АДМИН-ПАНЕЛИ ---

@dp.callback_query(F.data == "admin_stats")
async def callback_stats(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return

    try:
        cursor.execute("SELECT COUNT(DISTINCT id) FROM users")
        total_users = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM files")
        total_files = cursor.fetchone()[0]

        stats_message = (
            "📊 **Статистика использования:**\n"
            f"👤 Всего пользователей: `{total_users}`\n"
            f"🔊 Всего обработано файлов: `{total_files}`"
        )
        await callback.message.answer(stats_message, parse_mode=ParseMode.MARKDOWN)
        await callback.answer()
    except Exception as e:
        logger.exception("Ошибка при получении статистики")
        await callback.message.answer("❌ Ошибка при получении статистики.")
        await callback.answer()

@dp.callback_query(F.data == "admin_clear_cache")
async def callback_clear_cache(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return

    try:
        files_deleted = 0
        # Удаляем файлы из папки UPLOAD
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                files_deleted += 1
        # Удаляем файлы из папки OUTPUT
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                files_deleted += 1

        await callback.message.answer(f"🗑️ Кэш очищен. Удалено файлов: {files_deleted}")
        await callback.answer()
    except Exception as e:
        logger.exception("Ошибка при очистке кэша")
        await callback.message.answer("❌ Произошла ошибка при очистке кэша.")
        await callback.answer()
async def send_restart_notification():
    """Отправляет уведомление о перезапуске бота"""
    try:
        restart_chat_id = config.get("restart_message_to")
        if restart_chat_id:
            await bot.send_message(
                restart_chat_id,
                "🔄 Бот успешно перезапущен и готов к работе!\n"
                "✅ Все функции активированы\n"
                "📅 Дата корректно определяется: " + datetime.now().strftime("%d.%m.%Y")
            )
            # Очищаем ID после отправки
            if "restart_message_to" in config:
                config["restart_message_to"] = None
                with open("config.json", "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Ошибка отправки уведомления о перезапуске: {e}")
async def send_last_log_lines(callback: CallbackQuery, filename: str, lines_count: int = 50):
    try:
        if not os.path.exists(filename):
            await callback.message.answer(f"⚠️ Файл `{filename}` не найден.", parse_mode=ParseMode.MARKDOWN)
            return

        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            await callback.message.answer(f"📃 Файл `{filename}` пуст.", parse_mode=ParseMode.MARKDOWN)
            return

        last_lines = "".join(lines[-lines_count:])
        await callback.message.answer(f"Последние {len(lines[-lines_count:])} строк из `{filename}`:\n"
                                      f"```\n{last_lines}```", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception(f"Ошибка при чтении файла логов {filename}")
        await callback.message.answer(f"❌ Ошибка при чтении файла `{filename}`.", parse_mode=ParseMode.MARKDOWN)

@dp.callback_query(F.data == "admin_view_log_bot")
async def callback_view_log_bot(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return
    await send_last_log_lines(callback, 'bot.log')
    await callback.answer()

@dp.callback_query(F.data == "admin_view_log_error")
async def callback_view_log_error(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return
    await send_last_log_lines(callback, 'error.log')
    await callback.answer()

# [УДАЛЕНО] 📌 ИСПРАВЛЕНО: Удалена устаревшая, дублирующая функция get_prompts_with_dates.
# Теперь используется импорт из llm_client.py

@dp.callback_query(F.data == "admin_edit_prompt")
async def admin_edit_prompt_menu(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return

    # 📌 ИСПРАВЛЕНО: Получаем промпты из llm_client, а не из удаленной функции
    # Передаем пустой транскрипт, т.к. он нужен только для определения года в 'todo'
    prompts = get_llm_prompts(transcript="", mode="all")
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=key, callback_data=f"edit_prompt_{key}")]
            for key in prompts.keys()
        ]
    )
    await callback.message.answer("📋 Выберите промпт для редактирования:", reply_markup=kb)
    await callback.answer()

@dp.callback_query(F.data.startswith("edit_prompt_"))
async def admin_choose_prompt(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return

    mode = callback.data.replace("edit_prompt_", "")
    
    # 📌 ИСПРАВЛЕНО: Получаем промпты из llm_client
    prompts = get_llm_prompts(transcript="", mode="all")
    if mode not in prompts:
        await callback.message.answer("❌ Такого промпта нет.")
        return

    admin_state[callback.from_user.id] = {"mode": "edit_prompt", "prompt_key": mode}
    await callback.message.answer(
        f"✍️ Введите новый текст для промпта <b>{mode}</b>:\n\n"
        f"📄 Текущий текст начинается так:\n<code>{prompts[mode][:300]}...</code>",
        parse_mode="HTML"
    )
    await callback.answer()

@dp.message(F.text)
async def handle_admin_input(message: Message):
    user_id = message.from_user.id
    if user_id not in admin_state:
        # Это не ответ на запрос админ-панели, обрабатываем как обычное сообщение
        if is_allowed(user_id):
            await message.answer(
                "Извините, я не понял вашу команду. "
                "Пожалуйста, используйте кнопки на клавиатуре или одну из команд: /start, /help, /support."
            )
        else:
            await message.answer("⛔️ У вас нет доступа к этому боту.")
        return

    state = admin_state[user_id]
    action = state["mode"]

    if action == "edit_prompt":
        prompt_key = state["prompt_key"]
        new_prompt = message.text.strip()

        # Читаем llm_client.py
        try:
            with open("llm_client.py", "r", encoding="utf-8") as f:
                code = f.read()
        except FileNotFoundError:
            await message.answer("❌ КРИТИЧЕСКАЯ ОШИБКА: Файл llm_client.py не найден.")
            del admin_state[user_id]
            return
        except Exception as e:
            await message.answer(f"❌ Ошибка чтения llm_client.py: {e}")
            del admin_state[user_id]
            return

        # Ищем многострочный текст для ключа
        # 📌 ИСПРАВЛЕНО: Улучшен паттерн для поиска f-строк
        pattern = rf'"{prompt_key}"\s*:\s*f?"""[\s\S]*?"""'
        # Заменяем, сохраняя f-строку (f""")
        replacement = f'"{prompt_key}": f"""{new_prompt}"""'

        updated_code, count = re.subn(pattern, replacement, code, count=1)
        if count == 0:
            # Попробуем найти без f-строки (хотя они все f-строки)
            pattern = rf'"{prompt_key}"\s*:\s*"""[\s\S]*?"""'
            replacement = f'"{prompt_key}": f"""{new_prompt}"""' # Всегда делаем f-строкой для поддержки дат
            updated_code, count = re.subn(pattern, replacement, code, count=1)

        if count == 0:
            await message.answer("❌ Не удалось найти промпт в llm_client.py. Проверьте паттерн поиска.")
            del admin_state[user_id]
            return

        # Записываем изменения в файл
        try:
            with open("llm_client.py", "w", encoding="utf-8") as f:
                f.write(updated_code)
        except Exception as e:
            await message.answer(f"❌ Ошибка записи в llm_client.py: {e}")
            del admin_state[user_id]
            return

        await message.answer(f"✅ Промпт <b>{prompt_key}</b> обновлён в llm_client.py", parse_mode="HTML")
        del admin_state[user_id]

        # Перезапуск бота
        await message.answer("♻ Перезапускаю бота для применения изменений...")
        
        # Сохраняем ID для уведомления о перезапуске
        config["restart_message_to"] = message.from_user.id
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Не удалось сохранить restart_message_to: {e}")
            
        os.execl(sys.executable, sys.executable, *sys.argv)
        return

    try:
        target_id = int(message.text.strip())
    except ValueError:
        await message.answer("❗ Введите числовой ID.")
        return

    if action == "add":
        if target_id in ALLOWED_USERS:
            await message.answer("✅ Пользователь уже есть.")
        else:
            ALLOWED_USERS.add(target_id)
            config["allowed_users"] = list(ALLOWED_USERS)
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            await message.answer(f"➕ Пользователь <code>{target_id}</code> добавлен.", parse_mode="HTML")

    elif action == "remove":
        if target_id not in ALLOWED_USERS:
            await message.answer("❌ Такого пользователя нет.")
        else:
            ALLOWED_USERS.remove(target_id)
            config["allowed_users"] = list(ALLOWED_USERS)
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            await message.answer(f"➖ Пользователь <code>{target_id}</code> удалён.", parse_mode="HTML")

    del admin_state[user_id]

@dp.callback_query(F.data == "admin_restart")
async def callback_restart_bot(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return

    # Сохраняем ID администратора для отправки сообщения после перезапуска
    config["restart_message_to"] = callback.from_user.id
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    await callback.message.answer("♻ Перезапускаю бота...")
    await callback.answer()
    
    # Грамотный перезапуск
    os.execl(sys.executable, sys.executable, *sys.argv)

@dp.callback_query(F.data == "admin_list")
async def callback_list_users(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return
    if not ALLOWED_USERS:
        await callback.message.answer("📭 Список пуст.")
        return
    users_text = "\n".join(f"• <code>{uid}</code>" for uid in sorted(ALLOWED_USERS))
    await callback.message.answer(f"📋 Список пользователей:\n{users_text}", parse_mode="HTML")
    await callback.answer()

@dp.callback_query(F.data == "admin_add")
async def callback_add_user(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return
    admin_state[callback.from_user.id] = {"mode": "add"}
    await callback.message.answer("✍️ Введите ID пользователя, которого хотите <b>добавить</b>:", parse_mode="HTML")
    await callback.answer()

@dp.callback_query(F.data == "admin_remove")
async def callback_remove_user(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔️ Нет доступа", show_alert=True)
        return
    admin_state[callback.from_user.id] = {"mode": "remove"}
    await callback.message.answer("✍️ Введите ID пользователя, которого хотите <b>удалить</b>:", parse_mode="HTML")
    await callback.answer()

if __name__ == "__main__":
    logging.info("🚀 Бот запускается...")
    set_model(MODEL_NAME)
    
    # Создаем задачу для отправки уведомления о перезапуске
    async def main():
        # Запускаем бота
        bot_task = asyncio.create_task(dp.start_polling(bot))
        
        # Ждем немного и отправляем уведомление о перезапуске
        await asyncio.sleep(2)
        await send_restart_notification()
        
        # Ждем завершения работы бота
        await bot_task
    
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"❌ Ошибка запуска бота: {e}")