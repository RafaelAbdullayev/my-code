import requests
import re
from datetime import datetime, timedelta
import logging
import json
import os
import pytz  # 📌 ИСПРАВЛЕНО: Добавлен импорт для работы с часовыми поясами
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# --- Конфигурация ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load configuration from config.json
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    OLLAMA_URL = config_data.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    MODEL_NAME = config_data.get("MODEL_NAME", "llama3")
except FileNotFoundError:
    logger.warning("config.json not found, using environment variables or defaults")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
except json.JSONDecodeError as e:
    logger.error(f"Error parsing config.json: {e}, using defaults")
    OLLAMA_URL = "http://localhost:11434/api/chat"
    MODEL_NAME = "llama3"

LLM_PRESETS = {
    "точно": {"temperature": 0.0, "top_p": 0.9, "repeat_penalty": 1.2, "top_k": 20, "num_predict": 3072},
}
CURRENT_PRESET = "точно"


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    ollama_url: str
    model_name: str
    preset: str = CURRENT_PRESET
    timeout: int = 120
    max_retries: int = 2


# --- Вспомогательные функции (исправленные) ---

def validate_input(text: str) -> Tuple[bool, str]:
    """Валидация входного транскрипта"""
    if not text or not text.strip():
        logger.warning("Валидация: получен пустой текст")
        return False, "Текст не может быть пустым"
    
    text_length = len(text.strip())
    
    if text_length < 10:
        logger.warning(f"Валидация: текст слишком короткий ({text_length} символов)")
        return False, "Текст слишком короткий для анализа (минимум 10 символов)"
    
    if text_length > 50000:
        logger.warning(f"Валидация: текст слишком длинный ({text_length} символов)")
        return False, "Текст слишком длинный (максимум 50000 символов)"
    
    logger.debug(f"Валидация пройдена: текст содержит {text_length} символов")
    return True, "OK"


def preprocess_input(text: str) -> str:
    """Предобработка текста для улучшения распознавания времени и дат"""
    if not text or not text.strip():
        return ""
    
    # Предварительная замена сокращений с точками
    text = re.sub(r'\bK\.?\s*P\.?\s*I\.?\b', 'KPI', text, flags=re.IGNORECASE)
    text = re.sub(r'\bC\.?\s*R\.?\s*M\.?\b', 'CRM', text, flags=re.IGNORECASE)
    text = re.sub(r'\bH\.?\s*R\.?\b', 'HR', text, flags=re.IGNORECASE)
    text = re.sub(r'\bA\.?\s*B\.?\s*тест', 'AB тест', text, flags=re.IGNORECASE)
    text = re.sub(r'\bQ\.?\s*(\d+)', r'Q\1', text, flags=re.IGNORECASE)
    
    # Нормализация времени "X часов ровно" -> "X:00"
    text = re.sub(r"\b(\d{1,2})\s+час(а|ов)?\s+ровно\b", lambda m: f"{int(m.group(1)):02d}:00", text, flags=re.IGNORECASE)
    
    # "X часов Y минут" -> "X:Y"
    text = re.sub(r"\b(\d{1,2})\s+час(а|ов)?\s+(\d{1,2})\s+минут[ыа]?\b", lambda m: f"{int(m.group(1)):02d}:{int(m.group(3)):02d}", text, flags=re.IGNORECASE)
    
    # Старые правила (оставляем для совместимости)
    text = re.sub(r"\b[Вв]\s+(\d{1,2})\s+[Чч]ас(а|ов)?\b", lambda m: f"в {int(m.group(1)):02d}:00", text)
    
    # "в 9" -> "в 9:00" (исключаем случаи типа "в 9 сентября")
    months_list = ["января", "февраля", "марта", "апреля", "мая", "июня", 
                   "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    month_pattern = "|".join(months_list)
    text = re.sub(rf"\b[Вв]\s+(\d{{1,2}})(?![:\d\.]|\s+(?:{month_pattern}))", r"в \1:00", text)
    
    # "15.30" -> "15:30" ТОЛЬКО для времени (НЕ для дат!)
    text = re.sub(r"\b(\d{1,2})\.(\d{2})\b(?!\.\d{4})", r"\1:\2", text)
    
    # Очистка лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_llm_response(text: str) -> str:
    """Очистка ответа LLM"""
    if not text:
        return ""
    
    text = text.strip()
    # Удаление HTML тегов
    text = re.sub(r'</?\w+>', '', text)
    
    # Расширенная замена англицизмов
    replacements = {
        'CRM': 'система управления клиентами',
        'HR': 'отдел кадров',
        'KPI': 'ключевые показатели эффективности',
        'CEO': 'генеральный директор',
        'IT': 'информационные технологии',
        'C.R.M': 'система управления клиентами',
        'K.P.I': 'ключевые показатели эффективности',
        'H.R': 'отдел кадров',
        'A.B': 'AB',
        'Digital Trend': 'цифровые тренды',
        'Zoom': 'видеозвонок',
        'фидбэк': 'отзыв'
    }
    
    for pattern, replacement in replacements.items():
        pattern_variations = [
            pattern,
            pattern.replace(' ', ''),
            pattern.replace('.', ''),
            pattern.replace('. ', '.'),
            pattern.replace(' .', '.')
        ]
        
        for variant in pattern_variations:
            text = re.sub(r'\b' + re.escape(variant) + r'\b', replacement, text, flags=re.IGNORECASE)
    
    # Исправление структурных ошибок
    text = re.sub(r'\bВ задач нет\.?\b', 'Задачи:\n(Нет задач)', text, flags=re.IGNORECASE)
    text = re.sub(r'\bЗадачи нет\.?\b', 'Задачи:\n(Нет задач)', text, flags=re.IGNORECASE)
    
    # Убираем дубликаты событий
    lines = text.split('\n')
    processed_lines = []
    seen_events = set()
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append(line)
            continue
            
        # 📌 ИСПРАВЛЕНО: Проверяем строки, начинающиеся с длинного тире "—", как указано в промпте.
        # Если это элемент списка (встреча или задача)
        if line.startswith('— '):
            # Проверяем полную строку включая время
            if line.lower() not in seen_events:
                seen_events.add(line.lower())
                processed_lines.append(line)
        else:
            # Заменяем тире только в обычных предложениях
            line = re.sub(r'\s-\s', ' — ', line)
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)




def extract_participants(text: str):
    """
    Extract participant names from Russian text.
    Identifies Russian names in contexts like встреча с X, звонок с Y.
    
    Args:
        text: The transcript text
        
    Returns:
        List of identified participant names
    """
    import re
    participants = set()
    
    patterns = [
        r'(?:встреча|звонок|обед|совещание|разговор|переговоры|обсуждение|созвон)\s+(?:с|со)\s+([А-ЯЁ][а-яё]+(?:ой|ом|ым|ей|ем)?)',
        r'(?:от|для|у)\s+([А-ЯЁ][а-яё]+(?:ы|и|а)?)',
        r'([А-ЯЁ][а-яё]+(?:ой|ом|ым|ей|ем)?)\s+([А-ЯЁ][а-яё]+(?:ой|ом|ым|ей|ем|ичем|овной)?)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                name = ' '.join(match)
            else:
                name = match
            name = re.sub(r'(ом|ой|ым|ей|ем|ичем|овной)$', '', name, flags=re.IGNORECASE)
            if len(name) >= 3:
                participants.add(name.strip())
    
    logger.debug(f'Extracted participants: {participants}')
    return sorted(list(participants))

def extract_year_from_transcript(transcript: str) -> int:
    """
    Извлекает год из транскрипта или использует текущий год по умолчанию.
    """
    current_year = datetime.now().year
    
    # Поиск явного указания года в разных форматах
    year_patterns = [
        r'\b(20\d{2})\b',  # 2025
        r'\b(\d{2})\.(\d{2})\.(20\d{2})\b',  # 13.09.2025
        r'\b(\d{1,2})\s+[а-я]+\s+(20\d{2})',  # 13 сентября 2025
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, transcript)
        for match in matches:
            if isinstance(match, tuple):
                year_str = match[-1]  # берем последнюю группу (год)
            else:
                year_str = match
                
            try:
                year = int(year_str)
                if 2020 <= year <= 2030:  # разумные пределы
                    logger.info(f"Найден год в транскрипте: {year}")
                    return year
            except ValueError:
                continue
    
    logger.info(f"Год не найден в транскрипте, используем текущий: {current_year}")
    return current_year
    
    # Поиск в датах типа "13.09.2025"
    date_match = re.search(r'\b(\d{2})\.(\d{2})\.(20\d{2})\b', transcript)
    if date_match:
        try:
            year = int(date_match.group(3))
            if not (current_year - 10 <= year <= current_year + 10):
                logger.warning(f"Year {year} is out of reasonable range, using current year {current_year}")
                return current_year
            return year
        except ValueError:
            logger.warning("Invalid year format in date, using current year")
            return current_year
    
    # Если год не найден, используем текущий
    return current_year


def parse_date_from_text(date_text: str, transcript_year: int) -> Optional[datetime]:
    """
    Парсит дату из текста в формате "Понедельник, 13 сентября"
    с использованием года из транскрипта.
    """
    months = {
        "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
        "мая": 5, "июня": 6, "июля": 7, "августа": 8,
        "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
    }
    
    # Паттерн для "Понедельник, 13 сентября"
    pattern = r"(?P<weekday>Понедельник|Вторник|Среда|Четверг|Пятница|Суббота|Воскресенье),\s*(?P<day>\d{1,2})\s+(?P<month>[а-яА-Я]+)"
    
    match = re.search(pattern, date_text, re.IGNORECASE)
    if match:
        try:
            day = int(match.group("day"))
        except ValueError:
            logger.warning(f"Invalid day in date_text: {match.group('day')}")
            return None
        month_name = match.group("month").lower()
        month = months.get(month_name)
        
        if month:
            try:
                return datetime(transcript_year, month, day)
            except ValueError:
                logger.warning(f"Некорректная дата: {day} {month_name} {transcript_year}")
    
    return None


def normalize_dates(text: str, year: int) -> str:
    """
    Нормализует даты в тексте к формату "ДД.ММ.ГГГГ, день_недели"
    """
    current_year = datetime.now().year
    if not (current_year - 1 <= year <= current_year + 2):
        logger.warning(f"Год {year} выходит за разумные пределы, используем {current_year}")
        year = current_year
    weekdays_variants = ["понедельник", "вторник", "среда", "среду", "четверг", "пятница", "пятницу", "суббота", "субботу", "воскресенье"]
    normalize_weekday = {
        "понедельник": "понедельник",
        "вторник": "вторник",
        "среда": "среда",
        "среду": "среда",
        "четверг": "четверг",
        "пятница": "пятница",
        "пятницу": "пятница",
        "суббота": "суббота",
        "субботу": "суббота",
        "воскресенье": "воскресенье",
    }
    weekdays_list = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    months_map = {
        "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
        "мая": "05", "июня": "06", "июля": "07", "августа": "08",
        "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"
    }
    
    weekdays_pattern = "|".join(weekdays_variants)
    months_pattern = "|".join(months_map.keys())

    date_pattern = re.compile(
        r"(?:(?P<w1>" + weekdays_pattern + r"),\s*(?P<d1>\d{1,2})\s+(?P<m1>" + months_pattern + r"))" +
        r"|(?:(?P<d2>\d{1,2})\s+(?P<m2>" + months_pattern + r")\s*(?P<w2>" + weekdays_pattern + r")?)" +
        r"|\b(?P<w3>" + weekdays_pattern + r")\b",
        re.IGNORECASE
    )

    new_text = text
    offset = 0
    for match in date_pattern.finditer(text):
        if match.group("w3"):
            weekday = normalize_weekday.get(match.group("w3").lower())
            if weekday:
                standardized = weekday.capitalize()  # Keep as is, since no day
                # For standalone, we don't replace with full date here, handle in split
                continue  # Skip replacement for standalone in normalize, handle in split
        else:
            day_str = match.group("d1") or match.group("d2")
            if day_str:
                day = int(day_str)
            else:
                continue
            month_name = (match.group("m1") or match.group("m2") or "").lower()
            weekday = normalize_weekday.get((match.group("w1") or match.group("w2") or "").lower(), "")
            month_num = months_map.get(month_name, "")
            if month_num and day:
                if not weekday:
                    try:
                        date_obj = datetime(year, int(month_num), day)
                        weekday = weekdays_list[date_obj.weekday()]
                    except ValueError:
                        continue
                standardized = f"{day:02d}.{month_num}.{year}, {weekday}"
                start = match.start() + offset
                end = match.end() + offset
                new_text = new_text[:start] + standardized + new_text[end:]
                offset += len(standardized) - (end - start)

    return new_text


def split_by_dates(transcript: str) -> Dict[str, str]:
    """
    📌 ИСПРАВЛЕНО И ОПТИМИЗИРОВАНО:
    Делит транскрипт на блоки по датам с автоматическим определением года.
    Теперь поддерживает форматы "Понедельник, 13 сентября" и "13 октября понедельник".
    Возвращает словарь: { "13.09.2025, понедельник": "текст событий...", ... }
    """
    transcript_year = extract_year_from_transcript(transcript)
    logger.info(f"Определен год из транскрипта: {transcript_year}")

    weekdays_variants = ["понедельник", "вторник", "среда", "среду", "четверг", "пятница", "пятницу", "суббота", "субботу", "воскресенье"]
    normalize_weekday = {
        "понедельник": "понедельник",
        "вторник": "вторник",
        "среда": "среда",
        "среду": "среда",
        "четверг": "четверг",
        "пятница": "пятница",
        "пятницу": "пятница",
        "суббота": "суббота",
        "субботу": "суббота",
        "воскресенье": "воскресенье",
    }
    weekdays_list = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    months_map = {
        "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
        "мая": "05", "июня": "06", "июля": "07", "августа": "08",
        "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"
    }
    
    # Паттерн для дней недели и месяцев для использования в regex
    weekdays_pattern = "|".join(weekdays_variants)
    months_pattern = "|".join(months_map.keys())

    # Гибкий паттерн, который находит оба формата + standalone weekday
    date_pattern = re.compile(
        r"(?:(?P<w1>" + weekdays_pattern + r"),\s*(?P<d1>\d{1,2})\s+(?P<m1>" + months_pattern + r"))" +
        r"|(?:(?P<d2>\d{1,2})\s+(?P<m2>" + months_pattern + r")\s*(?P<w2>" + weekdays_pattern + r")?)" +
        r"|\b(?P<w3>" + weekdays_pattern + r")\b",
        re.IGNORECASE
    )

    blocks = {}
    matches = list(date_pattern.finditer(transcript))
    
    # Обработка случая без совпадений или текста перед первым совпадением
    if not matches:
        logger.warning("В транскрипте не найдено дат в поддерживаемом формате.")
        if transcript.strip():
            blocks["Неопределенная дата"] = transcript.strip()
    else:
        # Текст перед первой датой как неопределенная
        pre_start = 0
        first_match = matches[0]
        pre_text = transcript[pre_start:first_match.start()].strip()
        if pre_text:
            blocks["Неопределенная дата"] = pre_text
        
        last_date = None
        for i, match in enumerate(matches):
            date_obj = None
            if match.group("w3"):
                weekday = normalize_weekday.get(match.group("w3").lower())
                if weekday and last_date:
                    target_weekday_idx = weekdays_list.index(weekday)
                    days_delta = (target_weekday_idx - last_date.weekday()) % 7
                    if days_delta == 0:
                        days_delta = 7  # next week if same day
                    date_obj = last_date + timedelta(days=days_delta)
                    day = date_obj.day
                    month_num = f"{date_obj.month:02d}"
                    year = date_obj.year
                    date_key = f"{day:02d}.{month_num}.{year}, {weekday}"
            else:
                day_str = match.group("d1") or match.group("d2")
                if day_str:
                    day = int(day_str)
                else:
                    continue
                month_name = (match.group("m1") or match.group("m2")).lower()
                weekday = normalize_weekday.get((match.group("w1") or match.group("w2") or "").lower(), "")
                month_num = months_map.get(month_name, "")
                if month_num and day:
                    try:
                        date_obj = datetime(transcript_year, int(month_num), day)
                        if not weekday:
                            weekday = weekdays_list[date_obj.weekday()]
                        date_key = f"{day:02d}.{month_num}.{transcript_year}, {weekday}"
                    except ValueError:
                        logger.warning(f"Не удалось создать дату для {day}.{month_num}.{transcript_year}")
                        continue

            if date_obj:
                last_date = date_obj

            if 'date_key' not in locals():
                continue

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(transcript)
            content = transcript[start:end].strip()
            
            # Убираем точку в начале контента, если она есть
            # Comment: Extended to handle leading comma as well for cases like ", 8:00..."
            if content and content[0] in '.,':
                content = content[1:].strip()

            blocks[date_key] = content

    return blocks


def get_date_context() -> Dict[str, Any]:
    """
    📌 ИСПРАВЛЕНО: Определяет КОНТЕКСТ ТЕКУЩЕЙ ДАТЫ (СЕГОДНЯ) для промптов.
    Использует Московское время, как в bot.py.
    Возвращает словарь с датами для использования в промптах.
    """
    
    # 📌 ИСПРАВЛЕНО: Используем актуальную текущую дату, а не дату из транскрипта
    tz = pytz.timezone("Europe/Moscow")
    target_date = datetime.now(tz)
    
    # 📌 ИСПРАВЛЕНО: Убрано извлечение года из транскрипта.
    # Эта функция должна отвечать ТОЛЬКО за АКТУАЛЬНЫЙ контекст.
    
    # Подготовка дат для промптов
    today_str = target_date.strftime("%d.%m.%Y")
    days_of_week = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    today_weekday = days_of_week[target_date.weekday()]
    
    # Завтрашняя дата
    tomorrow = target_date + timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%d.%m.%Y")
    tomorrow_day_num = str(tomorrow.day)
    tomorrow_month_genitive = ["января", "февраля", "марта", "апреля", "мая", "июня", 
                               "июля", "августа", "сентября", "октября", "ноября", "декабря"][tomorrow.month - 1]
    tomorrow_weekday = days_of_week[tomorrow.weekday()]

    # Прошлая неделя (относительно ТЕКУЩЕЙ даты)
    days_since_monday = target_date.weekday()
    start_of_current_week = target_date - timedelta(days=days_since_monday)
    end_of_last_week = start_of_current_week - timedelta(days=1)
    start_of_last_week = end_of_last_week - timedelta(days=6)
    
    last_week_start = start_of_last_week.strftime("%d.%m.%Y")
    last_week_end = end_of_last_week.strftime("%d.%m.%Y")

    return {
        "current_year": target_date.year, # 📌 ИСПРАВЛЕНО: Добавлен текущий год для fallback
        "target_date": target_date,
        "today_str": today_str,
        "today_weekday": today_weekday,
        "tomorrow_str": tomorrow_str,
        "tomorrow_day_num": tomorrow_day_num,
        "tomorrow_month_genitive": tomorrow_month_genitive,
        "tomorrow_weekday": tomorrow_weekday,
        "last_week_start": last_week_start,
        "last_week_end": last_week_end
    }


def get_prompts_with_dates(transcript: str, mode: str) -> str | Dict[str, str]:
    """
    Генерирует промпты с корректными датами.
    📌 ИСПРАВЛЕНО: Тип возврата изменен для поддержки 'all'
    """
    
    # 📌 ИСПРАВЛЕНО: Контекст даты берется АКТУАЛЬНЫЙ, а не из транскрипта
    date_context = get_date_context()
    
    # 📌 ИСПРАВЛЕНО: Год для промпта 'todo' определяется из транскрипта,
    # с fallback на текущий год из (актуального) контекста
    transcript_year = extract_year_from_transcript(transcript) or date_context["current_year"]
    
    today_str = date_context["today_str"]
    today_weekday = date_context["today_weekday"]
    tomorrow_str = date_context["tomorrow_str"]
    tomorrow_day_num = date_context["tomorrow_day_num"]
    tomorrow_month_genitive = date_context["tomorrow_month_genitive"]
    tomorrow_weekday = date_context["tomorrow_weekday"]
    last_week_start = date_context["last_week_start"]
    last_week_end = date_context["last_week_end"]

    classification = """
<критерий_классификации>
- Встречи: события, включающие взаимодействие с людьми (встреча с..., звонок с..., обед с..., совещание с..., обсуждение с..., неформальная встреча с..., вечерняя встреча с..., онлайн-встреча с...)
- Задачи: самостоятельные действия (подготовить..., проверить..., обновить..., настроить..., утвердить..., ставить..., согласовать...)
- Если событие имеет время, но является задачей - помести в задачи, включив время в описание (например, — Подготовить отчет в 10:00)
- Если событие является встречей без времени - помести в встречи без времени в формате — Описание встречи
</критерий_классификации>
    """

    text_format = """
<формат_текста>
Текст представляет собой последовательность событий, разделенных запятыми или точками, в формате "ЧЧ:ММ описание, ЧЧ:ММ описание..."
Разбери их и извлеки все события.
</формат_текста>
    """

    prompts = {
        "todo": f"""
<роль>
Ты — сверхточный супер робот-аналитик для извлечения структурированных данных СТРОГО из предоставленного транскрипта.
Твоя единственная функция — преобразовать ТОЛЬКО этот текст в план, без добавления внешней информации.
</роль>

<критически_важные_ограничения>
🚫 СТРОГО ЗАПРЕЩЕНО:
- Выдумывать события, которых нет в тексте
- Добавлять информацию из памяти или других источников
- Интерпретировать или додумывать события
- Использовать данные из предыдущих транскриптов
- Дублировать задачи или создавать фразы вроде "Создать задачу создать..."
- Смешивать задачи между разными датами

✅ РАЗРЕШЕНО ТОЛЬКО:
- Извлекать события, которые ЯВНО упомянуты в тексте
- Работать исключительно с предоставленным транскриптом
- Формулировать задачи четко и без повторений
- Размещать каждую задачу СТРОГО в своей дате
- выводи все встречи, события
</критически_важные_ограничения>

<алгоритм_извлечения_по_датам>
1. Найди дату (например "13.09.{transcript_year}, понедельник")
2. Читай события ТОЛЬКО после этой даты ДО следующей даты
3. Размести встречи и задачи в блок этой даты
4. Перейди к следующей дате и повтори

ПРИМЕР:
"13.09.{transcript_year}, понедельник: встреча А, задача X
14.09.{transcript_year}, вторник: встреча Б, задача Y"
→ 13.09.{transcript_year}, понедельник: встреча А, задача X
→ 14.09.{transcript_year}, вторник: встреча Б, задача Y
НЕ СМЕШИВАЙ задачи между датами!
</алгоритм_извлечения_по_датам>

{text_format}

{classification}

<формат_ответа>
ДД.ММ.{transcript_year}, день недели на русском языке
Встречи:
— ЧЧ:ММ — Описание встречи
Задачи:
— Описание задачи (четко, без слова "задача" в начале)

КРИТИЧЕСКИ ВАЖНО ПО ФОРМАТИРОВАНИЮ:
- Дни недели ТОЛЬКО на русском: понедельник, вторник, среда, четверг, пятница, суббота, воскресенье
- Используй ДЛИННОЕ ТИРЕ (—) для списков
- ВСЕГДА пиши "Задачи:" (во множественном числе), НЕ "Задача:"
- КАЖДАЯ задача начинается с "— " (длинное тире + пробел)
- НЕ пиши задачи предложениями через точки: "Создать. Внедрить. Запустить."
- КАЖДАЯ задача на отдельной строке с тире: "— Создать", "— Внедрить", "— Запустить"
- НЕ добавляй лишних строк в конце типа "Рабочих задач нет."
- Если у конкретной даты нет событий - просто не показывай эту дату
- Сортируй встречи и задачи хронологически по времени
</формат_ответа>
""",

        "today_meetings": f"""
<роль>
Ты — лазерно-точный робот-фильтр для извлечения ВСЕХ событий дня {today_str}, {today_weekday}.
Твоя задача — показать и встречи, и задачи на сегодня.
</роль>

{text_format}

<критически_важная_задача>
ИЗВЛЕКИ АБСОЛЮТНО ВСЕ встречи и задачи из предоставленного текста, которые относятся к дате {today_str} (сегодня, {today_weekday}).
⚠️ ОСОБЕННО ВАЖНО ДЛЯ ВРЕМЕНИ:
- "13 часов ровно" = "13:00"
- "15 часов 30 минут" = "15:30"
- НЕ МЕНЯЙ время на свое усмотрение!
- ТОЧНО копируй все встречи из блока!

⚠️ ОСОБЕННО ВАЖНО ДЛЯ ЗАДАЧ:
Задачи в тексте идут сразу после встреч.
</критически_важная_задача>

{classification}

<строгие_требования_к_ответу>
🚫 СТРОГО ЗАПРЕЩЕНО:
- Показывать размышления или объяснения  
- Добавлять комментарии о процессе анализа
- Писать на английском языке (ТОЛЬКО РУССКИЙ!)
- Объяснять алгоритм работы
- Добавлять лишние строки типа 'Если событий нет: "На сегодня рабочих планов нет"'
- Переводить задачи на английский язык
- ИЗМЕНЯТЬ ВРЕМЯ ИЗ ОРИГИНАЛА!
✅ РАЗРЕШЕНО ТОЛЬКО:
- Чистый список встреч и задач НА РУССКОМ ЯЗЫКЕ
- Ответ сразу в требуемом формате
- Задачи формулировать по-русски (запустить, создать, организовать)
- ТОЧНОЕ время в формате ЧЧ:ММ
</строгие_требования_к_ответу>

<точный_формат>
Встречи:
— 13:00 — описание встречи
— 15:30 — описание встречи  
— 18:00 — описание встречи

Задачи:
— описание задачи
— описание задачи

КРИТИЧЕСКИ ВАЖНО:
- ВСЕГДА используй ДЛИННОЕ ТИРЕ (—) в начале каждого пункта
- НЕ используй обычное тире (-) 
- "Задачи:" во множественном числе, даже если задача одна
- ВСЕ ЗАДАЧИ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
- НЕ добавляй никаких лишних строк или комментариев
- ВРЕМЯ СТРОГО В ФОРМАТЕ ЧЧ:ММ (13:00, а не 13:30)
- Сортируй встречи и задачи хронологически по времени

Если событий нет: "На сегодня рабочих планов нет"

НАЧИНАЙ ОТВЕТ СРАЗУ СО СЛОВА "Встречи:" БЕЗ РАЗМЫШЛЕНИЙ!
</точный_формат>
""",

        "tomorrow_meetings": f"""
<роль>
Ты — сверх-точный робот-фильтр для анализа планов на {tomorrow_str}, {tomorrow_weekday}.
</роль>

<контекст_даты>
- Завтрашний день: **{tomorrow_day_num} {tomorrow_month_genitive} ({tomorrow_weekday})**.
- Формальная дата: {tomorrow_str}.
</контекст_даты>

{text_format}

<критически_важная_задача>
Извлеки АБСОЛЮТНО ВСЕ встречи и задачи из предоставленного текста, которые относятся к дате {tomorrow_str} (завтра, {tomorrow_weekday}).
⚠️ КРИТИЧЕСКИ ВАЖНО ДЛЯ ВРЕМЕНИ:
- "12 часов ровно" → ОБЯЗАТЕЛЬНО преобразуй в "12:00"
- "15 часов ровно" → ОБЯЗАТЕЛЬНО преобразуй в "15:00"
- "17 часов 30 минут" → ОБЯЗАТЕЛЬНО преобразуй в "17:30"
- НЕ оставляй "часов ровно" в ответе!
- ВСЕГДА используй формат ЧЧ:ММ

⚠️ КРИТИЧЕСКИ ВАЖНО:
НЕ пропускай ни одной встречи или задачи!
Внимательно читай ВЕСЬ текст!
</критически_важная_задача>

{classification}

<точный_формат>
Встречи:
— 12:00 — описание встречи
— 15:00 — описание встречи
— 17:30 — описание встречи

Задачи:
— описание задачи
— описание задачи

КРИТИЧЕСКИ ВАЖНО:
- ВСЕГДА используй ДЛИННОЕ ТИРЕ (—) в начале каждого пункта
- НЕ используй обычное тире (-)
- "Задачи:" во множественном числе, даже если задача одна
- ВРЕМЯ ТОЛЬКО В ФОРМАТЕ ЧЧ:ММ (12:00, НЕ "12 часов ровно")
- Сортируй встречи и задачи хронологически по времени
- Если планов нет: "На завтра рабочих планов нет"

НАЧИНАЙ ОТВЕТ СРАЗУ СО СЛОВА "Встречи:" БЕЗ РАЗМЫШЛЕНИЙ!
</точный_формат>
""",

        "last_week_meetings": f"""
<роль>
Ты — строгий архивариус для анализа РАБОЧИХ событий прошлой недели из предоставленного транскрипта.
</роль>

<критически_важная_задача>
Найди события за прошлую неделю ({last_week_start} - {last_week_end}) СТРОГО из предоставленного транскрипта.
НЕ ВЫДУМЫВАЙ события, которых нет в тексте!
</критически_важная_задача>

{classification}

<строгие_правила_форматирования>
ДД.ММ.ГГГГ, день недели на русском языке (📌 ИСПРАВЛЕНО: ГГГГ добавлен для ясности)
Встречи:
— ЧЧ:ММ — Описание встречи
Задачи:
— Описание задачи

КРИТИЧЕСКИ ВАЖНО:
- ВСЕГДА используй ДЛИННОЕ ТИРЕ (—) в начале каждого пункта
- НЕ используй обычное тире (-)
- "Задачи:" во множественном числе, даже если задача одна
- Группируй строго по датам!
Сортируй хронологически!
- Каждое событие размещай в правильной дате
- Дни недели ТОЛЬКО на русском: понедельник, вторник, среда, четверг, пятница, суббота, воскресенье
- Если рабочих событий нет: "На прошлой неделе рабочих встреч не было"
- Начинай ответ сразу с даты, без пояснений
- НЕ добавляй события из памяти или других источников
</строгие_правила_форматирования>
"""
    }
    
    # 📌 ИСПРАВЛЕНО: Добавлена поддержка 'all' для админ-панели
    if mode == "all":
        return prompts
    
    return prompts.get(mode, "")


class LLMClient:
    """Client for interacting with Ollama LLM service"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def check_connection(self) -> bool:
        """Check if Ollama service is available"""
        try:
            health_url = self.config.ollama_url.replace('/api/chat', '/api/tags')
            self.logger.debug(f"Checking Ollama availability: {health_url}")

            response = requests.get(health_url, timeout=10)
            is_available = response.status_code == 200

            if is_available:
                self.logger.info("✅ Ollama is available and ready")
            else:
                self.logger.error(f"❌ Ollama unavailable: HTTP {response.status_code}")

            return is_available

        except requests.exceptions.ConnectionError:
            self.logger.error("❌ Cannot connect to Ollama (check if service is running)")
            return False
        except requests.exceptions.Timeout:
            self.logger.error("❌ Connection timeout to Ollama")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error checking Ollama: {e}")
            return False

    def query(self, transcript: str, mode: str = "todo") -> str:
        """Query LLM for transcript analysis"""
        self.logger.info(f"🔍 Starting transcript analysis in mode '{mode}'")
        
        # Validate input
        
        is_valid, error_msg = validate_input(transcript)
        if not is_valid:
            self.logger.error(f"❌ Validation error: {error_msg}")
            return f"❌ Validation error: {error_msg}"

        # Check service availability
        if not self.check_connection():
            self.logger.error("❌ Ollama unavailable")
            return "❌ Ollama unavailable. Check service startup."

        # Get prompts with correct dates from transcript
        # 📌 ИСПРАВЛЕНО: Теперь эта функция использует АКТУАЛЬНЫЕ даты для 'today', 'tomorrow' и 'last_week'
        prompt_text = get_prompts_with_dates(transcript, mode)
        if not prompt_text:
            available_modes = ["todo", "today_meetings", "tomorrow_meetings", "last_week_meetings"]
            self.logger.error(f"❌ Invalid mode: {mode}. Available: {available_modes}")
            return f"❌ Invalid analysis mode. Available modes: {available_modes}"

        # Preprocess transcript
        self.logger.debug("🔄 Preprocessing transcript")
        preprocessed_transcript = preprocess_input(transcript)

        # Normalize dates
        transcript_year = extract_year_from_transcript(transcript)
        normalized_transcript = normalize_dates(preprocessed_transcript, transcript_year)

        # 📌 ИСПРАВЛЕНО: Убрана некорректная предварительная фильтрация.
        # LLM теперь всегда получает полный нормализованный текст
        # и фильтрует его сам на основе дат, указанных в промпте.
        analysis_text = normalized_transcript
        
        # 📌 ИСПРАВЛЕНО: Добавлена проверка на пустой транскрипт ПОСЛЕ нормализации
        if not analysis_text.strip():
            self.logger.warning("Транскрипт пуст после нормализации.")
            if mode == "today_meetings":
                return "На сегодня рабочих планов нет"
            if mode == "tomorrow_meetings":
                return "На завтра рабочих планов нет"
            if mode == "last_week_meetings":
                return "На прошлой неделе рабочих встреч не было"
            if mode == "todo":
                return "Нет событий для планирования."
            return "Текст пуст, анализ невозможен." # Общий fallback

        # Build final prompt
        final_prompt = f"{prompt_text.strip()}\n\n<текст_для_анализа>\n{analysis_text.strip()}\n</текст_для_анализа>"

        # Build request payload
        params = LLM_PRESETS[self.config.preset]
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "stream": False,
            **params
        }

        self.logger.info(f"🤖 Sending request to LLM (mode: {mode}, model: {self.config.model_name})")
        self.logger.debug(f"[LLM PAYLOAD]\n{json.dumps(payload, indent=2, ensure_ascii=False)}")

        # Execute request with error handling
        try:
            response = requests.post(self.config.ollama_url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            result = response.json()
            raw_text = result.get("message", {}).get("content", "⚠️ Model returned no text.")

            self.logger.info(f"✅ Received LLM response (length: {len(raw_text)} characters)")
            self.logger.debug(f"[RAW LLM RESPONSE]\n{raw_text}")

        except requests.exceptions.ConnectionError:
            error_msg = "Connection error to Ollama"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}. Check service."
        except requests.exceptions.Timeout:
            error_msg = "Request timeout to LLM"
            self.logger.error(f"❌ {error_msg} (exceeded {self.config.timeout} seconds)")
            return f"❌ {error_msg}. Try with shorter text."
        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP request error: {e}"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {e}"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"
        except KeyError as e:
            error_msg = f"Unexpected response structure: {e}"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"

        # Clean and process response
        self.logger.debug("🧹 Cleaning and processing LLM response")
        sanitized_text = sanitize_llm_response(raw_text)

        self.logger.info(f"✅ Analysis completed successfully (mode: {mode})")
        self.logger.debug(f"[FINAL RESULT]\n{sanitized_text}")

        return sanitized_text

    def query_with_retry(self, text: str, mode: str = "todo") -> str:
        """
        📌 ИСПРАВЛЕНО: Запрос к LLM с повторными попытками.
        Логика переписана на цикл `while` для большей стабильности и предсказуемости.
        """
        attempt = 0
        result = ""
        max_iterations = self.config.max_retries + 10  # Fallback to prevent theoretical infinite loop
        iteration = 0  # 📌 ИСПРАВЛЕНО: Добавлена инициализация переменной
        
        while attempt <= self.config.max_retries and iteration < max_iterations:
            iteration += 1
            if attempt > 0:
                self.logger.warning(
                    f"❗️ LLM вернул пустой ответ, но в тексте есть ключевые слова. "
                    f"Повторяю запрос... (Попытка {attempt}/{self.config.max_retries})"
                )
            
            result = self.query(text, mode)

            # Критерии для повторной попытки
            no_plans_phrases = ["планов нет", "встреч не было", "задач нет", "рабочих задач нет"]
            contains_meeting_words = any(word in text.lower() for word in
                                         ["встреча", "задач", "совещание", "план", "звонок", "переговоры", "обсуждение"])
            is_empty_response = any(phrase in result.lower() for phrase in no_plans_phrases)
            
            # Условие для выхода из цикла: если ответ не пустой ИЛИ в тексте нет ключевых слов,
            # то результат считаем финальным и выходим.
            should_retry = (self.config.max_retries > 0 and 
                            is_empty_response and
                            contains_meeting_words and 
                            len(text) > 100)

            if not should_retry:
                break
            
            attempt += 1

        return result


# Global client instance
_config = LLMConfig(
    ollama_url=OLLAMA_URL,
    model_name=MODEL_NAME,
    preset=CURRENT_PRESET
)
_llm_client = LLMClient(_config)

# --- Функции ---

def check_ollama_connection() -> bool:
    """Проверка доступности сервиса Ollama"""
    return _llm_client.check_connection()

def query_llm(transcript: str, mode: str = "todo") -> str:
    """Основная функция анализа транскриптов с помощью LLM"""
    return _llm_client.query(transcript, mode)

def query_llm_with_retry(text: str, mode: str = "todo", retries: int = 2) -> str:
    """Запрос к LLM с повторными попытками"""
    # Создаем временную копию конфигурации, чтобы не менять глобальные настройки
    temp_config = LLMConfig(
        ollama_url=_llm_client.config.ollama_url,
        model_name=_llm_client.config.model_name,
        preset=_llm_client.config.preset,
        timeout=_llm_client.config.timeout,
        max_retries=retries
    )
    temp_client = LLMClient(temp_config)
    # Вызываем уже исправленный метод с циклом внутри
    return temp_client.query_with_retry(text, mode)

def set_model(model_name: str):
    """Установка модели LLM"""
    global _llm_client
    _llm_client.config.model_name = model_name
    logger.info(f"🔄 Модель установлена: {model_name}")

def get_model() -> str:
    """Получение текущей модели LLM"""
    return _llm_client.config.model_name

def set_preset(preset: str) -> bool:
    """Установка пресета параметров LLM"""
    global _llm_client
    if preset in LLM_PRESETS:
        _llm_client.config.preset = preset
        logger.info(f"🔄 Режим LLM установлен: {preset}")
        return True
    else:
        logger.warning(f"❌ Неизвестный пресет: {preset}")
        return False

# --- ОСНОВНЫЕ ФУНКЦИИ ДЛЯ ИСПОЛЬЗОВАНИЯ ---

def analyze_todo(audio_text: str) -> str:
    """📝 Что мне нужно запланировать?"""
    return query_llm_with_retry(audio_text, mode="todo")

def analyze_today(audio_text: str) -> str:
    """👥 С кем я сегодня встречался?"""
    return query_llm_with_retry(audio_text, mode="today_meetings")

def analyze_tomorrow(audio_text: str) -> str:
    """📅 Какой план встреч у меня завтра?"""
    return query_llm_with_retry(audio_text, mode="tomorrow_meetings")

def analyze_last_week(audio_text: str) -> str:
    """📊 Какие встречи были на прошлой неделе?"""
    return query_llm_with_retry(audio_text, mode="last_week_meetings")

# --- Тестирование ---
if __name__ == "__main__":
    # Тестовый текст с будущей датой
    future_text = """
    План на октябрь 2025 года.
    13 октября понедельник. Встреча в 9:00 совещание с командой разработки. 
    В 14.30 звонок с клиентом по проекту.
    Задача - подготовить отчет по продажам.
    
    14 октября вторник. Встреча в 10:00 с HR по найму нового сотрудника.
    Задача - обновить CRM систему.
    
    15 октября среда. Встреча с директором в 16:00.
    """
    
    print("🤖 ТЕСТИРОВАНИЕ СИСТЕМЫ С БУДУЩИМИ ДАТАМИ")
    print("=" * 60)
    
    print(f"📋 Транскрипт содержит даты октября 2025 года")
    print(f"📅 Текущая дата (для контекста): {datetime.now().strftime('%d.%m.%Y')}")
    print("=" * 60)
    
    print("\n📋 1. Что мне нужно запланировать? (Полный список из текста)")
    print("-" * 40)
    result = analyze_todo(future_text)
    print(result)
    
    # Проверяем, что даты корректны
    if "10.2025" in result:
        print("\n✅ Год корректно определен из транскрипта!")
    else:
        print("\n❌ Проблема с определением года!")
        
    if "13.10.2025, понедельник" in result and "14.10.2025, вторник" in result:
        print("✅ Даты из транскрипта распознаны и разделены правильно!")
    else:
        print("❌ Проблема с распознаванием или разделением дат!")

    # 📌 ИСПРАВЛЕНО: Добавлен тест для проверки новой логики (сегодня/завтра)
    print("\n📋 2. Тестирование 'сегодня' и 'завтра' (контекст актуальной даты)")
    print("-" * 40)
    
    # Предполагаем, что сегодня НЕ 13-15 октября 2025
    today_result = analyze_today(future_text)
    print(f"Результат 'Сегодня': {today_result}")
    if "планов нет" in today_result:
        print("✅ 'Сегодня' корректно вернул 'нет планов' (т.к. даты в тексте не совпадают с реальным 'сегодня')")
    else:
        print("❌ 'Сегодня' вернул события, хотя не должен был!")
        
    tomorrow_result = analyze_tomorrow(future_text)
    print(f"Результат 'Завтра': {tomorrow_result}")
    if "планов нет" in tomorrow_result:
        print("✅ 'Завтра' корректно вернул 'нет планов' (т.к. даты в тексте не совпадают с реальным 'завтра')")
    else:
        print("❌ 'Завтра' вернул события, хотя не должен был!")