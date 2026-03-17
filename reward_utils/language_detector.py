from typing import Optional

try:
    from lingua import Language, LanguageDetectorBuilder
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False
    Language = None
    LanguageDetectorBuilder = None


LANG_MAP_TO_LINGUA = {
    "en": Language.ENGLISH if LINGUA_AVAILABLE else None,
    "zh": Language.CHINESE if LINGUA_AVAILABLE else None,
    "de": Language.GERMAN if LINGUA_AVAILABLE else None,
    "ru": Language.RUSSIAN if LINGUA_AVAILABLE else None,
    "ko": Language.KOREAN if LINGUA_AVAILABLE else None,
    "fr": Language.FRENCH if LINGUA_AVAILABLE else None,
    "es": Language.SPANISH if LINGUA_AVAILABLE else None,
    "pt": Language.PORTUGUESE if LINGUA_AVAILABLE else None,
    "it": Language.ITALIAN if LINGUA_AVAILABLE else None,
    "nl": Language.DUTCH if LINGUA_AVAILABLE else None,
}


class LanguageDetector:
    def __init__(self):
        self._detector = None
        if LINGUA_AVAILABLE:
            languages = [lang for lang in LANG_MAP_TO_LINGUA.values() if lang is not None]
            self._detector = LanguageDetectorBuilder.from_languages(*languages).build()
        else:
            raise ImportError("lingua is not installed, cannot use language detector")
    
    def is_language_match(self, text: str, target_lang_code: str) -> bool:
        if not LINGUA_AVAILABLE or self._detector is None:
            return True
        
        if not text or not text.strip():
            return True
        
        target_lang = LANG_MAP_TO_LINGUA.get(target_lang_code)
        if target_lang is None:
            return True
        
        detected_lang = self._detector.detect_language_of(text)
        return detected_lang == target_lang


_language_detector_instance: Optional[LanguageDetector] = None


def get_language_detector() -> LanguageDetector:
    global _language_detector_instance
    if _language_detector_instance is None:
        _language_detector_instance = LanguageDetector()
    return _language_detector_instance


def is_language_match(text: str, target_lang_code: str) -> bool:
    detector = get_language_detector()
    return detector.is_language_match(text, target_lang_code)
