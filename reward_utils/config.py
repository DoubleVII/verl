# --- Language Configuration ---
VALID_LANGS = [
    "en",
    "zh",
    "de",
    "ja",
    "fr",
    "es",
    "pt",
    "ru",
    "it",
    "nl",
]

TOWER_LANGS = ["en", "de", "fr", "nl", "it", "es", "pt", "ko", "ru", "zh"]
OOD_LANGS = ["ru", "ja", "uk"]

LANG_MAP = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "ru": "Russian",
    "ko": "Korean",
    "ja": "Japanese",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "th": "Thai",
    "ro": "Romanian",
    "ar": "Arabic",
    "el": "Greek",
    "vi": "Vietnamese",
    "uk": "Ukrainian",
}



flores_langcode_map = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "nl": "nld_Latn",
    "it": "ita_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "zh": "cmn_Hans",
    "ja": "jpn_Jpan",
    "lt": "lit_Latn",
    "th": "tha_Thai",
    "cs": "ces_Latn",
    "uk": "ukr_Cyrl",
    "ar": "ara_Arab",
}


LANGUAGE_BY_CODE = {
    "de_DE": "German",
    "es_MX": "Spanish",
    "fr_FR": "French",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
    "ko_KR": "Korean",
    "nl_NL": "Dutch",
    "pt_PT": "Portuguese",
    "ru_RU": "Russian",
    "uk_UA": "Ukrainian",
    "zh_CN": "Chinese",
}



RANKING_TEST_DATA_META_INFO = {
    "tower_zhen_ranking_testset": {
        "src_lang": "zh/en",
        "trg_lang": "en/zh",
        "path": "parquet_data/TowerBlocks-MT-Ranking/test.parquet",
    },
    "wmt_newstest2020_psqm": {
        "src_lang": "zh",
        "trg_lang": "en",
        "path": "parquet_data/MT_Ranking_Metric_Test/data/wmt_newstest2020_zhen_psqm_group_test.parquet",
    },
    "wmt_generalMT2022_enzh_mqm": {
        "src_lang": "en",
        "trg_lang": "zh",
        "path": "parquet_data/MT_Ranking_Metric_Test/data/wmt_generalMT2022_enzh_mqm_group_test.parquet",
    },
    "wmt_generalMT2022_zhen_mqm": {
        "src_lang": "zh",
        "trg_lang": "en",
        "path": "parquet_data/MT_Ranking_Metric_Test/data/wmt_generalMT2022_zhen_avgmqm_group_test.parquet",
    },
    "wmt_generalMT2022_ende_mqm": {
        "src_lang": "en",
        "trg_lang": "de",
        "path": "parquet_data/MT_Ranking_Metric_Test/data/wmt_generalMT2022_ende_avgmqm_group_test.parquet",
    },
    "wmt_generalMT2022_enru_mqm": {
        "src_lang": "en",
        "trg_lang": "ru",
        "path": "parquet_data/MT_Ranking_Metric_Test/data/wmt_generalMT2022_enru_avgmqm_group_test.parquet",
    },
    "seedx_challenge_ranking": {
        "src_lang": "en/zh",
        "trg_lang": "zh/en",
        "path": "parquet_data/MT_Ranking_Metric_Test/data/seedx_challenge.ranking.max_bleurt.parquet",
    },
}

MT_TEST_DATA_META_INFO = {
    "mtr1_zhen_test": {
        "src_lang": None,
        "trg_lang": None,
        "path": "parquet_data/test_data_mt/mtr1_zhen_test.parquet",
    },
    "seedx_challenge": {
        "src_lang": None,
        "trg_lang": None,
        "path": "parquet_data/test_data_mt/seedx_challenge.parquet",
    },
    "wmt23_zhen": {
        "src_lang": "zh",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/wmt23_zhen.parquet",
    },
    "wmt24_enzh": {
        "src_lang": "en",
        "trg_lang": "zh",
        "path": "parquet_data/test_data_mt/wmt24_enzh.parquet",
    },
    "seedx_challenge_zhen": {
        "src_lang": "zh",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/seedx_challenge_zhen.parquet",
    },
    "seedx_challenge_enzh": {
        "src_lang": "en",
        "trg_lang": "zh",
        "path": "parquet_data/test_data_mt/seedx_challenge_enzh.parquet",
    },
    "flores_de_en": {
        "src_lang": "de",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_de_en.parquet",
    },
    "flores_en_de": {
        "src_lang": "en",
        "trg_lang": "de",
        "path": "parquet_data/test_data_mt/flores/flores_en_de.parquet",
    },
    "flores_en_es": {
        "src_lang": "en",
        "trg_lang": "es",
        "path": "parquet_data/test_data_mt/flores/flores_en_es.parquet",
    },
    "flores_es_en": {
        "src_lang": "es",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_es_en.parquet",
    },
    "flores_en_fr": {
        "src_lang": "en",
        "trg_lang": "fr",
        "path": "parquet_data/test_data_mt/flores/flores_en_fr.parquet",
    },
    "flores_fr_en": {
        "src_lang": "fr",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_fr_en.parquet",
    },
    "flores_en_it": {
        "src_lang": "en",
        "trg_lang": "it",
        "path": "parquet_data/test_data_mt/flores/flores_en_it.parquet",
    },
    "flores_it_en": {
        "src_lang": "it",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_it_en.parquet",
    },
    "flores_en_nl": {
        "src_lang": "en",
        "trg_lang": "nl",
        "path": "parquet_data/test_data_mt/flores/flores_en_nl.parquet",
    },
    "flores_nl_en": {
        "src_lang": "nl",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_nl_en.parquet",
    },
    "flores_en_pt": {
        "src_lang": "en",
        "trg_lang": "pt",
        "path": "parquet_data/test_data_mt/flores/flores_en_pt.parquet",
    },
    "flores_pt_en": {
        "src_lang": "pt",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_pt_en.parquet",
    },
    "flores_en_ja": {
        "src_lang": "en",
        "trg_lang": "ja",
        "path": "parquet_data/test_data_mt/flores/flores_en_ja.parquet",
    },
    "flores_ja_en": {
        "src_lang": "ja",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_ja_en.parquet",
    },
    "flores_en_ko": {
        "src_lang": "en",
        "trg_lang": "ko",
        "path": "parquet_data/test_data_mt/flores/flores_en_ko.parquet",
    },
    "flores_ko_en": {
        "src_lang": "ko",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_ko_en.parquet",
    },
    "flores_en_ru": {
        "src_lang": "en",
        "trg_lang": "ru",
        "path": "parquet_data/test_data_mt/flores/flores_en_ru.parquet",
    },
    "flores_ru_en": {
        "src_lang": "ru",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_ru_en.parquet",
    },
    "flores_en_uk": {
        "src_lang": "en",
        "trg_lang": "uk",
        "path": "parquet_data/test_data_mt/flores/flores_en_uk.parquet",
    },
    "flores_uk_en": {
        "src_lang": "uk",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_uk_en.parquet",
    },
    "flores_en_zh": {
        "src_lang": "en",
        "trg_lang": "zh",
        "path": "parquet_data/test_data_mt/flores/flores_en_zh.parquet",
    },
    "flores_zh_en": {
        "src_lang": "zh",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/flores/flores_zh_en.parquet",
    },
    "wmt24pp_en_de": {
        "src_lang": "en",
        "trg_lang": "de",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_de.parquet",
    },
    "wmt24pp_en_es": {
        "src_lang": "en",
        "trg_lang": "es",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_es.parquet",
    },
    "wmt24pp_en_fr": {
        "src_lang": "en",
        "trg_lang": "fr",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_fr.parquet",
    },
    "wmt24pp_en_it": {
        "src_lang": "en",
        "trg_lang": "it",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_it.parquet",
    },
    "wmt24pp_en_nl": {
        "src_lang": "en",
        "trg_lang": "nl",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_nl.parquet",
    },
    "wmt24pp_en_pt": {
        "src_lang": "en",
        "trg_lang": "pt",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_pt.parquet",
    },
    "wmt24pp_en_ja": {
        "src_lang": "en",
        "trg_lang": "ja",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_ja.parquet",
    },
    "wmt24pp_en_ko": {
        "src_lang": "en",
        "trg_lang": "ko",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_ko.parquet",
    },
    "wmt24pp_en_ru": {
        "src_lang": "en",
        "trg_lang": "ru",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_ru.parquet",
    },
    "wmt24pp_en_uk": {
        "src_lang": "en",
        "trg_lang": "uk",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_uk.parquet",
    },
    "wmt24pp_en_zh": {
        "src_lang": "en",
        "trg_lang": "zh",
        "path": "parquet_data/test_data_mt/wmt24pp/wmt24pp_en_zh.parquet",
    },
    "wmt23_de_en": {
        "src_lang": "de",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/wmt23/wmt23_de_en.parquet",
    },
    "wmt23_ja_en": {
        "src_lang": "ja",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/wmt23/wmt23_ja_en.parquet",
    },
    "wmt23_ru_en": {
        "src_lang": "ru",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/wmt23/wmt23_ru_en.parquet",
    },
    "wmt23_uk_en": {
        "src_lang": "uk",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/wmt23/wmt23_uk_en.parquet",
    },
    "wmt23_zh_en": {
        "src_lang": "zh",
        "trg_lang": "en",
        "path": "parquet_data/test_data_mt/wmt23/wmt23_zh_en.parquet",
    },
}


candidate_identifiers = ["A", "B", "C", "D", "E", "F", "G", "H"]
