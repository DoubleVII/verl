import utils


def test_compute_score():
    batch_size = 60

    data_sources = "unknown"

    good_solution_strs = "abaaba <think> reasoning process here </think> <translate> 特别为儿童准备了宠物动物园、“稻草充气城堡”、拖拉机驾驶（有专人看护）和拖拉机冲浪。</translate> abaaba."
    bad_solution_strs = "abaaba <think> reasoning process here </think> <translate> 专门针对儿童的游乐园，川古堡和拉斯维加斯帝国（有专人看管）和拖拉机</translate> abaaba."
    none_solution_strs = "abaaba <think> reasoning process here </think> <translate> 专门针对儿童的游乐园，川古堡和拉斯维加斯帝国（有专人看管）和拖拉机abaaba."

    extra_infos = {'src_text': 'Specially for children, there is a petting zoo, "straw bouncy castle", tractor driving (under supervision) and tractor surfing.', 'tgt_text': 'not available', "lang_pair": "en-zh"}

    data_sources = [data_sources] * batch_size
    solution_strs = [good_solution_strs] * (batch_size//3) + [none_solution_strs] * (batch_size//3) + [bad_solution_strs] * (batch_size//3)
    extra_infos = [extra_infos] * batch_size

    ground_truths = [extra_infos_item['tgt_text'] for extra_infos_item in extra_infos]

    scores = utils.compute_score(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )

    print(scores)






def test_compute_score_progressive():
    batch_size = 60

    data_sources = "unknown"

    good_solution_strs = """<draft> The girl looks very much like her mother, with light blue eyes and white hair adorned with blue ribbons. The Ice King believes having a daughter will make him appear weak. After all, until now, he only had sons. </draft>

<analysis> The draft translation accurately captures the meaning of the original Chinese text. It conveys the physical description of the girl and the Ice King's perception of his daughter. The sentence structure and word choice are appropriate and maintain the original intent. The translation is clear and coherent, preserving the nuances of the source text. </analysis>

<translation> The girl looks very much like her mother, with light blue eyes and white hair adorned with blue ribbons. The Ice King believes having a daughter will make him appear weak. After all, until now, he only had sons. </translation>
"""
    bad_solution_strs = """<draft> The girl looked exactly like her father, with light yellow eyes and white hair. The Ice King thought that having a daughter would make him look weak. After all, he had only had one father so far. </draft>

<analysis> The draft translation accurately captures the meaning of the original Chinese text. It conveys the physical description of the girl and the Ice King's perception of his daughter. The sentence structure and word choice are appropriate and maintain the original intent. The translation is clear and coherent, preserving the nuances of the source text. </analysis>

<translation> The girl looked exactly like her father, with light blue eyes and white hair. The Ice King thought that having a daughter would make him look weak. After all, he had only had sons so far. </translation>
"""
    none_solution_strs = """<drafwt> The girl looked exactly like her father, with light yellow eyes and white hair. The Ice King thought that having a daughter would make him look weak. After all, he had only had one father so far. </draft>

<analysis> The draft translation accurately captures the meaning of the original Chinese text. It conveys the physical description of the girl and the Ice King's perception of his daughter. The sentence structure and word choice are appropriate and maintain the original intent. The translation is clear and coherent, preserving the nuances of the source text. </analysis>

<translatio The girl looked exactly like her father, with light blue eyes and white hair. The Ice King thought that having a daughter would make him look weak. After all, he had only had sons so far. </translation>
"""

    extra_infos = {'src_text': '那个女孩长相很像她的母亲，有着浅蓝色的眼睛和白色的头发，上面还点缀着蓝色的丝带。冰之王认为有了女儿会使他显得软弱。毕竟，直到现在他只有儿子。', 'tgt_text': 'not available', "lang_pair": "zh-en"}

    data_sources = [data_sources] * batch_size
    solution_strs = [good_solution_strs] * (batch_size//3) + [none_solution_strs] * (batch_size//3) + [bad_solution_strs] * (batch_size//3)
    extra_infos = [extra_infos] * batch_size

    ground_truths = [extra_infos_item['tgt_text'] for extra_infos_item in extra_infos]

    scores = utils.compute_score_progressive(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )

    print(scores)



def test_get_bleu_penalty():
    mt_texts = [
        'Девочка была очень похожа на свою мать, с голубыми глазами и белыми волосами, украшенными синими лентами. Ледяной Король думал, что рождение дочери сделает его слабым. В конце концов, до сих пор у него были только сыновья.',
        'Девочка была очень похожа на свою мать, с голубыми глазами и белыми волосами, украшенными синими лентами. 冰之王 Король думал, что рождение дочери сделает его слабым. В конце концов, до сих пор у него были только сыновья.',
        'The girl looked very much like her mother, with blue eyes and white hair adorned with blue ribbons. The Ice King thought that having a daughter would make him weak. After all, he had only had sons so far.',
        '1+ 西索的陆地、水描绘中心新画廊展览',
        'Siso 的陆地、水描绘 center 新画廊展览',
        "New Gallery Exhibit at Cecil's Land, Water Depictions of land Center"
        ]
    src_texts = [
        '那个女孩长相很像她的母亲，有着浅蓝色的眼睛和白色的头发，上面还点缀着蓝色的丝带。冰之王认为有了女儿会使他显得软弱。毕竟，直到现在他只有儿子。',
        '那个女孩长相很像她的母亲，有着浅蓝色的眼睛和白色的头发，上面还点缀着蓝色的丝带。冰之王认为有了女儿会使他显得软弱。毕竟，直到现在他只有儿子。',
        '那个女孩长相很像她的母亲，有着浅蓝色的眼睛和白色的头发，上面还点缀着蓝色的丝带。冰之王认为有了女儿会使他显得软弱。毕竟，直到现在他只有儿子。',
        'Изображения земли и воды Сисо в новой галерее выставки',
        'Изображения земли и воды Сисо в новой галерее выставки',
        'Изображения земли и воды Сисо в новой галерее выставки',
    ]
    ref_texts = [
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        "1+ Siso's depictions of land, water center new gallery exhibition",
        "Siso's depictions of land, water center new gallery exhibition",
        "Siso's depictions of land, water center new gallery exhibition",
        ]
    src_langs = [
        'zh',
        'zh',
        'zh',
        'ru',
        'ru',
        'ru',
    ]
    print(utils.get_bleu_penalty(mt_texts, src_texts, ref_texts, src_langs))



def test_get_length_penalty():
    mt_texts = [
        'Девочка была очень похожа на свою мать, с голубыми глазами и белыми волосами, украшенными синими лентами. Ледяной Король думал, что рождение дочери сделает его слабым. В конце концов, до сих пор у него были только сыновья.',
        3*'Девочка была очень похожа на свою мать, с голубыми глазами и белыми волосами, украшенными синими лентами. 冰之王 Король думал, что рождение дочери сделает его слабым. В конце концов, до сих пор у него были только сыновья.',
        4*'The girl looked very much like her mother, with blue eyes and white hair adorned with blue ribbons. The Ice King thought that having a daughter would make him weak. After all, he had only had sons so far.',
        5*'The girl looked very much like her mother, with blue eyes and white hair adorned with blue ribbons. The Ice King thought that having a daughter would make him weak. After all, he had only had sons so far.',
        ]
    ref_texts = [
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        ]
    src_langs = [
        'zh',
        'zh',
        'zh',
    ]
    print(utils.get_length_penalty(mt_texts, ref_texts))


def test_apply_length_penalty_filter():
    scores = [0] * 4
    mt_texts = [
        'Девочка была очень похожа на свою мать, с голубыми глазами и белыми волосами, украшенными синими лентами. Ледяной Король думал, что рождение дочери сделает его слабым. В конце концов, до сих пор у него были только сыновья.',
        3*'Девочка была очень похожа на свою мать, с голубыми глазами и белыми волосами, украшенными синими лентами. 冰之王 Король думал, что рождение дочери сделает его слабым. В конце концов, до сих пор у него были только сыновья.',
        4*'The girl looked very much like her mother, with blue eyes and white hair adorned with blue ribbons. The Ice King thought that having a daughter would make him weak. After all, he had only had sons so far.',
        5*'The girl looked very much like her mother, with blue eyes and white hair adorned with blue ribbons. The Ice King thought that having a daughter would make him weak. After all, he had only had sons so far.',
        ]
    ref_texts = [
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        'The girl looked a lot like her mother, with light blue eyes and white hair accented with blue ribbons. The Ice King thought that having a daughter would make him look weak. After all, until now he had only had sons.',
        ]
    response_lens = [30, 30, 32, 33]

    print(utils.apply_length_penalty_filter(scores, mt_texts, ref_texts, response_lens, 32))



# test_get_bleu_penalty()
# test_get_length_penalty()
test_apply_length_penalty_filter()