_mii_model = None
_radgraph_model = None
_cache = None


def _init_models():
    global _mii_model, _radgraph_model, _cache
    if _mii_model is None:
        from medimageinsightmodel import MedImageInsight
        _mii_model = MedImageInsight(
            model_dir="/2024.09.27",
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth",
            devices="cuda"
        )
        _mii_model.load_model()
    if _radgraph_model is None:
        from radgraph import RadGraph
        _radgraph_model = RadGraph(model_type="radgraph-xl", device="cuda")
    if _cache is None:
        class SimpleCache:
            def __init__(self):
                self.image_emb = {}
                self.text_emb = {}

        _cache = SimpleCache()


def _get_image_embedding(image_info, model):
    import base64
    import hashlib
    key = hashlib.md5(image_info['path'].encode()).hexdigest() if 'path' in image_info else hashlib.md5(
        image_info['bytes']).hexdigest()
    if key not in _cache.image_emb:
        if image_info.get('bytes') is None:
            with open(image_info['path'], "rb") as f:
                image_info['bytes'] = base64.encodebytes(f.read()).decode("utf-8")
        _cache.image_emb[key] = model.encode(images=[image_info['bytes']])['image_embeddings'][0]
    return _cache.image_emb[key]


def _get_text_embedding(text, model):
    import hashlib
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _cache.text_emb:
        _cache.text_emb[key] = model.encode(texts=[text])['text_embeddings'][0]
    return _cache.text_emb[key]


def _check_reflection_needed(report, current_image_emb, mii_model, radgraph_model, threshold=0.3):
    from sklearn.metrics.pairwise import cosine_similarity

    report_emb = _get_text_embedding(report, mii_model)
    similarity = cosine_similarity([current_image_emb], [report_emb])[0][0]

    reflection_type = None
    if report and len(report.strip()) >= 10:
        radgraph_results = radgraph_model.predict([report])[0]
        entities = radgraph_results.get("entities", {})
        reflection_types = []
        for entity_id, entity_info in entities.items():
            if entity_info.get("label") in ["ANAT_DER", "OBS_DER"]:
                reflection_types.append("Anatomical Entity Error")
            if entity_info.get("label") == "OBSERVATION" and "located_at" not in entity_info.get("relations", {}):
                reflection_types.append("Diagnostic Relationship Deviation")
            if entity_info.get("label") == "OBSERVATION" and len(entity_info.get("modifiers", [])) == 0:
                reflection_types.append("Descriptive Information Omission")
        if reflection_types:
            reflection_type = "; ".join(list(set(reflection_types)))

    need_reflection = similarity < threshold or reflection_type is not None
    return need_reflection, similarity, reflection_type


def check_medical_report_and_reflect(inputs):
    _init_models()

    reflection_prompt_template = "{Reflection_Type} Please verify description accuracy (score: {similarity:.2f}) {patient_report}."
    similarity_threshold = 0.3

    current_images = [input['images'][0] for input in inputs]
    current_image_embs = [_get_image_embedding(img, _mii_model) for img in current_images]

    contents = [input['messages'][-1]['content'] for input in inputs]

    for idx, (input, current_image_emb) in enumerate(zip(inputs, current_image_embs)):
        content = contents[idx]
        need_reflection, similarity, reflection_type = _check_reflection_needed(
            content, current_image_emb, _mii_model, _radgraph_model, similarity_threshold
        )

        reflection_prompt = reflection_prompt_template.format(
            Reflection_Type=reflection_type if reflection_type else "Semantic Consistency Check",
            similarity=similarity,
            patient_report=content
        )

        if need_reflection and reflection_prompt not in content:
            content += "\n\n" + reflection_prompt
            input['messages'][-1]['content'] = content
            input['finished'] = False
        else:
            input['finished'] = True

    return inputs


def check_medical_report_and_reflect_multi_turn(inputs):
    _init_models()
    reflection_prompt_template = "{Reflection_Type} Please verify description accuracy (score: {similarity:.2f})"
    similarity_threshold = 0.3

    current_images = [input['images'][0] for input in inputs]
    current_image_embs = [_get_image_embedding(img, _mii_model) for img in current_images]

    def _find_last_assistant_content(messages):
        for msg in reversed(messages):
            if msg['role'] == 'assistant':
                return msg['content']
        return ""

    contents = [_find_last_assistant_content(input['messages']) for input in inputs]

    for idx, (input, current_image_emb) in enumerate(zip(inputs, current_image_embs)):
        content = contents[idx]
        need_reflection, similarity, reflection_type = _check_reflection_needed(
            content, current_image_emb, _mii_model, _radgraph_model, similarity_threshold
        )

        reflection_prompt = reflection_prompt_template.format(
            Reflection_Type=reflection_type if reflection_type else "Semantic Consistency Check",
            similarity=similarity
        )

        if need_reflection and reflection_prompt not in [msg['content'] for msg in input['messages']]:
            input['messages'].append({'role': 'user', 'content': reflection_prompt})
            input['finished'] = False
        else:
            input['finished'] = True

    return inputs



def check_math_result_and_give_tips(inputs):
    from .orm import MathAccuracy
    acc = MathAccuracy()
    # a trick
    prompt = 'But wait... It seems I made a mistake,'
    contents = [input['messages'][-1]['content'] for input in inputs]
    rewards = acc(contents, [input['solution'] for input in inputs])
    for reward, input in zip(rewards, inputs):
        content = input['messages'][-1]['content']
        if reward < 1 and prompt not in content:
            if '<answer>' in content:
                content = content[:content.index('<answer>')]
            if '</think>' in content:
                content = content[:content.index('</think>')]
            content += prompt
            input['messages'][-1]['content'] = content
            input['finished'] = False
        else:
            input['finished'] = True
    return inputs


def check_math_result_and_give_tips_multi_turn(inputs):
    from .orm import MathAccuracy
    acc = MathAccuracy()
    prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'
    contents = [input['messages'][-1]['content'] for input in inputs]
    rewards = acc(contents, [input['solution'] for input in inputs])
    for reward, input in zip(rewards, inputs):
        content = input['messages'][-2]['content']
        if reward < 1 and prompt not in content:
            input['messages'].append({'role': 'user', 'content': prompt})
            input['finished'] = False
        else:
            input['finished'] = True
    return inputs


multi_turns = {
    # 'math_tip_trick': check_math_result_and_give_tips,
    # 'math_tip_trick_multi_turn': check_math_result_and_give_tips_multi_turn,
    'medical_reflect': check_medical_report_and_reflect,
    'medical_reflect_multi_turn': check_medical_report_and_reflect_multi_turn,
}
