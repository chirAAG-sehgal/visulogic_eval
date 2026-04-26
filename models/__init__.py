def load_model(args):
    if args.model_path in ["doubao-vision-pro-32k","doubao-vision-lite-32k","doubao-1.5-vision-pro-32k"]:
        from models.doubao_api import DoubaoAPIModel
        model = DoubaoAPIModel(model_name = args.model_path,
                            api_key = args.api_key,
                            user_prompt = args.user_prompt)
    elif args.model_path in ['gpt-4o','gpt-4o-mini']:
        from models.openai_api import OpenAIAPIModel
        model = OpenAIAPIModel(model_name = args.model_path,
                            api_key = args.api_key,
                            user_prompt = args.user_prompt)
    elif args.model_path in ['kimi-latest','moonshot-v1-8k-vision-preview','moonshot-v1-32k-vision-preview','moonshot-v1-128k-vision-preview']:
        from models.kimi_api import MoonshotAPIModel
        model = MoonshotAPIModel(model_name = args.model_path,
                            api_key = args.api_key,
                            user_prompt = args.user_prompt)
    elif 'qwen3' in args.model_path.lower() and 'vl' in args.model_path.lower():
        from models.qwen3vl_model import Qwen3VisionModel
        model = Qwen3VisionModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'qwen' in args.model_path.lower() and 'vl' in args.model_path.lower():
        from models.qwenvl_model import QwenVisionModel
        model = QwenVisionModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'internvl' in args.model_path.lower():
        from models.internvl_model import InternVLModel
        model = InternVLModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'llava' in args.model_path.lower() and not 'onevision' in args.model_path.lower():
        from models.llava_model import LlavaModel
        model = LlavaModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'llava' in args.model_path.lower() and 'onevision' in args.model_path.lower():
        from models.llavaonevision import LlavaOnevisionModel
        model = LlavaOnevisionModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'minicpm-o' in args.model_path.lower():
        from models.minicpm_o_model import MiniCPMOModel
        model = MiniCPMOModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'mplug' in args.model_path.lower():
        from models.mplug_model import mPLUGModel
        model = mPLUGModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'ovis' in args.model_path.lower():
        from models.ovis2_model import Ovis2Model
        model = Ovis2Model(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'glm' in args.model_path.lower():
        from models.glm4v_model import GLM4VModel
        model = GLM4VModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    elif 'sharegpt' in args.model_path.lower():
        from models.sharegpt4_model import ShareGPT4VModel
        model = ShareGPT4VModel(model_path = args.model_path,
                                user_prompt = args.user_prompt)
    return model
