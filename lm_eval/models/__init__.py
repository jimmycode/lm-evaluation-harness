from . import hf
from . import litegpt3
from . import gpt3
from . import textsynth
from . import dummy

MODEL_REGISTRY = {
    "hf": hf.HFLM,
    "litegpt3": litegpt3.liteGPT3LM,
    "gpt3": gpt3.GPT3LM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
