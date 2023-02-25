import transformers
import torch
from lm_eval.base import BaseLM
from litegpt3.models.gpt_neox_ps_lora.modeling_gpt_neox import GPTNeoXPSLoRAForCausalLM
from litegpt3.models.gpt_neox_ps_lora.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from litegpt3.models.gpt2_ps.modeling_gpt2_ps import GPT2PSLMHeadModel
from litegpt3.models.gpt2_ps.tokenization_gpt2_ps_fast import GPT2PSTokenizerFast

LITEGPT3_MODELS = {
    "gpt_neox_ps_lora": GPTNeoXPSLoRAForCausalLM,
    "gpt2_ps": GPT2PSLMHeadModel,
}

LITEGPT3_TOKENIZERS = {
    "gpt_neox_ps_lora": GPTNeoXTokenizerFast,
    "gpt2_ps": GPT2PSTokenizerFast,
}


class liteGPT3LM(BaseLM):
    def __init__(
            self,
            model_type="gpt_neox_ps_lora",
            checkpoint_dir=None,
            tokenizer_path=None,
            dtype="bf16",
            device="cuda",
            batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        assert model_type in LITEGPT3_MODELS.keys(), f"model_type must be one of {LITEGPT3_MODELS.keys()}"
        self.litegpt3_lm = LITEGPT3_MODELS[model_type].from_pretrained(checkpoint_dir)

        if dtype == "bf16":
            self.litegpt3_lm.bfloat16()
        elif dtype == "fp16":
            self.litegpt3_lm.half()
        elif dtype == "fp32":
            self.litegpt3_lm.float()
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        self.litegpt3_lm.to(self.device)
        self.litegpt3_lm.eval()

        self.tokenizer = LITEGPT3_TOKENIZERS[model_type].from_pretrained(tokenizer_path)

        # assert isinstance(
        #     self.tokenizer,
        #     (
        #         transformers.GPT2Tokenizer,
        #         transformers.GPT2TokenizerFast,
        #         transformers.T5Tokenizer,
        #         transformers.T5TokenizerFast,
        #     ),
        # ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        # if isinstance(
        #         self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        # ):
        #     assert self.tokenizer.encode("hello\n\nhello") == [
        #         31373,
        #         198,
        #         198,
        #         31373,
        #     ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.litegpt3_lm.config.n_ctx
        except AttributeError:
            # GPTNeo(X)Config doesn't have n_ctx apparently
            return self.litegpt3_lm.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.litegpt3_lm(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.litegpt3_lm.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
