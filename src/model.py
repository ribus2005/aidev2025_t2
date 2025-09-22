import re
import torch
from transformers import AutoTokenizer, pipeline


class MT5XLSumModel:
    """
    An mT5-based multilingual summarizer model.
    https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum
    """

    def __init__(self, *, device: str = None):
        """
        Initialize and load the model.

        Parameters
        ----------
        device: str
            The device to host the model.
        """

        name = "csebuetnlp/mT5_multilingual_XLSum"

        tokenizer = AutoTokenizer.from_pretrained(
            name,
            use_fast=False,
            legacy=False,
        )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.model = pipeline(
            "summarization",
            model=name,
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def preprocess(text: str) -> str:
        """
        Remove newlines and extra tabulation from the input.

        Parameters
        ----------
        text: str
            Text.

        Returns
        -------
        str
            Text with newlines and extra tabulation removed.
        """

        text = text.strip()
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        if len(text) == 0:
            raise ValueError("text is empty after preprocessing")

        return text

    def __call__(
        self,
        text: str,
        *,
        cutoff_len: int = 50,
        min_len: int = 5,
        do_sample: bool = False,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 2,
        length_penalty: float = 1.0,
    ) -> str:
        """
        Run the model.

        Parameters
        ----------
        text: str
            Preprocessed text.

        cutoff_len: int
            The maximum amount of tokens the model can generate.
            Note that setting this parameter doesn't mean that the model will
            try to generate a good summary with length less or equal to this
            number, just that the generation will be stopped at this length.

        min_len: int
            The minimum length of the sequence to be generated.

        do_sample: bool
            Whether or not to use sampling; use greedy decoding otherwise.

        num_beams: int
            Number of beams for beam search. 1 means no beam search.

        no_repeat_ngram_size: int
            If set to int > 0, all ngrams of that size can only occur once.

        length_penalty: float
            Exponential penalty to the length. 1.0 means no penalty.
            Set to values < 1.0 in order to encourage the model to generate shorter sequences,
            to a value > 1.0 in order to encourage the model to produce longer sequences.

        Returns
        -------
        str
            The model output.
        """

        assert cutoff_len > 0, "cutoff must be positive"
        assert min_len > 0, "min_len must be positive"
        assert num_beams > 0, "num_beams must be positive"
        assert no_repeat_ngram_size >= 0, "no_repeat_ngram_size must be non-negative"
        assert length_penalty > 0, "length_penalty must be positive"

        out = self.model(
            text,
            max_new_tokens=cutoff_len,
            min_length=min_len,
            do_sample=do_sample,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
        )

        return out[0]["summary_text"].strip()
