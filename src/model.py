import re
from transformers import AutoTokenizer, pipeline


class MT5XLSumModel:
    """
    An mT5-based multilingual summarizer model.
    https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum
    """

    def __init__(self, *, device: str = "cuda"):
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

    def __call__(self, text: str, *, cutoff_len: int = 50) -> str:
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

        Returns
        -------
        str
            The model output.
        """

        out = self.model(
            text,
            max_new_tokens=cutoff_len,
            min_length=5,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
        )

        return out[0]["summary_text"].strip()
