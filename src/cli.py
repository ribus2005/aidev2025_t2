import sys
import logging
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)

parser = ArgumentParser(
    description="Summarize your text using https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum.",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-c",
    "--cutoff",
    type=int,
    help="The maximum amount of tokens the model can generate. Note that setting this parameter doesn't mean that the model will try to generate a good summary with length less or equal to this number, just that the generation will be stopped at this length.",
    default=50,
)

parser.add_argument(
    "-d",
    "--device",
    type=str,
    help="The device to host the model. (auto means CUDA if it's available, otherwise CPU)",
    default="auto",
)

parser.add_argument(
    "--min-length",
    type=int,
    help="The minimum length of the sequence to be generated.",
    default=5,
)

parser.add_argument(
    "--do-sample",
    action=BooleanOptionalAction,
    help="Whether or not to use sampling; use greedy decoding otherwise.",
    default=False,
)

parser.add_argument(
    "--num-beams",
    type=int,
    help="Number of beams for beam search. 1 means no beam search.",
    default=4,
)

parser.add_argument(
    "--no-repeat-ngram-size",
    type=int,
    help="If set to int > 0, all ngrams of that size can only occur once.",
    default=2,
)

parser.add_argument(
    "--length-penalty",
    type=float,
    help="Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 to encourage shorter sequences, to a value > 1.0 to encourage longer sequences.",
    default=1.0,
)

parser.add_argument(
    "-l",
    "--log-level",
    type=str,
    help="Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]",
    default="ERROR",
)


def setup_logging(level: str):
    """
    Configure Hugging Face logger and standard library logger to the selected level.
    """

    from transformers.utils import logging as tf_logging

    tf_logging.disable_default_handler()
    tf_logging.enable_propagation()

    logging.basicConfig(level=level)
    tf_logging.set_verbosity(level)


def main():
    args = parser.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    if sys.stdin.isatty():
        print("Enter your text and then press Ctrl+D:", file=sys.stderr)
    else:
        log.info("reading the text")

    text = sys.stdin.read()

    log.info("loading model")

    try:
        from model import MT5XLSumModel

        if args.device == "auto":
            device = None
        else:
            device = args.device

        model = MT5XLSumModel(device=device)

    except Exception as e:
        log.fatal(f"Error while loading the model: {e}")
        exit(1)

    log.info("preprocessing")

    try:
        text = MT5XLSumModel.preprocess(text)
    except Exception as e:
        log.fatal(f"Error during preprocessing: {e}")
        exit(1)

    log.info("summarizing")

    try:
        summarized = model(
            text,
            cutoff_len=args.cutoff,
            min_len=args.min_length,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            length_penalty=args.length_penalty,
        )
    except Exception as e:
        log.fatal(f"Error during inference: {e}")
        exit(1)

    print(summarized, end="")


if __name__ == "__main__":
    main()
