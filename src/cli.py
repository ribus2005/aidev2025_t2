import sys
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

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
    help="The device to host the model.",
    default="cuda",
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

        model = MT5XLSumModel(device=args.device)
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
        summarized = model(text, cutoff_len=args.cutoff)
    except Exception as e:
        log.fatal(f"Error during inference: {e}")
        exit(1)

    print(summarized, end="")


if __name__ == "__main__":
    main()
