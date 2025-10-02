import sys
import logging
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
    FileType,
    ArgumentTypeError,
)


def uint(value):
    '''
    целочисленный тип данных для количества абзацев в тексте
    '''
    ivalue = int(value)
    if ivalue <= 0:
        raise ArgumentTypeError(f"{value} не положительное число")
    return ivalue

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
    default=5.0,
)

parser.add_argument(
    "-l",
    "--log-level",
    type=str,
    help="Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]",
    default="ERROR",
)

parser.add_argument(
    "-p",
    "--paragraphs",
    type = uint,
    help = "paragpraph count",
    default= 3,
)

parser.add_argument(
    "--verbose",
    action = "store_true",
    help = "show per paragraph results",
    default = False,
)

parser.add_argument('TEXT_FILE', type=FileType('r'))


def setup_logging(level: str):
    """
    Configure Hugging Face logger and standard library logger to the selected level.
    """

    from transformers.utils import logging as tf_logging

    tf_logging.disable_default_handler()
    tf_logging.enable_propagation()

    logging.basicConfig(level=level)
    tf_logging.set_verbosity(level)


def split_text(text, amount=3, separator='\n\n'):
    '''
    разбивает текстовый файл на список абзацев
    '''
    try:
        content = text
        paragraphs = [p.strip() for p in content.split(separator) if p.strip()]
        if len(paragraphs) != amount:
            raise ValueError(f"Количество абзацев в файле {len(paragraphs)} не совпадает с целевым {amount}")
        return paragraphs
    except Exception as e:
        raise ValueError(f"Ошибка в разделении на абзацы {e}")
    

def load_model(device):
    '''
    загружает модель на устройство 
    '''
    try:
        from model import MT5XLSumModel

        if device == "auto":
            device = None
        else:
            device = device

        model = MT5XLSumModel(device=device)

    except Exception as e:
        raise ValueError(f"ошибка при загрузке модели {e}")
    
    return model

def count_words(text):
    '''
    считает количество слов в тексте 
    '''
    words = text.split()
    return len(words)
        

def deep_summarize(model, paragraphs, model_args={}, verbose=False):
    '''
    прогоняет через модель каждый абзац, а потом сумму результатов
    '''
    summarized = ''
    for paragraph in paragraphs:
        try:
            prep = model.preprocess(paragraph)
        except Exception as e:
            raise ValueError(f"Error during preprocessing: {e}")
            

        try:
            output = model(prep, **model_args)
        except Exception as e:
            raise ValueError(f"Error during inference: {e}")
        if verbose:
            print(20*'=')
            print(output)
            print(20*'=')
        summarized += output + "\n"

    try:
        result = model(summarized, **model_args)
    except Exception as e:
        raise ValueError(f"Error during final inference: {e}")
    
    
    return result, count_words(summarized)



def main():
    args = parser.parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    

    model_args = {
                   'cutoff_len': args.cutoff,
                   'min_len': args.min_length,
                   "do_sample": args.do_sample, 
                   'num_beams': args.num_beams, 
                   'no_repeat_ngram_size': args.no_repeat_ngram_size,
                   'length_penalty': args.length_penalty
                   }

    try:
        input_text = args.TEXT_FILE.read()
        log.info('loading model')
        model = load_model(args.device)

        log.info("reading the text")
        paragraphs = split_text(input_text, args.paragraphs)

        log.info('processing text')
        summarized, medium_len = deep_summarize(model, paragraphs, model_args, args.verbose)

        log.info('finished')
        input_len = count_words(input_text)
        output_len = count_words(summarized)
    except Exception as e:
        log.fatal(e)
        raise ValueError(e)    

    print("Результат:")
    print(summarized)
    print(f'{100*(output_len/input_len):.0f}% от длины первоначального текста')
    print(f'{100*(output_len/medium_len):.0f}% от суммарной длины сокращений абзацев')

if __name__ == "__main__":
    main()