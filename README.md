# Forked from egormalyutin
# Summarize a text

Uses [an mT5-based multilingual summarizer model](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum).

## Installation

[Install uv](https://docs.astral.sh/uv/getting-started/installation).

Clone the project:

```bash
git clone https://github.com/ribus2005/aidev2025_t2
cd aidev2025_t2
uv sync
source .venv/bin/activate
```

## Usage
```bash
> uv run summarize --help
usage: summarize [-h] [-c CUTOFF] [-d DEVICE] [--min-length MIN_LENGTH] [--do-sample | --no-do-sample] [--num-beams NUM_BEAMS] [--no-repeat-ngram-size NO_REPEAT_NGRAM_SIZE]
                 [--length-penalty LENGTH_PENALTY] [-l LOG_LEVEL] [-p PARAGRAPHS] [--verbose]
                 TEXT_FILE

Summarize your text using https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum.

positional arguments:
  TEXT_FILE

options:
  -h, --help            show this help message and exit
  -c, --cutoff CUTOFF   The maximum amount of tokens the model can generate. Note that setting this parameter doesn't mean that the model will try to generate a good summary with
                        length less or equal to this number, just that the generation will be stopped at this length. (default: 50)
  -d, --device DEVICE   The device to host the model. (auto means CUDA if it's available, otherwise CPU) (default: auto)
  --min-length MIN_LENGTH
                        The minimum length of the sequence to be generated. (default: 5)
  --do-sample, --no-do-sample
                        Whether or not to use sampling; use greedy decoding otherwise. (default: False)
  --num-beams NUM_BEAMS
                        Number of beams for beam search. 1 means no beam search. (default: 4)
  --no-repeat-ngram-size NO_REPEAT_NGRAM_SIZE
                        If set to int > 0, all ngrams of that size can only occur once. (default: 2)
  --length-penalty LENGTH_PENALTY
                        Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 to encourage shorter sequences, to a value > 1.0 to encourage longer sequences.
                        (default: 5.0)
  -l, --log-level LOG_LEVEL
                        Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL] (default: ERROR)
  -p, --paragraphs PARAGRAPHS
                        paragpraph count (default: 3)
  --verbose             show per paragraph results (default: False)
```

```bash
> uv run summarize riba.txt
  Результат:
  В Киеве пройдет масштабный фестиваль украинской культуры «Спадщина», который состоится в последние выходные месяца. Об этом Би-би-си рассказывает корреспондент BBC News Украина с места
  7% от длины первоначального текста
```