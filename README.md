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
usage: summarize [-h] [-c CUTOFF] [-d DEVICE] [-l LOG_LEVEL] TEXT_FILE

Summarize your text using
https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum.

options:
  -h, --help            show this help message and exit
  -c, --cutoff CUTOFF   The maximum amount of tokens the model can generate.
                        Note that setting this parameter doesn't mean that the
                        model will try to generate a good summary with length
                        less or equal to this number, just that the generation
                        will be stopped at this length. (default: 50)
  -d, --device DEVICE   The device to host the model. (default: cuda)
  -l, --log-level LOG_LEVEL
                        Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                        (default: ERROR)
```

```bash
> uv run summarize riba.txt
  Результат:
  В Киеве пройдет масштабный фестиваль украинской культуры «Спадщина», который состоится в последние выходные месяца. Об этом Би-би-си рассказывает корреспондент BBC News Украина с места
  7% от длины первоначального текста
```