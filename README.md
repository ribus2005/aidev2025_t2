# Summarize a text

Uses [an mT5-based multilingual summarizer model](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum).

## Installation

[Install uv](https://docs.astral.sh/uv/getting-started/installation).

Clone the project:

```bash
git clone https://github.com/egormalyutin/aidev2025_t1
cd aidev2025_t1
uv sync
source .venv/bin/activate
```

## Usage
```bash
> uv run summarize --help
usage: summarize [-h] [-c CUTOFF] [-d DEVICE] [-l LOG_LEVEL]

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
> uv run summarize <<EOF
МОСКВА, 18 сен — РИА Новости. В Санкт-Петербурге задержали агентов украинских спецслужб, готовивших убийство главы одного из оборонных предприятий, сообщила ФСБ.
"Федеральной службой безопасности <..> пресечена деятельность агентурной сети украинских спецслужб, состоявшей из трех граждан России 1993, 1994 и 2006 годов рождения, причастных к подготовке террористического акта в отношении руководителя одного из предприятий оборонно-промышленного комплекса путем подрыва его автомобиля с применением самодельного взрывного устройства", — говорится в релизе.
Главное об операции:
- злоумышленники — юноша и две девушки — действовали по указке куратора из террористической организации, курируемой Главным управлением разведки Минобороны Украины;
- двое задержанных следили за предполагаемой жертвой с помощью камер, установленных на велосипеде, они провели разведку по месту его жительства;
- исполнитель забрал взрывное устройство из тайника, оборудованного на Волковском кладбище;
- перед минированием автомобиля главы оборонного предприятия он переоделся в женскую одежду, чтобы пустить следствие по ложному пути, но был задержан в момент закладки бомбы;
- Киев обещал исполнителю 25 тысяч рублей на покупку женской одежды;
- одна из задержанных рассказала, что они должны были экстренно бежать на Украину после срыва теракта.
EOF
В Санкт-Петербурге задержали агентов украинских спецслужб, готовивших убийство главы одного из оборонных предприятий, сообщила ФСБ России.
```