# MonkeyOCRv2: layout-aware local integration

Audit date: 2026-07-31.

## Pinned sources

- Model: `zenosai/MonkeyOCRv2-B-Parsing`, immutable revision
  `de7a993bd0f39a97b122dac767e82ae04935bce4`.
- Official repository: <https://github.com/Yuliang-Liu/MonkeyOCRv2>, `main`
  as inspected on 2026-07-31 (33 commits were shown by GitHub). The integration
  follows `parsing/core_runner.py` from that snapshot. The upstream repository
  had no release tag for this parsing snapshot, so the access date is recorded
  rather than inventing a commit hash.
- Public Space: <https://huggingface.co/spaces/zenosai/monkey-ocr-v2>, observed
  revision `b6af3bd` in the Space history on 2026-07-31.
- Model card: <https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing>.

Local dependency pins are `torch==2.6.0`, `torchvision==0.21.0`,
`transformers==4.57.6`, `accelerate==1.10.1`, and `einops==0.8.1`.

## Audit: old implementation versus official pipeline

The removed implementation called the official `END2END` prompt once for a
whole page. Before that call it resized every input exceeding 1,003,520 pixels;
a 2480x3508 page therefore became approximately 843x1192. It then fabricated a
uniform block confidence of 0.86 and tried to recover truncated/empty fields by
rerunning the lower part of the page.

The official non-END2END pipeline is two-stage:

1. `LAYOUT`: `Please output the categories and coordinates of the document
   elements in reading order.` The returned coordinates use the 0..1000 space.
2. Each resulting bbox is mapped to page pixels, cropped, and recognized with a
   task prompt. Text-like labels use `Please output the text content from the
   image.`; tables use the OTSL prompt; formulas use the LaTeX prompt.
3. Official recognition uses one crop per request, `max_tokens=5000`, and
   converts OTSL to HTML while preserving reading order.
4. `Picture` does not undergo text recognition. It is retained as a visual zone.

## Local implementation

`MonkeyOCRParsingEngine` implements the same stage split. Layout detection sees
the complete page through a memory-safe aspect-preserving preview; that preview
is used only to obtain normalized coordinates. Every content crop is taken from the original rendered page,
never from the model's layout preview. The approximately one-megapixel limit is
`recognition_max_pixels` and applies only to an individual crop; smaller crops
are not enlarged or reduced.

On the local 4 GiB Transformers runtime, layout decoding stops immediately
after a complete balanced layout list has been generated. The configurable
`layout_max_seconds` limit fails closed when a response never produces a valid
closing list. It does not accept a partial layout and does not affect content
crop resolution.

The checkpoint's stored generation configuration has `use_cache=false`.
Running that setting through plain Transformers recalculates the complete
visual prefix for every output token. The local integration explicitly enables
`transformers_use_kv_cache`; for the 39-element spec5 layout (1314 input and
approximately 1423 output tokens) the calculated BF16 KV cache is about 299
MiB. The setting remains configurable, and the structured/time limits remain
fail-closed.

Content generation also has an early repeated-tail stop and a configurable
`element_max_seconds` limit. Repetitive or timed-out output is retained only as
diagnostic `raw_response`, is excluded from normalized OCR text, and is not
stored in the crop cache. A doubled retry is allowed only for a genuinely
length-truncated Table or Formula, never for repetition, timeouts, or ordinary
Text blocks.

The output stores the normalized and original pixel bbox, reading order, label,
raw response, native OTSL/HTML, native Markdown, crop SHA-256, finish reason and
cache status. Because MonkeyOCR does not publish calibrated block probabilities,
`model_confidence` is `null`; business extractors calculate evidence confidence.

An unchanged crop is reused from `runtime/cache/monkey_elements`. The document
OCR cache format is version 3, making old single-pass entries incompatible.

## Windows / 4 GiB differences

The official preferred runtime is vLLM. Native Windows and a 4 GiB RTX 3050 do
not provide a practical supported vLLM deployment, so the local default executes
the same prompts and stage order sequentially through Transformers, with
`batch_size=1`, BF16 weights, SDPA, `torch.inference_mode()` and CUDA cleanup
between elements. No quantization, checkpoint substitution or model pruning is
used. An official OpenAI-compatible vLLM server remains usable with
`--monkey-runtime http --monkey-server-url ...`.

## Debug and comparison

With `--debug-level full`, the `monkey/` report directory contains page layout,
original-resolution crops, raw recognition responses, native Markdown/tables,
the neutral OCR document and timings. `compare-monkey` compares a saved local
result with a JSON/Markdown web result and never starts the model.
