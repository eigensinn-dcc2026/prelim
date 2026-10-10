# M3 offline candidate

This is a packaging template, not an adopted model. `mission3.pt` is added only
after three-seed internal qualification and official Macro F1 >= 0.6540.

Run offline:
```
python inference.py --audio_dir AUDIO --label_dir LABELS --ckpt_path mission3.pt --output outputs/mission3.csv
```
Only `utterances[].text` is read. Nine labels keep their original order and
threshold 0.5. The checkpoint contains model weights, tokenizer, module settings
and provenance; no model download is required. At most six 256-token windows are
used, so very long calls may have gaps between sampled windows.
