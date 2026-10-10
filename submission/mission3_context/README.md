# M3 context candidate package

This directory is a packaging template, not an accepted trained model.
The packaging command supplies `mission3.pt`, embedded tokenizer/configuration,
`mission3_bert.py`, `mission3_context.py`, the official decision, and a SHA256 freeze manifest.
Method A additionally embeds Training vectors, labels, and exclusion identities in the checkpoint.

```bash
python inference.py --audio_dir /path/to/wav --label_dir /path/to/json \
  --ckpt_path mission3.pt --output outputs/mission3.csv
```

Reads only `utterances[].text`; speaker, patient metadata, symptom annotations,
and audio are not model inputs. Identity is used only to exclude retrieval matches.
Threshold is always `>=0.5`. Output is `label file name,symptom`; empty `[]` is allowed.
No network or external model download is needed during inference.
The existing `submission/mission3` candidate is retained until the final score is at least 0.6540.
