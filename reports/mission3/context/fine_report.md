# M3 context experiment

Status: **fine_pilots_failed**

Threshold >= 0.5. Training-only retrieval. Official Validation excluded from fitting.

| Model | Macro F1 | Micro F1 |
|---|---:|---:|
| roberta003 | 0.6441 | 0.6568 |
| existing_ensemble | 0.6539 | 0.6665 |
| control_frozen | 0.6434 | 0.6568 |
| memory_frozen | 0.6426 | 0.6560 |
| evidence_frozen | 0.6428 | 0.6552 |
| control_fine | 0.6406 | 0.6537 |
| memory_fine | 0.6408 | 0.6541 |
| evidence_fine | 0.6405 | 0.6537 |
