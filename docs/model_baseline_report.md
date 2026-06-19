# Model Baseline Report

Generated at: 2026-04-20 15:20:41

## Summary
- Total Video Files: 272
- Mapped Samples: 250
- Evaluated Samples: 250
- Skipped Unmapped: 22
- Failed Processing: 0
- Top1 Accuracy: 0.6240
- Avg Softmax Confidence: 0.5450
- Avg Top Logit: 6.4758
- Acceptance Rate: 1.0000
- Accepted Precision: 0.6240
- Effective Accuracy: 0.6240
- Runtime Seconds: 1021.3847
- Seconds Per Sample: 4.0855

## Top Confusion Candidates
1. true='after' predicted='arm' count=1
2. true='all' predicted='weus' count=1
3. true='alligator' predicted='clean' count=1
4. true='arm' predicted='open' count=1
5. true='backyard' predicted='up' count=1
6. true='because' predicted='for' count=1
7. true='before' predicted='have' count=1
8. true='beside' predicted='fine' count=1
9. true='better' predicted='arm' count=1
10. true='boat' predicted='book' count=1

## Lowest Accuracy Classes
- thankyou: accuracy=1.0000 support=2 correct=2

## Notes
- Primary source: data/signs_videos
- Ground truth is aligned to the same model label map used by model.tflite
- If video mode and DB mode disagree, prefer video mode for evaluating the model itself
