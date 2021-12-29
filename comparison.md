# Comparision

| Normalisation | Pipeline        | Matcher v1       | Matcher v2           |
| ------------- | --------------- | ---------------- | -------------------- |
| Yes           | Phrase Matching | 340 µs ± 16.1 µs | **282 µs ± 3 µs**    |
| Yes           | RegEx           | 488 µs ± 86.6 µs | **375 µs ± 19.2 µs** |
| No            | Phrase Matching |                  |                      |
| No            | RegEx           |                  |                      |
