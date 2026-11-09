# Data

## `article_index.csv` (included)

One row per article in the final dataset (4,535 rows). No article text is included.

| Column | Description |
|---|---|
| `link` | Original Al Jazeera URL |
| `publication_date` | Publication date (from the URL) |
| `section` | Class label = Al Jazeera section: `sport`, `economy` (`/ebusiness/`), `health`, `arts`, `family` |
| `split` | `development` (before 26 Aug 2025) or `test` (on/after 26 Aug 2025) |
| `group_id` | Confirmed related-article group (`G1`–`G3`), empty otherwise |
| `fold` | CV fold 0–4 for development articles; `-1` for test articles |
| `title_is_topic_tag` | `True` for 19 articles where the scraper captured a topic tag instead of the headline |

## Raw files (not included)

The notebooks expect these files in `data/raw/`:

- `aljazeera_after_duplicate_cleaning.csv` (5,505 articles after deduplication)
- `aljazeera_final_balanced_4535.csv` (final balanced dataset with title and body)
- `label_audit_final_results.csv` (250 audited articles with reviewer labels)

They can be rebuilt with `notebooks/00_data_collection.ipynb`, or by downloading the URLs in `article_index.csv`.
