# Data

This public portfolio repository deliberately excludes the original internship player database.

## Public demo data

`synthetic_players.csv` contains 26 fictitious players created only to demonstrate the expected input structure. It does not represent real athletes.

The bundled `reference/` files are also synthetic demonstration tables:

- `synthetic_hfa_boys_reference.csv` uses the schema `Month, P3, P50, P97`;
- `synthetic_bmi_boys_reference.csv` uses the schema `Month, L, M, S`.

They are **not official WHO or clinical reference data** and must not be used for clinical, medical or player-selection decisions. Their purpose is solely to make the repository reproducible from a clean clone.

## Using an external reference source

The script also accepts compatible CSV or Excel reference tables supplied through command-line arguments. If a third-party reference file is used, verify its methodology, provenance and redistribution terms before publishing it.

## Privacy rule

Never commit the original internship database or any file containing identifiable youth-player information.
