# Verified Prop Engine (VPE)

Statistical player-prop engine run from Claude chats (skills `verified-prop-engine` and `verified-prop-engine-nba`).

- `vpe_nfl.py` — NFL engine (nflverse data)
- `vpe_nba.py` — NBA engine (sportsdataverse ESPN box scores)
- `SHA256SUMS` — checksums; each skill pins a tag and verifies these before running.

Win chances = learned mix of model and no-vig market (Calib tab of the Pick Log). No API keys, no personal data in this repo.
