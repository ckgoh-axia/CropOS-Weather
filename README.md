# CropOS

Hyper-local 24–48 hour precipitation forecast for Thai smallholder farms.

CropOS fuses ERA5-Land grid data, GPM IMERG precipitation observations, and METAR airport station observations through a heterogeneous GNN (LocalizedWeatherGNN) to produce farm-coordinate-specific rain probability forecasts — outperforming TMD district forecasts on convective micro-rain events.

## Quick Start

```bash
poetry install
python scripts/download_data.py  # 2-4 hours on first run
pytest tests/unit/
```

## Architecture

ERA5-Land (11km grid) + METAR airports (16 Thai stations) + Farm GPS → HeteroSAGE GNN → Rain probability [12h, 24h, 36h, 48h]

Alert rule: `if rain_probability_24h >= 0.5: send LINE alert`
