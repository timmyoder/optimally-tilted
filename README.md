# Solar Tilt Optimizer ☀️

A physics-based solar panel tilt angle optimizer that uses clear-sky modeling to determine optimal panel angles for maximum energy production.

### Live App: [https://optimally-tilted-solar.streamlit.app/]()


> ⚠️ **Disclaimer**: This whole thing was vibe coded. Don't judge me. While functional and based on solid solar engineering principles, use results as guidance rather than professional engineering analysis. Always consult qualified solar engineers for actual installations.

## Features

- **Dual Implementation**:
  - CLI tool for quick calculations
  - Interactive Streamlit web app with visualizations
- **Physics-Based Optimization**: Uses pvlib-python for accurate solar modeling
  - Hay-Davies transposition model for POA irradiance
  - PVSYST temperature modeling
  - PVWatts DC power calculations
- **Multiple Time Horizons**: Optimizes for 1 day, 14 days, and 30 days
- **Interactive Visualizations**: 
  - PV array side-view diagrams with sun position
  - Monthly optimal tilt trends
  - Annual sun path diagrams with analemmas
  - Custom date range analysis

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.8+

## Usage

### CLI Tool

```bash
python calculate_optimal_tilt.py
```

Edit the bottom of the file to change the target date:
```python
main('2025-09-15')  # Or None for today
```

### Streamlit Web App

```bash
streamlit run tilt_optimizer_app.py
```

Then open your browser to the provided localhost URL (typically `http://localhost:8501`)

## How It Works

The optimizer uses a bounded scalar minimization algorithm to find the tilt angle that maximizes total DC energy production. For each candidate tilt angle:

1. **Clear Sky Weather Generation**: Models hourly irradiance (GHI, DNI, DHI)
2. **POA Transposition**: Converts to plane-of-array irradiance using Hay-Davies model
3. **Temperature Modeling**: Calculates cell temperature based on mounting type and conditions
4. **DC Power Calculation**: Applies PVWatts model with temperature coefficient
5. **Energy Integration**: Sums hourly power to get total kWh over analysis period

The optimization searches tilt angles from 15-55° (physical system constraints) to find the maximum energy output.

## Configuration

Default system parameters (configurable in Streamlit app):
- **Location**: 39.796678°N, 79.092463°W (Meyersdale, PA)
- **System Capacity**: 17.55 kWp
- **Azimuth**: 180° (true south)
- **Albedo**: 0.2 (grass/ground)
- **Temperature Coefficient**: -0.003 (-0.30%/°C)
- **Mounting**: Freestanding

## Key Technologies

- **[pvlib-python](https://pvlib-python.readthedocs.io/)**: Industry-standard PV system modeling
- **[scipy](https://scipy.org/)**: Optimization algorithms
- **[Streamlit](https://streamlit.io/)**: Interactive web interface
- **[Plotly](https://plotly.com/)**: Interactive visualizations

## License

MIT License - See [LICENSE](LICENSE) for details

## Contributing

This is an experimental/educational project. Feel free to fork and experiment!
