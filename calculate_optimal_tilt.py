import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance, get_extra_radiation
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.pvsystem import pvwatts_dc
from scipy.optimize import minimize_scalar
import datetime


# --- CONFIGURATION ---
LATITUDE = 39.796678
LONGITUDE = -79.092463
TZ = 'America/New_York'
AZIMUTH = 180  # True South
TILT_RANGE = range(15, 56)  # 15 to 55 inclusive
ALBEDO = 0.2
SYSTEM_CAPACITY_W = 17550  # 17.55 kW
TEMP_COEFF = -0.003  # -0.30%/C
MODULE_TYPE = 'glass_glass'  # For thermal model
MOUNTING = 'freestanding'  # For thermal model

# Default environmental conditions for Clear Sky model
DEFAULT_TEMP = 20.0  # Celsius
DEFAULT_WIND = 2.0   # m/s

def generate_clearsky_weather(location, start_date, days=32):
    """
    Generates hourly Clear Sky weather data for a specified period.
    Returns a DataFrame with GHI, DNI, DHI, Ambient Temperature, and Wind Speed.
    """
    # Create hourly timestamps
    # Start from midnight of the start_date
    start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + pd.Timedelta(days=days)
    
    times = pd.date_range(start=start, end=end, freq='h', tz=location.tz)
    
    # Calculate Clear Sky Irradiance
    print(f"Generating Clear Sky data for {days} days starting {start.date()}...")
    cs = location.get_clearsky(times, model='ineichen')
    
    # Create DataFrame with defaults for Temp and Wind
    df = pd.DataFrame({
        'ghi': cs['ghi'],
        'dhi': cs['dhi'],
        'dni': cs['dni'],
        'temp_air': DEFAULT_TEMP,
        'wind_speed': DEFAULT_WIND
    }, index=times)
    
    return df

def calculate_energy(tilt, weather_df, location, solar_pos):
    """
    Calculates total DC energy (kWh) for a given tilt over the weather_df period.
    """
    # 1. Transposition (POA)
    # Let's use Hay-Davies for simplicity and robustness without extra airmass calculations.
    
    dni_extra = get_extra_radiation(weather_df.index)
    
    poa = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=AZIMUTH,
        dni=weather_df['dni'],
        ghi=weather_df['ghi'],
        dhi=weather_df['dhi'],
        solar_zenith=solar_pos['apparent_zenith'],
        solar_azimuth=solar_pos['azimuth'],
        dni_extra=dni_extra,
        model='haydavies', # Using Hay-Davies
        albedo=ALBEDO
    )
    
    # 2. Cell Temperature
    # Get thermal parameters for open rack glass-glass
    params = TEMPERATURE_MODEL_PARAMETERS['pvsyst'][MOUNTING]
    cell_temp = pvlib.temperature.pvsyst_cell(
        poa_global=poa['poa_global'],
        temp_air=weather_df['temp_air'],
        wind_speed=weather_df['wind_speed'],
        u_c=params['u_c'],
        u_v=params['u_v']
    )
    
    # 3. DC Power
    dc_power = pvwatts_dc(
        effective_irradiance=poa['poa_global'],
        temp_cell=cell_temp,
        pdc0=SYSTEM_CAPACITY_W,
        gamma_pdc=TEMP_COEFF
    )
    
    # Energy in kWh (data is hourly)
    energy_kwh = dc_power.sum() / 1000.0
    
    return energy_kwh

def optimize_tilt(weather_df, location, period_label):
    """
    Finds the optimal tilt for the given weather data using scalar minimization.
    """
    # Calculate solar position once for the whole dataframe
    solar_pos = location.get_solarposition(weather_df.index)
    
    # Objective function to minimize (negative energy)
    def objective(tilt):
        return -calculate_energy(tilt, weather_df, location, solar_pos)
    
    # Use bounded optimization
    result = minimize_scalar(
        objective, 
        bounds=(15, 55), 
        method='bounded'
    )
    
    best_tilt = result.x
    max_energy = -result.fun
            
    return best_tilt, max_energy

def main(target_date_str=None):
    print(f"--- Solar Tilt Optimization (Clear Sky) ---")
    print(f"Location: {LATITUDE}, {LONGITUDE}")
    print(f"System: {SYSTEM_CAPACITY_W/1000} kWp Array")
    print(f"Bifacial Albedo: {ALBEDO}")
    print("-" * 30)

    # Setup Location
    site = Location(LATITUDE, LONGITUDE, tz=TZ, name='Meyersdale_PV')

    # Determine Start Date
    if target_date_str:
        try:
            # Parse user date
            base_date = pd.Timestamp(target_date_str).tz_localize(TZ)
        except ValueError:
            print(f"Error: Could not parse date '{target_date_str}'. Using today.")
            base_date = pd.Timestamp.now(tz=TZ)
    else:
        base_date = pd.Timestamp.now(tz=TZ)

    # We want to analyze starting from the "next day" relative to the base date
    # e.g. if I say "2025-06-01", I want optimization for June 2nd, June 2-15, June 2-30
    start_analysis = base_date + pd.Timedelta(days=1)
    start_analysis = start_analysis.replace(hour=0, minute=0, second=0, microsecond=0)
    
    print(f"Base Date: {base_date.date()}")
    print(f"Analysis Start: {start_analysis.date()}")

    # Generate Weather Data (32 days to cover all periods)
    weather_df = generate_clearsky_weather(site, start_analysis, days=32)
    
    # Period 1: Day 1 (Tomorrow)
    end_day1 = start_analysis + pd.Timedelta(hours=23, minutes=59)
    df_day1 = weather_df.loc[start_analysis:end_day1]
    
    opt_tilt, energy = optimize_tilt(df_day1, site, "Day 1")
    print(f"\n✅ Day 1 ({start_analysis.date()}):")
    print(f"   Optimal Tilt: {opt_tilt:.2f}°")
    print(f"   Est. DC Production (Clear Sky): {energy:.2f} kWh")

    # Period 2: Next 14 Days
    end_14d = start_analysis + pd.Timedelta(days=14) - pd.Timedelta(minutes=1)
    df_14d = weather_df.loc[start_analysis:end_14d]
    
    opt_tilt, energy = optimize_tilt(df_14d, site, "Next 14 Days")
    print(f"\n✅ Next 14 Days ({start_analysis.date()} to {end_14d.date()}):")
    print(f"   Optimal Tilt: {opt_tilt:.2f}°")
    print(f"   Est. DC Production (Clear Sky): {energy:.2f} kWh")

    # Period 3: Next 30 Days
    end_30d = start_analysis + pd.Timedelta(days=30) - pd.Timedelta(minutes=1)
    df_30d = weather_df.loc[start_analysis:end_30d]
    
    opt_tilt, energy = optimize_tilt(df_30d, site, "Next 30 Days")
    print(f"\n✅ Next 30 Days ({start_analysis.date()} to {end_30d.date()}):")
    print(f"   Optimal Tilt: {opt_tilt:.2f}°")
    print(f"   Est. DC Production (Clear Sky): {energy:.2f} kWh")

if __name__ == "__main__":
    # Example usage: Pass a date string YYYY-MM-DD or None for today
    # main('2025-06-20') 
    main('2025-09-15')
