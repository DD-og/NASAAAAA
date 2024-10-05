import streamlit as st
import math
from datetime import datetime, timezone
from skyfield.api import load
from skyfield.data import mpc
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN

# Constants for magnitude calculations
STANDARD_MAGNITUDES = {
    'mercury': -0.36, 'venus': -4.40, 'mars': -1.52,
    'jupiter': -9.40, 'saturn': -8.88, 'uranus': -7.19, 'neptune': -6.87
}

# Load ephemeris once at module level
eph = load('de421.bsp')

def calculate_magnitude(body_name, distance, phase_angle):
    if body_name == 'moon':
        return -12.7 + 0.026 * abs(phase_angle) + 4e-9 * phase_angle**4
    elif body_name in STANDARD_MAGNITUDES:
        std_mag = STANDARD_MAGNITUDES[body_name]
        return std_mag + 5 * math.log10(max(distance * (distance - 1), 1e-10))
    else:
        return None

def calculate_planetary_positions(selected_datetime=None):
    try:
        earth = eph['earth']
        sun = eph['sun']

        bodies = {
            'mercury': 'mercury barycenter',
            'venus': 'venus barycenter',
            'earth': 'earth barycenter',
            'mars': 'mars barycenter',
            'jupiter': 'jupiter barycenter',
            'saturn': 'saturn barycenter',
            'uranus': 'uranus barycenter',
            'neptune': 'neptune barycenter',
        }

        ts = load.timescale()
        t = ts.from_datetime(selected_datetime) if selected_datetime else ts.now()

        planet_data = {}
        for body_name, body_id in bodies.items():
            body = eph[body_id]
            astrometric = earth.at(t).observe(body)
            ra, dec, distance = astrometric.radec()
            
            sun_body = sun - body
            earth_body = earth - body
            
            sun_body_t = sun_body.at(t)
            earth_body_t = earth_body.at(t)
            
            phase_angle = sun_body_t.separation_from(earth_body_t)
            
            magnitude = calculate_magnitude(body_name, distance.au, phase_angle.degrees)
            
            planet_data[body_name] = {
                'distance': distance.au,
                'ra': ra.hours,
                'dec': dec.degrees,
                'magnitude': magnitude,
                'phase_angle': phase_angle.degrees
            }

        return planet_data
    except Exception as e:
        st.error(f"Error in calculate_planetary_positions: {str(e)}")
        return str(e)

# Streamlit UI
st.set_page_config(page_title="Solar System Position Calculator", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .small-font {
        font-size:14px !important;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
    }
    .planet-card {
        border: 1px solid #4B4B4B;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">Solar System Position Calculator</p>', unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown('<p class="medium-font">Controls</p>', unsafe_allow_html=True)
    
    with st.expander("Date and Time Selection", expanded=True):
        selected_date = st.date_input("Date", datetime.now())
        col_hour, col_minute = st.columns(2)
        with col_hour:
            hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=datetime.now().hour)
        with col_minute:
            minute = st.number_input("Minute (0-59)", min_value=0, max_value=59, value=datetime.now().minute)
    
    selected_datetime = datetime.combine(selected_date, datetime.min.time()).replace(hour=hour, minute=minute)
    selected_datetime = selected_datetime.replace(tzinfo=timezone.utc)

    if st.button("Calculate Positions", key="calculate_button"):
        with st.spinner("Calculating planetary positions..."):
            planet_data = calculate_planetary_positions(selected_datetime)
        st.session_state['planet_data'] = planet_data
        st.session_state['calculation_time'] = selected_datetime

with col1:
    st.markdown('<p class="medium-font">Planetary Positions</p>', unsafe_allow_html=True)
    if 'planet_data' in st.session_state:
        st.markdown(f'<p class="small-font">Celestial body positions at {st.session_state["calculation_time"].strftime("%Y-%m-%d %H:%M:%S UTC")}</p>', unsafe_allow_html=True)
        for planet, data in st.session_state['planet_data'].items():
            with st.expander(planet.capitalize(), expanded=True):
                st.markdown(f'<div class="planet-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="medium-font">{planet.capitalize()}</p>', unsafe_allow_html=True)
                st.markdown(f'Distance: {data["distance"]:.6f} AU', unsafe_allow_html=True)
                st.markdown(f'Right Ascension: {data["ra"]:.6f} hours', unsafe_allow_html=True)
                st.markdown(f'Declination: {data["dec"]:.6f} degrees', unsafe_allow_html=True)
                st.markdown(f'Magnitude: {data["magnitude"]:.2f}' if data["magnitude"] is not None else 'Magnitude: N/A', unsafe_allow_html=True)
                st.markdown(f'Phase Angle: {data["phase_angle"]:.2f} degrees', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please select a date and time, then click 'Calculate Positions' to view planetary data.")

# Information sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This app calculates and displays the positions of planets "
        "in our solar system based on the selected date and time. "
        "It uses the Skyfield library for accurate astronomical calculations."
    )

    st.header("Instructions")
    st.markdown(
        """
        1. Select a date using the date picker.
        2. Enter the hour (0-23) and minute (0-59).
        3. Click the 'Calculate Positions' button.
        4. View the detailed information for each planet.
        """
    )

    st.header("Legend")
    st.markdown(
        """
        - **AU**: Astronomical Unit
        - **RA**: Right Ascension (in hours)
        - **Dec**: Declination (in degrees)
        - **Magnitude**: Apparent magnitude (brightness)
        - **Phase Angle**: Angle between Earth and Sun as seen from the planet
        """
    )
