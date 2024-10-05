import streamlit as st
import math
from datetime import datetime, timezone
import numpy as np
from skyfield.api import load
from skyfield.data import mpc
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.almanac import find_discrete, risings_and_settings
import plotly.graph_objects as go 

# Constants for magnitude calculations
STANDARD_MAGNITUDES = {
    'mercury': -0.36, 'venus': -4.40, 'mars': -1.52,
    'jupiter': -9.40, 'saturn': -8.88, 'uranus': -7.19, 'neptune': -6.87
}

def calculate_approximate_position(planet, jde):
    planet_constants = {
        'mercury': [0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
        'venus': [0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
        'mars': [1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
        'jupiter': [5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
        'saturn': [9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
        'uranus': [19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
        'neptune': [30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574]
    }

    try:
        a, e, I, L, long_peri, long_node = planet_constants[planet]

        cy = (jde - 2451545.0) / 36525
        L = (L + 360.0 * 99999.0 * cy) % 360
        M = (L - long_peri) % 360
        E = M + (180 / math.pi) * e * math.sin(math.radians(M))
        v = 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(math.radians(E / 2)))
        v = math.degrees(v)
        r = a * (1 - e * math.cos(math.radians(E)))

        x = r * (math.cos(math.radians(long_node)) * math.cos(math.radians(v + long_peri - long_node)) - 
                 math.sin(math.radians(long_node)) * math.sin(math.radians(v + long_peri - long_node)) * math.cos(math.radians(I)))
        y = r * (math.sin(math.radians(long_node)) * math.cos(math.radians(v + long_peri - long_node)) + 
                 math.cos(math.radians(long_node)) * math.sin(math.radians(v + long_peri - long_node)) * math.cos(math.radians(I)))
        z = r * math.sin(math.radians(v + long_peri - long_node)) * math.sin(math.radians(I))

        return x, y, z
    except Exception as e:
        st.error(f"Error in approximate position calculation for {planet}: {str(e)}")
        return None

def calculate_magnitude(body_name, distance, phase_angle):
    if body_name == 'moon':
        return -12.7 + 0.026 * abs(phase_angle) + 4e-9 * phase_angle**4
    elif body_name in STANDARD_MAGNITUDES:
        std_mag = STANDARD_MAGNITUDES[body_name]
        return std_mag + 5 * math.log10(distance * (distance - 1))
    else:
        return None

def calculate_planetary_positions(selected_datetime=None):
    try:
        eph = load('de421.bsp')
        earth = eph['earth']
        sun = eph['sun']

        bodies = {
            'moon': 'moon',
            'mercury': 'mercury barycenter',
            'venus': 'venus barycenter',
            'mars': 'mars barycenter',
            'jupiter': 'jupiter barycenter',
            'saturn': 'saturn barycenter',
            'uranus': 'uranus barycenter',
            'neptune': 'neptune barycenter',
            'pluto': 'pluto barycenter'
        }

        ts = load.timescale()
        if selected_datetime:
            t = ts.from_datetime(selected_datetime)
        else:
            t = ts.now()

        jde = t.tt

        results = []
        results.append(f"Celestial body positions at {t.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}:")
        results.append("Body      | Distance (AU) | RA          | Dec         | Magnitude | Phase Angle | Approx. Distance (AU)")
        results.append("-" * 110)

        positions = {}

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
            
            approx_distance = "N/A"
            if body_name in ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']:
                approx_pos = calculate_approximate_position(body_name, jde)
                if approx_pos:
                    x, y, z = approx_pos
                    approx_distance = f"{math.sqrt(x**2 + y**2 + z**2):.6f}"
            
            heliocentric = sun.at(t).observe(body)
            x, y, z = heliocentric.position.au
            positions[body_name] = [x, y, z]
            
            magnitude_str = f"{magnitude:.2f}" if magnitude is not None else "N/A"
            
            results.append(f"{body_name.capitalize():10} | {distance.au:.6f}      | {ra.hours:.6f}  | {dec.degrees:.6f} | {magnitude_str:8} | {phase_angle.degrees:.2f}    | {approx_distance}")

        return "\n".join(results), positions
    except Exception as e:
        st.error(f"Error in calculate_planetary_positions: {str(e)}")
        return str(e), {}

def plot_solar_system(positions):
    fig = go.Figure()

    # Add the Sun
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color='yellow'), name='Sun'))

    colors = {
        'mercury': 'gray', 'venus': 'orange', 'earth': 'blue', 'mars': 'red',
        'jupiter': 'brown', 'saturn': 'gold', 'uranus': 'lightblue', 'neptune': 'blue',
        'pluto': 'gray', 'moon': 'lightgray'
    }

    for body, position in positions.items():
        x, y, z = position
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(size=5, color=colors[body]), name=body.capitalize()))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            aspectmode='data'
        ),
        title='Solar System Visualization'
    )

    return fig

st.set_page_config(page_title="Solar System Positions", layout="wide")

st.title("Solar System Planetary Positions")

col1, col2 = st.columns([2, 1])

with col2:
    st.header("Select Date and Time")
    selected_date = st.date_input("Date", datetime.now())
    selected_time = st.time_input("Time", datetime.now().time())
    selected_datetime = datetime.combine(selected_date, selected_time)
    selected_datetime = selected_datetime.replace(tzinfo=timezone.utc)

    if st.button("Calculate Positions"):
        positions_text, positions = calculate_planetary_positions(selected_datetime)
        st.session_state['positions_text'] = positions_text
        st.session_state['positions'] = positions

with col1:
    if 'positions_text' in st.session_state:
        st.text(st.session_state['positions_text'])
    
    if 'positions' in st.session_state:
        fig = plot_solar_system(st.session_state['positions'])
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.header("About")
st.sidebar.info(
    "This app calculates and visualizes the positions of planets and other "
    "celestial bodies in our solar system based on the selected date and time. "
    "It uses the Skyfield library for accurate astronomical calculations."
)

st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1. Select a date and time using the inputs on the right.
    2. Click the 'Calculate Positions' button.
    3. View the textual output and 3D visualization of the solar system.
    4. You can rotate and zoom the 3D plot using your mouse.
    """
)
