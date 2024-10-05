import streamlit as st
import math
from datetime import datetime, timezone, timedelta
import numpy as np
from skyfield.api import load, wgs84
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield import almanac
import pandas as pd

# Load ephemeris
eph = load('de421.bsp')
ts = load.timescale()

# Constants
STANDARD_MAGNITUDES = {
    'mercury': -0.36, 'venus': -4.40, 'mars': -1.52,
    'jupiter': -9.40, 'saturn': -8.88, 'uranus': -7.19, 'neptune': -6.87
}

PLANET_PHYSICAL_DATA = {
    'mercury': {'mass': 3.3011e23, 'radius': 2439.7, 'rotation_period': 58.646},
    'venus': {'mass': 4.8675e24, 'radius': 6051.8, 'rotation_period': -243.025},
    'earth': {'mass': 5.97237e24, 'radius': 6371.0, 'rotation_period': 23.93444},
    'mars': {'mass': 6.4171e23, 'radius': 3389.5, 'rotation_period': 24.62},
    'jupiter': {'mass': 1.8982e27, 'radius': 69911, 'rotation_period': 9.925},
    'saturn': {'mass': 5.6834e26, 'radius': 58232, 'rotation_period': 10.656},
    'uranus': {'mass': 8.6810e25, 'radius': 25362, 'rotation_period': -17.24},
    'neptune': {'mass': 1.02413e26, 'radius': 24622, 'rotation_period': 16.11},
    'pluto': {'mass': 1.303e22, 'radius': 1188.3, 'rotation_period': -6.387}
}

def calculate_magnitude(body_name, distance, phase_angle):
    if body_name == 'moon':
        return -12.7 + 0.026 * abs(phase_angle) + 4e-9 * phase_angle**4
    elif body_name in STANDARD_MAGNITUDES:
        std_mag = STANDARD_MAGNITUDES[body_name]
        return std_mag + 5 * math.log10(max(distance * (distance - 1), 1e-10))
    else:
        return None  # For bodies we don't have magnitude data for, like Earth or Pluto

def calculate_planetary_positions(selected_datetime):
    earth = eph['earth']
    sun = eph['sun']

    bodies = {
        'mercury': eph['mercury barycenter'],
        'venus': eph['venus barycenter'],
        'earth': eph['earth'],
        'mars': eph['mars barycenter'],
        'jupiter': eph['jupiter barycenter'],
        'saturn': eph['saturn barycenter'],
        'uranus': eph['uranus barycenter'],
        'neptune': eph['neptune barycenter'],
        'pluto': eph['pluto barycenter']
    }

    t = ts.from_datetime(selected_datetime)

    planet_data = {}
    for body_name, body in bodies.items():
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
            'magnitude': magnitude,  # This might be None for some bodies
            'phase_angle': phase_angle.degrees
        }

    return planet_data

def get_moon_phase(date):
    t = ts.from_datetime(date)
    phase = almanac.moon_phase(eph, t)
    illumination = almanac.fraction_illuminated(eph, 'moon', t)
    return phase.degrees, illumination

def au_converter(value, from_unit, to_unit):
    au_to_km = 149597870.7
    au_to_miles = 92955807.3
    au_to_lightyears = 1 / 63241.1

    # Convert to AU first
    if from_unit == 'km':
        au_value = value / au_to_km
    elif from_unit == 'miles':
        au_value = value / au_to_miles
    elif from_unit == 'lightyears':
        au_value = value / au_to_lightyears
    else:
        au_value = value

    # Convert AU to desired unit
    if to_unit == 'km':
        return au_value * au_to_km
    elif to_unit == 'miles':
        return au_value * au_to_miles
    elif to_unit == 'lightyears':
        return au_value * au_to_lightyears
    else:
        return au_value

def calculate_barycenter(selected_datetime):
    t = ts.from_datetime(selected_datetime)
    sun = eph['sun']
    
    bodies = [
        ('jupiter', 1.898e27),
        ('saturn', 5.683e26),
        ('uranus', 8.681e25),
        ('neptune', 1.024e26),
    ]
    
    total_mass = sum(mass for _, mass in bodies) + 1.989e30  # Sun's mass
    
    barycenter = np.zeros(3)
    for body, mass in bodies:
        pos = eph[body + ' barycenter'].at(t).position.au
        barycenter += pos * mass
    
    barycenter /= total_mass
    
    sun_pos = sun.at(t).position.au
    offset = sun_pos - barycenter
    
    return offset

def is_planet_visible(magnitude):
    if magnitude is None:
        return "Unknown"
    return "Yes" if magnitude < 6 else "No"  # Generally, objects brighter than magnitude 6 are visible to the naked eye

def find_planetary_events(start_date, end_date):
    events = []
    t0 = ts.from_datetime(datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc))
    t1 = ts.from_datetime(datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc))
    earth = eph['earth']

    current_date = start_date
    while current_date <= end_date:
        t = ts.from_datetime(datetime.combine(current_date, datetime.min.time()).replace(tzinfo=timezone.utc))
        positions = calculate_planetary_positions(t.utc_datetime())
        
        # Check for conjunctions
        for planet1, data1 in positions.items():
            for planet2, data2 in positions.items():
                if planet1 < planet2:
                    separation = abs(data1['ra'] - data2['ra'])
                    if separation < 1/15:  # less than 1 degree
                        events.append(f"Conjunction between {planet1} and {planet2} on {current_date}")

        # Check for oppositions
        sun_pos = earth.at(t).observe(eph['sun']).radec()[0].hours
        for planet, data in positions.items():
            if planet not in ['mercury', 'venus']:
                separation = abs(data['ra'] - sun_pos)
                if abs(separation - 12) < 1/24:  # within 1 hour of exact opposition
                    events.append(f"Opposition of {planet} on {current_date}")

        current_date += timedelta(days=1)

    return events

def find_planetary_alignments(selected_datetime, tolerance=5/15):  # tolerance in hours (5 degrees / 15 degrees per hour)
    t = ts.from_datetime(selected_datetime)
    earth = eph['earth']
    planets = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    positions = []

    for planet in planets:
        body = eph[f'{planet} barycenter']
        ra, dec, _ = earth.at(t).observe(body).radec()
        positions.append((planet, ra.hours))

    positions.sort(key=lambda x: x[1])
    
    alignments = []
    for i in range(len(positions) - 2):
        if abs(positions[i][1] - positions[i+2][1]) <= tolerance:
            alignments.append(f"Alignment of {positions[i][0]}, {positions[i+1][0]}, and {positions[i+2][0]}")

    return alignments


# Streamlit UI
st.set_page_config(page_title="Enhanced Solar System Calculator", layout="wide")

st.title("Enhanced Solar System Calculator")

col1, col2 = st.columns([3, 1])

with col2:
    st.header("Controls")
    
    selected_date = st.date_input("Date", datetime.now())
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=datetime.now().hour)
    minute = st.number_input("Minute (0-59)", min_value=0, max_value=59, value=datetime.now().minute)
    
    selected_datetime = datetime.combine(selected_date, datetime.min.time()).replace(hour=hour, minute=minute)
    selected_datetime = selected_datetime.replace(tzinfo=timezone.utc)

    if st.button("Calculate", key="calculate_button"):
        with st.spinner("Calculating..."):
            planet_data = calculate_planetary_positions(selected_datetime)
            moon_phase, moon_illumination = get_moon_phase(selected_datetime)
            barycenter_offset = calculate_barycenter(selected_datetime)
            alignments = find_planetary_alignments(selected_datetime)
        
        st.session_state['planet_data'] = planet_data
        st.session_state['moon_phase'] = moon_phase
        st.session_state['moon_illumination'] = moon_illumination
        st.session_state['barycenter_offset'] = barycenter_offset
        st.session_state['alignments'] = alignments
        st.session_state['calculation_time'] = selected_datetime

    # Date range for planetary events
    st.subheader("Planetary Events")
    start_date = st.date_input("Start Date", datetime.now().date())
    end_date = st.date_input("End Date", (datetime.now() + timedelta(days=30)).date())
    if st.button("Find Events"):
        events = find_planetary_events(start_date, end_date)
        st.session_state['events'] = events

with col1:
    if 'planet_data' in st.session_state:
        st.header(f"Results for {st.session_state['calculation_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        st.subheader("Planetary Positions")
        for planet, data in st.session_state['planet_data'].items():
            with st.expander(planet.capitalize(), expanded=True):
                st.write(f"Distance: {data['distance']:.6f} AU")
                st.write(f"Right Ascension: {data['ra']:.6f} hours")
                st.write(f"Declination: {data['dec']:.6f} degrees")
                st.write(f"Magnitude: {data['magnitude']:.2f}" if data['magnitude'] is not None else "Magnitude: N/A")
                st.write(f"Phase Angle: {data['phase_angle']:.2f} degrees")
                st.write(f"Visible to naked eye: {is_planet_visible(data['magnitude'])}")
                
                # Physical data
                physical_data = PLANET_PHYSICAL_DATA[planet]
                st.write(f"Mass: {physical_data['mass']:.2e} kg")
                st.write(f"Radius: {physical_data['radius']:.1f} km")
                st.write(f"Rotation Period: {abs(physical_data['rotation_period']):.2f} {'Earth days' if abs(physical_data['rotation_period']) > 1 else 'Earth day'}")
                if physical_data['rotation_period'] < 0:
                    st.write("(Retrograde rotation)")

        st.subheader("Moon Information")
        st.write(f"Moon Phase: {st.session_state['moon_phase']:.2f} degrees")
        st.write(f"Moon Illumination: {st.session_state['moon_illumination']*100:.2f}%")

        st.subheader("Solar System Barycenter")
        st.write(f"Offset from Sun: {np.linalg.norm(st.session_state['barycenter_offset']):.6f} AU")

        if st.session_state['alignments']:
            st.subheader("Planetary Alignments")
            for alignment in st.session_state['alignments']:
                st.write(alignment)

    if 'events' in st.session_state:
        st.subheader("Upcoming Planetary Events")
        for event in st.session_state['events']:
            st.write(event)

    else:
        st.info("Please select a date and time, then click 'Calculate' to view astronomical data.")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This enhanced calculator provides detailed information about planetary positions, "
    "moon phases, planetary events, and more. It uses the Skyfield library for accurate "
    "astronomical calculations."
)

st.sidebar.header("AU Converter")
value = st.sidebar.number_input("Enter value:", value=1.0)
from_unit = st.sidebar.selectbox("From unit:", ['AU', 'km', 'miles', 'lightyears'])
to_unit = st.sidebar.selectbox("To unit:", ['AU', 'km', 'miles', 'lightyears'])

if st.sidebar.button("Convert"):
    result = au_converter(value, from_unit, to_unit)
    st.sidebar.write(f"{value} {from_unit} = {result:.6e} {to_unit}")

st.sidebar.header("Export Data")
if 'planet_data' in st.session_state:
    df = pd.DataFrame(st.session_state['planet_data']).T
    df = df.reset_index()
    df = df.rename(columns={'index': 'planet'})
    
    # Convert complex numbers to strings if present
    for col in df.columns:
        if df[col].dtype == 'complex128':
            df[col] = df[col].apply(lambda x: f"{x.real:.6f} + {x.imag:.6f}j" if x.imag != 0 else f"{x.real:.6f}")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="planetary_data.csv",
        mime="text/csv",
    )
else:
    st.sidebar.write("Calculate planetary positions first to enable data export.")

# Comparative View
st.sidebar.header("Comparative View")
compare_date = st.sidebar.date_input("Compare Date", datetime.now() + timedelta(days=30))
compare_hour = st.sidebar.number_input("Compare Hour (0-23)", min_value=0, max_value=23, value=datetime.now().hour)
compare_minute = st.sidebar.number_input("Compare Minute (0-59)", min_value=0, max_value=59, value=datetime.now().minute)

compare_datetime = datetime.combine(compare_date, datetime.min.time()).replace(hour=compare_hour, minute=compare_minute)
compare_datetime = compare_datetime.replace(tzinfo=timezone.utc)

if st.sidebar.button("Compare"):
    with st.spinner("Calculating comparison..."):
        current_data = calculate_planetary_positions(selected_datetime)
        compare_data = calculate_planetary_positions(compare_datetime)
        
        st.subheader("Planetary Position Comparison")
        st.write(f"Comparing positions between:")
        st.write(f"Current date: {selected_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        st.write(f"Compare date: {compare_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        for planet in current_data.keys():
            with st.expander(f"{planet.capitalize()} Comparison"):
                current = current_data[planet]
                future = compare_data[planet]
                st.write(f"Distance change: {future['distance'] - current['distance']:.6f} AU")
                st.write(f"RA change: {future['ra'] - current['ra']:.6f} hours")
                st.write(f"Dec change: {future['dec'] - current['dec']:.6f} degrees")
                if current['magnitude'] is not None and future['magnitude'] is not None:
                    mag_change = future['magnitude'] - current['magnitude']
                    if abs(mag_change) > 10:  # Arbitrary threshold, adjust as needed
                        st.write(f"Magnitude change: Unrealistic value ({mag_change:.2f}). Please check calculations.")
                    else:
                        st.write(f"Magnitude change: {mag_change:.2f}")
                else:
                    st.write("Magnitude change: Not available")
