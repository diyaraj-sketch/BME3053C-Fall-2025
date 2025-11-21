import streamlit as st
import numpy as np
from scipy import signal as scipy_signal
from scipy.fftpack import fft, ifft, fftfreq
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fourier Transform Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_time_vector(duration, fs):
    """
    Generate time vector for continuous signal.
    
    Parameters:
    - duration: Total duration in seconds
    - fs: Sampling rate in Hz
    
    Returns: Time vector array
    """
    return np.linspace(0, duration, int(fs * duration), endpoint=False)


def build_signal(components, presets, t, preset_applied=None):
    """
    Build a signal from custom components and optional presets.
    
    Parameters:
    - components: List of dicts with keys: amplitude, frequency, phase, waveform, enabled
    - presets: Dict with preset name -> callable
    - t: Time vector
    - preset_applied: Name of preset to apply (overrides components)
    
    Returns: Tuple of (signal array, LaTeX expression string)
    """
    if preset_applied and preset_applied in presets:
        signal_array = presets[preset_applied](t)
        latex_expr = f"\\text{{Preset: }} {preset_applied}"
        return signal_array, latex_expr
    
    signal_array = np.zeros_like(t)
    latex_terms = []
    
    for i, comp in enumerate(components):
        if not comp['enabled']:
            continue
        
        amp = comp['amplitude']
        freq = comp['frequency']
        phase = comp['phase']
        waveform = comp['waveform']
        
        if waveform == 'sine':
            sig = amp * np.sin(2 * np.pi * freq * t + phase)
            latex_terms.append(f"{amp:.2f}\\sin(2\\pi \\cdot {freq}t + {phase:.2f})")
        else:  # cosine
            sig = amp * np.cos(2 * np.pi * freq * t + phase)
            latex_terms.append(f"{amp:.2f}\\cos(2\\pi \\cdot {freq}t + {phase:.2f})")
        
        signal_array += sig
    
    if latex_terms:
        latex_expr = " + ".join(latex_terms)
    else:
        latex_expr = "\\text{No active components}"
    
    return signal_array, latex_expr


def fft_spectrum(x, fs, window_name='rectangular', zero_pad_factor=1, misalign=False):
    """
    Compute FFT spectrum with windowing and zero-padding.
    
    Parameters:
    - x: Signal array
    - fs: Sampling rate in Hz
    - window_name: Type of window ('rectangular', 'hann', 'hamming', 'blackman', 'kaiser')
    - zero_pad_factor: Padding multiplier
    - misalign: If True, slightly shift signal to cause spectral leakage
    
    Returns: Tuple of (frequencies, magnitude spectrum, phase spectrum)
    """
    N = len(x)
    padded_N = int(N * zero_pad_factor)
    
    # Apply window
    window = compute_window(window_name, N)
    x_windowed = x * window
    
    # Apply misalignment if requested (causes spectral leakage)
    if misalign:
        # Introduce a fractional sample shift
        shift_samples = 0.3
        x_windowed = np.interp(
            np.arange(N) + shift_samples,
            np.arange(N),
            x_windowed,
            left=0, right=0
        )
    
    # Zero-pad
    x_padded = np.pad(x_windowed, (0, padded_N - N), mode='constant')
    
    # Compute FFT
    X = fft(x_padded)
    freqs = fftfreq(padded_N, 1/fs)
    
    # Return only positive frequencies
    positive_idx = freqs >= 0
    freqs = freqs[positive_idx]
    magnitude = np.abs(X[positive_idx]) / N
    phase = np.angle(X[positive_idx])
    
    return freqs, magnitude, phase


def compute_window(name, N, beta=5.0):
    """
    Compute window function.
    
    Parameters:
    - name: Window type
    - N: Window length
    - beta: Kaiser beta parameter
    
    Returns: Window array
    """
    if name == 'rectangular':
        return np.ones(N)
    elif name == 'hann':
        return np.hanning(N)
    elif name == 'hamming':
        return np.hamming(N)
    elif name == 'blackman':
        return np.blackman(N)
    elif name == 'kaiser':
        return np.kaiser(N, beta)
    else:
        return np.ones(N)


def reconstruct_band(X, freqs, f_min, f_max, fs):
    """
    Reconstruct signal from a band of frequencies.
    
    Parameters:
    - X: FFT spectrum (only positive frequencies)
    - freqs: Frequency array
    - f_min, f_max: Frequency band bounds
    - fs: Sampling rate
    
    Returns: Reconstructed time-domain signal
    """
    X_band = X.copy()
    # Zero out frequencies outside the band
    outside_band = (freqs < f_min) | (freqs > f_max)
    X_band[outside_band] = 0
    
    # Create full spectrum (positive and negative frequencies)
    X_full = np.concatenate([X_band, X_band[-2:0:-1].conj()])
    
    # Inverse FFT
    x_reconstructed = ifft(X_full).real
    
    return x_reconstructed


def estimate_spectral_metrics(freqs, magnitude, window_name):
    """
    Estimate main-lobe width and peak sidelobe level.
    
    Returns: Tuple of (main_lobe_width, peak_sidelobe_db)
    """
    # Simplified estimates based on window type
    window_metrics = {
        'rectangular': (4 * (freqs[1] - freqs[0]), -13),
        'hann': (8 * (freqs[1] - freqs[0]), -32),
        'hamming': (8 * (freqs[1] - freqs[0]), -43),
        'blackman': (12 * (freqs[1] - freqs[0]), -58),
        'kaiser': (6 * (freqs[1] - freqs[0]), -69),
    }
    
    return window_metrics.get(window_name, (4, -13))


# ============================================================================
# PRESET SIGNAL GENERATORS
# ============================================================================

def preset_square_wave(t, f=10):
    """Generate square wave."""
    return scipy_signal.square(2 * np.pi * f * t)


def preset_triangle_wave(t, f=10):
    """Generate triangle wave."""
    return scipy_signal.sawtooth(2 * np.pi * f * t, width=0.5)


def preset_sawtooth_wave(t, f=10):
    """Generate sawtooth wave."""
    return scipy_signal.sawtooth(2 * np.pi * f * t)


def preset_gaussian_pulse(t, f=10, width=0.1):
    """Generate Gaussian pulse."""
    center = np.max(t) / 2
    return np.exp(-((t - center) ** 2) / (2 * width ** 2))


def preset_step(t):
    """Generate step function."""
    return np.where(t > np.max(t) / 2, 1.0, 0.0)


def preset_impulse(t):
    """Generate impulse (Dirac delta approximation)."""
    impulse = np.zeros_like(t)
    impulse[len(impulse) // 2] = 1.0
    return impulse


def preset_chirp(t, f_start=5, f_end=50):
    """Generate chirp signal."""
    return scipy_signal.chirp(t, f_start, np.max(t), f_end, method='linear')


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'tab_selection' not in st.session_state:
    st.session_state.tab_selection = "Signal Builder"

if 'components' not in st.session_state:
    st.session_state.components = [
        {'amplitude': 1.0, 'frequency': 10, 'phase': 0, 'waveform': 'sine', 'enabled': True},
        {'amplitude': 0.5, 'frequency': 20, 'phase': 0, 'waveform': 'sine', 'enabled': False},
        {'amplitude': 0.3, 'frequency': 30, 'phase': 0, 'waveform': 'sine', 'enabled': False},
        {'amplitude': 0.2, 'frequency': 40, 'phase': 0, 'waveform': 'sine', 'enabled': False},
        {'amplitude': 0.1, 'frequency': 50, 'phase': 0, 'waveform': 'sine', 'enabled': False},
    ]

if 'preset' not in st.session_state:
    st.session_state.preset = "None"

if 'duration' not in st.session_state:
    st.session_state.duration = 1.0

if 'fs' not in st.session_state:
    st.session_state.fs = 1000

if 'window_type' not in st.session_state:
    st.session_state.window_type = 'rectangular'

if 'kaiser_beta' not in st.session_state:
    st.session_state.kaiser_beta = 5.0

# ============================================================================
# MAIN TITLE & LAYOUT
# ============================================================================

st.markdown("# 📊 Interactive Fourier Transform Explorer")
st.markdown("""
This educational application lets you explore Fourier transforms interactively. Learn how signals decompose 
into frequency components, how sampling affects your data, and how windowing reduces spectral artifacts.
""")

# ============================================================================
# SIDEBAR: GLOBAL SETTINGS & SIGNAL BUILDER
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Global Settings")
    
    st.session_state.duration = st.slider(
        "Signal Duration (seconds)",
        min_value=0.1, max_value=5.0, value=float(st.session_state.duration), step=0.1,
        help="Total time span for signal generation."
    )
    
    st.session_state.fs = st.slider(
        "Sampling Rate (Hz)",
        min_value=100, max_value=10000, value=int(st.session_state.fs), step=100,
        help="Number of samples per second. Higher rates capture finer details."
    )
    
    st.markdown("---")
    st.markdown("## 🔧 Signal Builder")
    
    st.markdown("""
    **Add sinusoid components to compose your signal.** Each component can be toggled on/off.
    The final signal is the sum of all enabled components.
    """)
    
    preset_choice = st.selectbox(
        "Quick Presets",
        ["None", "Square Wave", "Triangle Wave", "Sawtooth", "Gaussian Pulse", "Step", "Impulse", "Chirp"],
        help="""
        **Presets explained:**
        - **Square Wave**: Rich harmonics; spectrum shows odd multiples of fundamental frequency.
        - **Triangle Wave**: Smoother than square; still has harmonics but decays faster.
        - **Sawtooth**: Very sharp edges; dense harmonic content across spectrum.
        - **Gaussian Pulse**: Smooth, localized in time; spectrum is also Gaussian, spread across frequencies.
        - **Step**: Discontinuity; spectrum shows slow decay with many frequencies needed to represent edge.
        - **Impulse**: Single sharp spike; requires all frequencies with equal magnitude.
        - **Chirp**: Frequency increases over time; spectrum shows frequency sweep.
        """
    )
    st.session_state.preset = preset_choice
    
    st.markdown("---")
    st.markdown("### Component Controls (Sidebar)")
    
    for i in range(5):
        with st.expander(f"Component {i+1}", expanded=(i == 0)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.session_state.components[i]['enabled'] = st.checkbox(
                    "Active", value=st.session_state.components[i]['enabled'],
                    key=f"enabled_{i}",
                    help="Enable/disable this component."
                )
            
            st.session_state.components[i]['amplitude'] = st.slider(
                "Amplitude", 0.0, 2.0, st.session_state.components[i]['amplitude'],
                step=0.1, key=f"amp_{i}",
                help="Height of the wave. Larger amplitude = more energy at this frequency."
            )
            
            st.session_state.components[i]['frequency'] = st.slider(
                "Frequency (Hz)", 1, 100, st.session_state.components[i]['frequency'],
                step=1, key=f"freq_{i}",
                help="Oscillations per second. Higher frequency = faster oscillations."
            )
            
            st.session_state.components[i]['phase'] = st.slider(
                "Phase (radians)", 0.0, 2*np.pi, st.session_state.components[i]['phase'],
                step=0.1, key=f"phase_{i}",
                help="Time shift of the wave. Controls where the wave starts."
            )
            
            st.session_state.components[i]['waveform'] = st.selectbox(
                "Waveform", ["sine", "cosine"], 
                index=0 if st.session_state.components[i]['waveform'] == 'sine' else 1,
                key=f"wave_{i}",
                help="Sine or cosine shape. Cosine is sine shifted by 90°."
            )

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Signal Builder",
    "🔄 Time-Frequency Views",
    "⏱️ Sampling & Aliasing",
    "🪟 Windowing & Leakage"
])

# ============================================================================
# TAB 1: SIGNAL BUILDER
# ============================================================================

with tab1:
    st.markdown("## 📈 Signal Builder Tab")
    
    st.info("""
    **Purpose:** Compose a complex signal by adding sinusoid components or applying preset waveforms.
    
    **Key Concepts:**
    - A complex signal is the **sum of simple sinusoids** at different frequencies, amplitudes, and phases.
    - The **Fourier theorem** states that any periodic signal can be built this way.
    - **Amplitude** controls energy; **frequency** controls oscillation rate; **phase** controls timing.
    
    **What You Can Do:**
    1. Use the sidebar to enable/disable components and adjust their parameters.
    2. Select a preset waveform to see how it decomposes into frequencies.
    3. View the signal expression and the time-domain plot below.
    """)
    
    # Generate signal based on preset or components
    t = generate_time_vector(st.session_state.duration, st.session_state.fs)
    presets = {
        "Square Wave": lambda t: preset_square_wave(t, f=10),
        "Triangle Wave": lambda t: preset_triangle_wave(t, f=10),
        "Sawtooth": lambda t: preset_sawtooth_wave(t, f=10),
        "Gaussian Pulse": lambda t: preset_gaussian_pulse(t, f=10),
        "Step": lambda t: preset_step(t),
        "Impulse": lambda t: preset_impulse(t),
        "Chirp": lambda t: preset_chirp(t),
    }
    
    signal_array, latex_expr = build_signal(
        st.session_state.components, 
        presets, 
        t,
        preset_applied=st.session_state.preset if st.session_state.preset != "None" else None
    )
    
    st.markdown("### Signal Expression")
    st.latex(latex_expr)
    
    st.markdown("""
    **What this means:**
    - Each term represents one sinusoid component.
    - **Coefficient** (e.g., 1.0, 0.5) = amplitude
    - **sin/cos** = waveform type
    - **2π·f·t** = frequency in radians per second
    - **+phase** = time shift
    """)
    
    st.markdown("### Time-Domain Waveform")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal_array, mode='lines', name='Signal',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title="Signal in Time Domain",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - The blue line shows how your signal evolves over time.
    - **Smooth curves** indicate low-frequency components dominate.
    - **Rapid oscillations** indicate high-frequency components.
    - **Sharp transitions** suggest the presence of many high-frequency harmonics.
    """)

# ============================================================================
# TAB 2: TIME-FREQUENCY LINKED VIEWS
# ============================================================================

with tab2:
    st.markdown("## 🔄 Time-Frequency Linked Views")
    
    st.info("""
    **Purpose:** Visualize how time-domain and frequency-domain representations relate.
    
    **Key Concepts:**
    - The **frequency domain** shows which frequencies are present and their magnitudes.
    - **Magnitude spectrum** tells you the amplitude of each frequency component.
    - **Phase spectrum** tells you the timing offset of each component.
    - Sharp transitions in time require many high frequencies to represent (Fourier principle).
    
    **What You Can Do:**
    1. View side-by-side time and frequency plots.
    2. Use the reconstruction tool to filter a frequency band and see its effect.
    3. Experiment: add high-frequency components and see the wiggles in time domain!
    """)
    
    # Regenerate signal
    t = generate_time_vector(st.session_state.duration, st.session_state.fs)
    signal_array, _ = build_signal(st.session_state.components, presets, t,
                                   preset_applied=st.session_state.preset if st.session_state.preset != "None" else None)
    
    # Compute FFT
    freqs, magnitude, phase = fft_spectrum(signal_array, st.session_state.fs, window_name='rectangular', zero_pad_factor=1)
    
    # Create subplots
    col1, col2 = st.columns(2)
    
    with col1:
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=t, y=signal_array, mode='lines', name='Signal',
            line=dict(color='blue', width=2)
        ))
        fig_time.update_layout(
            title="Time Domain",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode='x',
            height=400
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        fig_freq = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                  subplot_titles=("Magnitude Spectrum", "Phase Spectrum"),
                                  vertical_spacing=0.15)
        
        fig_freq.add_trace(go.Scatter(
            x=freqs, y=magnitude, mode='lines', name='Magnitude',
            line=dict(color='green', width=2)
        ), row=1, col=1)
        
        fig_freq.add_trace(go.Scatter(
            x=freqs, y=phase, mode='lines', name='Phase',
            line=dict(color='red', width=2)
        ), row=2, col=1)
        
        fig_freq.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig_freq.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig_freq.update_yaxes(title_text="Phase (rad)", row=2, col=1)
        fig_freq.update_layout(height=400, hovermode='x unified')
        
        st.plotly_chart(fig_freq, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Band Select & Reconstruct")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        f_min = st.slider("Min Frequency (Hz)", 0, 100, 5, help="Lower bound of frequency band to keep.")
    with col2:
        f_max = st.slider("Max Frequency (Hz)", 0, 100, 30, help="Upper bound of frequency band to keep.")
    with col3:
        st.write("")  # spacing
    
    st.markdown("""
    **How band reconstruction works:**
    - Select a frequency range (e.g., 5-30 Hz).
    - We zero out all other frequencies in the FFT.
    - Inverse FFT reconstructs a filtered version of the signal.
    - This is the principle behind **low-pass, high-pass, and band-pass filters** used in biomedical signal processing!
    """)
    
    # Reconstruct
    X_full = np.concatenate([magnitude, magnitude[-2:0:-1]])
    x_reconstructed = reconstruct_band(X_full, np.concatenate([freqs, freqs[-2:0:-1]]), f_min, f_max, st.session_state.fs)
    
    fig_recon = go.Figure()
    fig_recon.add_trace(go.Scatter(x=t, y=signal_array, mode='lines', name='Original Signal',
                                    line=dict(color='blue', width=2)))
    fig_recon.add_trace(go.Scatter(x=t[:len(x_reconstructed)], y=x_reconstructed[:len(t)], 
                                    mode='lines', name=f'Reconstructed ({f_min}-{f_max} Hz)',
                                    line=dict(color='red', width=2, dash='dash')))
    fig_recon.update_layout(
        title="Original vs. Band-Reconstructed Signal",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_recon, use_container_width=True)
    
    st.markdown("""
    **Key Insight:**
    The reconstructed signal (red dashed) shows only the frequency components in your selected range.
    Notice how removing high frequencies smooths out rapid oscillations!
    """)

# ============================================================================
# TAB 3: SAMPLING & ALIASING
# ============================================================================

with tab3:
    st.markdown("## ⏱️ Sampling, Aliasing & Resolution")
    
    st.info("""
    **Purpose:** Understand how sampling rate affects signal representation and frequency aliasing.
    
    **Key Concepts:**
    - **Nyquist Frequency** = fs / 2 (half the sampling rate). You can only represent frequencies up to this.
    - **Nyquist-Shannon Theorem**: To accurately capture a signal, sample at least 2× the highest frequency.
    - **Aliasing**: Frequencies above Nyquist "fold back" and appear as lower frequencies (distortion!).
    - **Zero-Padding**: Increases frequency resolution in the FFT but doesn't add real information.
    
    **What You Can Do:**
    1. Adjust sampling rate and see how sampled points align with continuous signal.
    2. Add a high-frequency component and watch it alias if fs is too low.
    3. Toggle zero-padding to see how it changes frequency resolution.
    """)
    
    # Regenerate signal with a high-frequency component for aliasing demo
    t = generate_time_vector(st.session_state.duration, st.session_state.fs)
    signal_array, _ = build_signal(st.session_state.components, presets, t,
                                   preset_applied=st.session_state.preset if st.session_state.preset != "None" else None)
    
    # Sampling controls
    col1, col2 = st.columns(2)
    with col1:
        fs_demo = st.slider("Sampling Rate (Hz)", 100, 2000, int(st.session_state.fs), step=100,
                           help="Higher sampling rates better capture signal details.")
        demo_freq = st.slider("Demo Frequency (Hz)", 5, 100, 30, step=1,
                             help="Frequency to test aliasing. If > fs/2, it will alias.")
    
    with col2:
        zero_pad = st.checkbox("Enable Zero-Padding", False,
                              help="Increases FFT bins without adding information.")
        if zero_pad:
            pad_factor = st.slider("Padding Factor", 1, 10, 2, step=1,
                                  help="Multiplier for zero-padding. More padding = smoother spectrum plot.")
        else:
            pad_factor = 1
    
    st.markdown(f"""
    **Nyquist Frequency for current settings: {fs_demo / 2:.1f} Hz**
    
    The demo frequency ({demo_freq} Hz) is:
    - **Representable** ✓ (below Nyquist) if {demo_freq} < {fs_demo / 2:.1f}
    - **Will alias** ✗ (above Nyquist) if {demo_freq} > {fs_demo / 2:.1f}
    """)
    
    # Create continuous reference and sampled version
    t_continuous = np.linspace(0, st.session_state.duration, 5000, endpoint=False)
    sig_continuous = np.sin(2 * np.pi * demo_freq * t_continuous)
    
    t_sampled = np.linspace(0, st.session_state.duration, int(fs_demo * st.session_state.duration), endpoint=False)
    sig_sampled = np.sin(2 * np.pi * demo_freq * t_sampled)
    
    fig_sampling = go.Figure()
    fig_sampling.add_trace(go.Scatter(x=t_continuous, y=sig_continuous, mode='lines', 
                                       name='Continuous', line=dict(color='blue', width=1)))
    fig_sampling.add_trace(go.Scatter(x=t_sampled, y=sig_sampled, mode='markers', 
                                       name=f'Sampled (fs={fs_demo} Hz)', 
                                       marker=dict(color='red', size=6)))
    fig_sampling.update_layout(
        title=f"Sampling: {demo_freq} Hz Signal at {fs_demo} Hz Sampling Rate",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=400,
        hovermode='x'
    )
    st.plotly_chart(fig_sampling, use_container_width=True)
    
    st.markdown("""
    **What you see:**
    - Blue line = continuous signal (ideal).
    - Red dots = actual samples at your sampling rate.
    - If dots are sparse, the continuous signal is undersampled (risk of aliasing).
    - If dots trace the curve closely, sampling is adequate.
    """)
    
    # FFT with aliasing demonstration
    freqs, magnitude, phase = fft_spectrum(sig_sampled, fs_demo, window_name='rectangular', 
                                           zero_pad_factor=pad_factor, misalign=False)
    
    nyquist = fs_demo / 2
    aliased_freq = demo_freq - fs_demo if demo_freq > nyquist else demo_freq
    
    fig_alias = go.Figure()
    fig_alias.add_trace(go.Scatter(x=freqs, y=magnitude, mode='lines', name='FFT Magnitude',
                                    line=dict(color='green', width=2)))
    
    if demo_freq > nyquist:
        fig_alias.add_vline(x=aliased_freq, line_dash="dash", line_color="red",
                           annotation_text=f"Aliased to {aliased_freq:.1f} Hz", 
                           annotation_position="top right")
    
    fig_alias.add_vline(x=nyquist, line_dash="dash", line_color="orange",
                       annotation_text=f"Nyquist = {nyquist:.1f} Hz",
                       annotation_position="top left")
    
    fig_alias.update_layout(
        title="Frequency Spectrum (FFT)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        height=400,
        hovermode='x'
    )
    st.plotly_chart(fig_alias, use_container_width=True)
    
    if demo_freq > nyquist:
        st.warning(f"""
        ⚠️ **Aliasing Detected!**
        
        Your {demo_freq} Hz signal exceeds the Nyquist frequency ({nyquist} Hz).
        It appears as a **phantom peak at {aliased_freq:.1f} Hz** in the spectrum.
        This is the **aliased frequency** and represents a distortion of the true signal.
        
        **Solution:** Increase sampling rate to at least {2 * demo_freq} Hz.
        """)
    else:
        st.success(f"✓ Signal is properly sampled. No aliasing detected.")
    
    st.markdown("""
    **Resolution & Zero-Padding Explained:**
    - **Frequency resolution** (bin width) = fs / N, where N is FFT size.
    - Zero-padding increases N, making the plot appear smoother, but doesn't reveal new frequencies.
    - It's useful for visualization and interpolation, but doesn't add real information.
    """)

# ============================================================================
# TAB 4: WINDOWING & SPECTRAL LEAKAGE
# ============================================================================

with tab4:
    st.markdown("## 🪟 Windowing & Spectral Leakage")
    
    st.info("""
    **Purpose:** Learn how windowing reduces spectral artifacts and improves frequency localization.
    
    **Key Concepts:**
    - **Spectral Leakage**: When a frequency doesn't align with FFT bins, energy "leaks" into adjacent bins.
    - **Windowing**: Multiply signal by a window function (taper edges) to reduce leakage.
    - Different windows have different tradeoffs (main-lobe width vs. sidelobe level).
    - **Main-lobe width**: How concentrated the peak is (narrower = better frequency resolution).
    - **Sidelobe level**: How much energy appears in adjacent bins (lower = cleaner spectrum).
    
    **What You Can Do:**
    1. Select different window types and see how they change the spectrum.
    2. Observe the window shape and understand why it reduces discontinuities.
    3. Toggle frequency misalignment to see spectral leakage in action.
    """)
    
    # Regenerate signal
    t = generate_time_vector(st.session_state.duration, st.session_state.fs)
    signal_array, _ = build_signal(st.session_state.components, presets, t,
                                   preset_applied=st.session_state.preset if st.session_state.preset != "None" else None)
    
    # Window selection
    col1, col2 = st.columns(2)
    with col1:
        window_type = st.selectbox(
            "Window Type",
            ["Rectangular", "Hann", "Hamming", "Blackman", "Kaiser"],
            help="""
            **Window Characteristics:**
            - **Rectangular**: Simple, sharp cutoffs; high sidelobes, narrow main lobe.
            - **Hann/Hamming**: Smooth tapering; moderate sidelobes, wider main lobe.
            - **Blackman**: Heavy tapering; very low sidelobes, wider main lobe.
            - **Kaiser**: Tunable; adjust β for tradeoff between main-lobe and sidelobe.
            
            **Use Cases:**
            - General analysis: Hann window
            - Close frequencies: Use narrow main-lobe (Rectangular or Kaiser with low β)
            - Noise reduction: Use low-sidelobe (Blackman or Kaiser with high β)
            """
        ).lower()
    
    with col2:
        if window_type == 'kaiser':
            kaiser_beta = st.slider("Kaiser β Parameter", 0.0, 20.0, 5.0, step=0.5,
                                   help="Higher β = lower sidelobes but wider main lobe.")
        else:
            kaiser_beta = 5.0
    
    misalign_toggle = st.checkbox("Misalign Frequency to Bin",
                                  help="""
                                  Toggle this to see spectral leakage.
                                  When checked, the signal frequency is shifted slightly so it 
                                  doesn't align with FFT bins, causing energy to spread across bins.
                                  """)
    
    st.markdown("---")
    st.markdown("### Window Shape (Time Domain)")
    
    N = len(signal_array)
    window = compute_window(window_type, N, beta=kaiser_beta)
    
    fig_window = go.Figure()
    t_window = np.arange(N) / st.session_state.fs
    fig_window.add_trace(go.Scatter(x=t_window, y=window, mode='lines', name='Window',
                                     line=dict(color='purple', width=2)))
    fig_window.update_layout(
        title=f"{window_type.capitalize()} Window",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
    )
    st.plotly_chart(fig_window, use_container_width=True)
    
    st.markdown(f"""
    **Why windowing helps:**
    - The raw signal has hard edges (discontinuities) at start/end.
    - These discontinuities require many high-frequency components to represent.
    - The {window_type.capitalize()} window tapers the edges smoothly.
    - This reduces the edge discontinuities and thus reduces spectral leakage.
    """)
    
    st.markdown("---")
    st.markdown("### Windowed vs. Rectangular Spectrum")
    
    # Compute FFT with and without window
    freqs_rect, mag_rect, _ = fft_spectrum(signal_array, st.session_state.fs, 
                                           window_name='rectangular', zero_pad_factor=2, 
                                           misalign=misalign_toggle)
    freqs_wind, mag_wind, _ = fft_spectrum(signal_array, st.session_state.fs, 
                                           window_name=window_type, zero_pad_factor=2,
                                           misalign=misalign_toggle)
    
    fig_spectrum_comp = go.Figure()
    fig_spectrum_comp.add_trace(go.Scatter(x=freqs_rect, y=mag_rect, mode='lines', 
                                            name='Rectangular Window',
                                            line=dict(color='red', width=2)))
    fig_spectrum_comp.add_trace(go.Scatter(x=freqs_wind, y=mag_wind, mode='lines',
                                            name=f'{window_type.capitalize()} Window',
                                            line=dict(color='green', width=2)))
    fig_spectrum_comp.update_layout(
        title="Spectrum Comparison",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (log scale)",
        yaxis_type="log",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_spectrum_comp, use_container_width=True)
    
    if misalign_toggle:
        st.warning("""
        **Spectral Leakage in Action!**
        
        Notice the energy spreading (skirts) around the main peaks? 
        This is spectral leakage due to frequency misalignment.
        
        The windowed spectrum has lower sidelobes (cleaner background) than rectangular,
        but at the cost of a wider main lobe (less frequency resolution).
        """)
    else:
        st.success("""
        **Frequencies are well-aligned.**
        
        Enable "Misalign Frequency" above to see spectral leakage.
        """)
    
    st.markdown("---")
    st.markdown("### Spectral Metrics")
    
    main_lobe_width, peak_sidelobe = estimate_spectral_metrics(freqs_wind, mag_wind, window_type)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Main-Lobe Width (Hz)", f"{main_lobe_width:.2f}",
                  help="Narrower main lobe = better frequency resolution.")
    with col2:
        st.metric("Peak Sidelobe Level (dB)", f"{peak_sidelobe}",
                  help="Lower sidelobes = cleaner spectrum, less spectral leakage.")
    
    st.markdown("""
    **What these metrics mean:**
    - **Main-Lobe Width**: How spread out a frequency peak is. Narrower is better for distinguishing close frequencies.
    - **Peak Sidelobe Level**: How much energy leaks into adjacent bins. Lower is better for detecting weak signals near strong signals.
    
    **Window Tradeoffs:**
    - Rectangular: Narrowest main lobe, highest sidelobes. Best for isolated, strong signals.
    - Hann/Hamming: Moderate main lobe, moderate sidelobes. Good all-around choice.
    - Blackman: Widest main lobe, lowest sidelobes. Best when you need to suppress leakage.
    - Kaiser: Tunable β parameter lets you balance the tradeoff.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📚 Educational Resources

This app demonstrates concepts from **BME3053C: Computer Applications for BME**.

**Key Takeaways:**
- The Fourier transform decomposes signals into frequency components.
- Proper sampling rate (Nyquist theorem) is critical to avoid aliasing.
- Windowing reduces spectral leakage at the cost of frequency resolution.
- These principles are fundamental to biomedical signal processing (ECG, EEG, etc.).

**Further Learning:**
- Experiment with different component combinations in the Signal Builder.
- Notice how sharp transitions require many high-frequency components.
- Use band reconstruction to implement low-pass, high-pass, and band-pass filters.
- Observe how different windows affect spectral characteristics.

---
*Created for BME3053C - Interactive Fourier Transform Learning*
""")
