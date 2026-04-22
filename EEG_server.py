import time
import json
import asyncio
import argparse
import threading
from io import BytesIO

import numpy as np
import mne
import pygame
import websockets
from flask import Flask, jsonify, send_file, Response
from flask_cors import CORS
from pythonosc.udp_client import SimpleUDPClient
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

from mne_lsl.lsl import resolve_streams, StreamInlet
from libs.filters import filter_and_drop_dead_channels
from libs.parse import get_channels_from_xml_desc, parse_picks
from libs.plot import plot_to_pygame

# Global variables to store the latest data
latest_filtered_data = None
latest_raw_data = None
latest_info = {}
new_samples_count = 0
trim_seconds = 3
data_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

def plot_raw_eeg(raw, duration, start_offset=1.0, end_offset=1.0):
    """Plot raw EEG data with multiple channels"""
    data = raw.get_data()
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']

    # Define the time range to plot
    data_length_seconds = len(data[0]) / sfreq
    effective_duration = min(data_length_seconds, duration)
    start_time = start_offset
    end_time = effective_duration - end_offset
    start_sample = int(start_time * sfreq)
    end_sample = int(end_time * sfreq)

    # Create figure and subplots
    n_channels = len(ch_names)
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 0.75), sharex=True)

    y_min = 50 * 1e-6
    y_max = -50 * 1e-6

    # Calculate time array
    total_samples = end_sample - start_sample
    if duration - effective_duration > 0:
        left_bound = -duration + (duration - effective_duration) + start_offset
    else:
        left_bound = -duration + start_offset

    right_bound = -end_offset
    time_offsets = np.linspace(left_bound, right_bound, total_samples)

    # Plot each channel
    if n_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time_offsets, data[i, start_sample:end_sample], color='black', linewidth=0.25)
        ax.set_ylabel(ch_names[i], rotation=0, labelpad=5, ha='left')
        ax.set_xlim(-duration + start_offset, -end_offset)
        ax.set_ylim(y_min, y_max)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.set_yticks([])

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    return fig

@app.route('/data/filtered', methods=['GET'])
def get_filtered_data():
    """Serve filtered EEG data as JSON with channel names as keys, excluding first/last 2 seconds (filter ringing)"""
    with data_lock:
        if latest_filtered_data is None:
            return jsonify({'error': 'No data available yet'}), 503

        channels = latest_info.get('channels', [])
        sampling_rate = latest_info.get('sampling_rate', 0)

        # Calculate samples to slice away (2 seconds on each end)
        trim = int(sampling_rate * trim_seconds)
        n_samples_total = latest_filtered_data.shape[1]

        if n_samples_total > 2 * trim:
            start_idx = trim
            end_idx = n_samples_total - trim
            sliced_data = latest_filtered_data[:, start_idx:end_idx]
        else:
            sliced_data = latest_filtered_data
            start_idx = 0
            end_idx = n_samples_total

        # Create dict mapping channel name to data array
        channel_data = {
            channel: sliced_data[i].tolist()
            for i, channel in enumerate(channels)
        }

        # Calculate sample timestamps relative to current time
        n_samples = sliced_data.shape[1]
        current_time = time.time()
        time_per_sample = 1.0 / sampling_rate if sampling_rate > 0 else 0
        # Adjust for the sliced start position
        sample_timestamps = [
            current_time - (n_samples_total - start_idx - i) * time_per_sample
            for i in range(n_samples)
        ]

        return jsonify({
            'data': channel_data,
            'sampling_rate': sampling_rate,
            'timestamp': current_time,
            'sample_timestamps': sample_timestamps,
            'n_samples': n_samples
        })

@app.route('/data/raw', methods=['GET'])
def get_raw_data():
    """Serve raw (unfiltered) EEG data as JSON with channel names as keys, sliced to match filtered data timespan"""
    with data_lock:
        if latest_raw_data is None:
            return jsonify({'error': 'No data available yet'}), 503

        # Get original channel names (before any channel dropping)
        all_channels = latest_info.get('channels', [])
        sampling_rate = latest_info.get('sampling_rate', 0)

        trim = int(sampling_rate * trim_seconds)
        n_samples_total = latest_raw_data.shape[1]

        if n_samples_total > 2 * trim:
            start_idx = trim
            end_idx = n_samples_total - trim
            sliced_data = latest_raw_data[:, start_idx:end_idx]
        else:
            sliced_data = latest_raw_data
            start_idx = 0
            end_idx = n_samples_total

        # Create dict mapping channel name to data array
        channel_data = {
            channel: sliced_data[i].tolist()
            for i, channel in enumerate(all_channels)
        }

        # Calculate sample timestamps relative to current time
        n_samples = sliced_data.shape[1]
        current_time = time.time()
        time_per_sample = 1.0 / sampling_rate if sampling_rate > 0 else 0
        # Adjust for the sliced start position
        sample_timestamps = [
            current_time - (n_samples_total - start_idx - i) * time_per_sample
            for i in range(n_samples)
        ]

        return jsonify({
            'data': channel_data,
            'sampling_rate': sampling_rate,
            'timestamp': current_time,
            'sample_timestamps': sample_timestamps,
            'n_samples': n_samples
        })

@app.route('/plot/raw', methods=['GET'])
def get_raw_plot():
    """Serve a plot of the raw EEG data as PNG"""
    with data_lock:
        if latest_filtered_data is None:
            return Response('No data available yet', status=503)

        try:
            # Create MNE Raw object from filtered data
            raw = mne.io.RawArray(
                latest_filtered_data,
                mne.create_info(
                    latest_info.get('channels', []),
                    latest_info.get('sampling_rate', 250),
                    ch_types='eeg'
                )
            )

            # Generate plot
            duration = min(20, len(raw.times) / raw.info['sfreq'])
            fig = plot_raw_eeg(raw, duration, start_offset=1.5, end_offset=1.5)

            # Save to bytes buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close(fig)

            return send_file(buf, mimetype='image/png')
        except Exception as e:
            return Response(f'Error generating plot: {str(e)}', status=500)

@app.route('/status', methods=['GET'])
def get_status():
    """Get current status and info about the data stream"""
    with data_lock:
        return jsonify({
            'data_available': latest_filtered_data is not None,
            'channels': latest_info.get('channels', []),
            'n_channels': latest_info.get('n_channels', 0),
            'sampling_rate': latest_info.get('sampling_rate', 0),
            'data_shape': latest_filtered_data.shape if latest_filtered_data is not None else None
        })

def data_collection_thread(scale_factor, picks, max_seconds, pull_interval):
    """Background thread for collecting and filtering EEG data"""
    global latest_filtered_data, latest_raw_data, latest_info, new_samples_count

    # Resolve and connect to LSL stream
    streams = resolve_streams()
    print(f"Found {len(streams)} streams")

    eeg_streams = [stream for stream in streams if stream.stype.upper() == 'EEG']
    if not eeg_streams:
        raise ValueError('No EEG streams found')

    if len(eeg_streams) > 1:
        raise ValueError('Multiple EEG streams found, TODO: implement selection')

    inlet = StreamInlet(eeg_streams[0])
    inlet.open_stream()

    stream_info = inlet.get_sinfo()
    sampling_rate = stream_info.sfreq
    n_channels = stream_info.n_channels
    names = get_channels_from_xml_desc(stream_info.desc)

    print(f"Found {n_channels} channels: {names}")
    print(f"Sampling rate: {sampling_rate}")
    print(f"Units: {stream_info.get_channel_units()}")

    # Store stream info
    with data_lock:
        latest_info = {
            'channels': names,
            'n_channels': n_channels,
            'sampling_rate': sampling_rate
        }

    all_data = None

    while True:
        # Pull data from the LSL stream
        data, _ = inlet.pull_chunk()

        if all_data is None:
            all_data = data.copy()
        else:
            all_data = np.concatenate((all_data, data), axis=0)

        # Keep only the last max_seconds of data
        if len(all_data) > int(sampling_rate) * max_seconds:
            all_data = all_data[-int(sampling_rate) * max_seconds:]

        if len(data) > 0 and len(all_data) > sampling_rate * 3:
            # Store raw data
            raw_data_transposed = all_data.T * scale_factor

            # Create MNE Raw object and apply filtering
            raw = mne.io.RawArray(
                raw_data_transposed.copy(),
                mne.create_info(names, sampling_rate, ch_types='eeg')
            )
            filter_and_drop_dead_channels(raw, None)

            if picks:
                raw.pick_channels(picks)

            filtered_data = raw.get_data()

            # Update global variables with thread safety
            with data_lock:
                latest_filtered_data = filtered_data
                latest_raw_data = raw_data_transposed
                latest_info['channels'] = raw.ch_names
                latest_info['n_channels'] = len(raw.ch_names)
                new_samples_count += len(data)

            print(f"Updated data: {filtered_data.shape}")

        time.sleep(pull_interval)

ws_clients = set()

async def ws_handler(websocket):
    ws_clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        ws_clients.discard(websocket)

async def ws_broadcast(interval):
    while True:
        if ws_clients:
            with data_lock:
                if latest_filtered_data is not None:
                    channels = latest_info.get('channels', [])
                    sr = latest_info.get('sampling_rate', 0)
                    trim = int(sr * trim_seconds)
                    n = latest_filtered_data.shape[1]
                    if n > 2 * trim:
                        sliced = latest_filtered_data[:, trim:n - trim]
                    else:
                        sliced = latest_filtered_data
                    msg = json.dumps({
                        'data': {ch: sliced[i].tolist() for i, ch in enumerate(channels)},
                        'sampling_rate': sr,
                    })
            if latest_filtered_data is not None:
                websockets.broadcast(ws_clients, msg)
        await asyncio.sleep(interval)

def ws_server_thread(host, port, interval):
    async def run():
        async with websockets.serve(ws_handler, host, port):
            await ws_broadcast(interval)
    asyncio.run(run())

def osc_sender_thread(host, port, interval):
    """Send per-channel latest EEG values as OSC messages to TouchDesigner"""
    broadcast = host.endswith('.255') or host == '255.255.255.255'
    client = SimpleUDPClient(host, port, allow_broadcast=broadcast)
    print(f"OSC sender → {host}:{port}{' (broadcast)' if broadcast else ''}")
    while True:
        with data_lock:
            if latest_filtered_data is not None:
                channels = latest_info.get('channels', [])
                sr = latest_info.get('sampling_rate', 0)
                trim = int(sr * trim_seconds)
                n = latest_filtered_data.shape[1]
                end = n - trim if n > 2 * trim else n
                # Latest sample per channel
                snapshot = [(ch, float(latest_filtered_data[i, end - 1]))
                            for i, ch in enumerate(channels)
                            if end > 0]
            else:
                snapshot = []
        for ch, val in snapshot:
            client.send_message(f"/eeg/{ch}", val)
        time.sleep(interval)

def flask_server_thread(host, port):
    """Run Flask server in a background thread"""
    app.run(host=host, port=port, threaded=True)

def main():
    parser = argparse.ArgumentParser(
        prog='EEG_http_server',
        description='Serve filtered EEG data over HTTP and display filtered data plot on screen'
    )

    parser.add_argument('--convert-uv', action='store_true', help='Convert uV to V')
    parser.add_argument('--picks', type=str, default=None, help='Comma or space-separated list of channels to use')
    parser.add_argument('--port', type=int, default=5000, help='HTTP port (default: 5000)')
    parser.add_argument('--ws-port', type=int, default=5001, help='WebSocket port (default: 5001)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--max-seconds', type=int, default=20, help='Maximum seconds of data to keep in buffer')
    parser.add_argument('--pull-interval', type=float, default=0.02, help='Interval in seconds between LSL data pulls (default: 0.3)')
    parser.add_argument('--trim-seconds', type=float, default=3, help='Seconds to trim from each end to remove filter ringing (default: 3)')
    parser.add_argument('--osc-host', type=str, default=None, help='OSC target host (e.g. 127.0.0.1). If unset, OSC is disabled.')
    parser.add_argument('--osc-port', type=int, default=7000, help='OSC target port (default: 7000, TouchDesigner default)')
    parser.add_argument('--osc-interval', type=float, default=0.02, help='Interval in seconds between OSC sends (default: 0.02 = 50Hz)')

    args = parser.parse_args()
    picks = parse_picks(args.picks)
    scale_factor = 1e-6 if args.convert_uv else 1.0

    global trim_seconds
    trim_seconds = args.trim_seconds

    # Start data collection thread
    collection_thread = threading.Thread(
        target=data_collection_thread,
        args=(scale_factor, picks, args.max_seconds, args.pull_interval),
        daemon=True
    )
    collection_thread.start()

    # Start Flask server in background thread
    server_thread = threading.Thread(
        target=flask_server_thread,
        args=(args.host, args.port),
        daemon=True
    )
    server_thread.start()

    # Start WebSocket server in background thread
    ws_thread = threading.Thread(
        target=ws_server_thread,
        args=(args.host, args.ws_port, args.pull_interval),
        daemon=True
    )
    ws_thread.start()

    if args.osc_host:
        osc_thread = threading.Thread(
            target=osc_sender_thread,
            args=(args.osc_host, args.osc_port, args.osc_interval),
            daemon=True
        )
        osc_thread.start()

    print(f"\nStarting HTTP server on {args.host}:{args.port}")
    print(f"Starting WebSocket server on {args.host}:{args.ws_port}")
    print("\nAvailable endpoints:")
    print(f"  - http://{args.host}:{args.port}/status - Get stream status")
    print(f"  - http://{args.host}:{args.port}/data/filtered - Get filtered EEG data (JSON)")
    print(f"  - http://{args.host}:{args.port}/data/raw - Get raw EEG data (JSON)")
    print(f"  - http://{args.host}:{args.port}/plot/raw - Get raw EEG plot (PNG)")
    print("\nStarting pygame display for filtered data...\n")

    # Initialize pygame for display
    pygame.init()
    pygame.display.init()
    screen_width, screen_height = 1200, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('EEG Filtered Data - HTTP Server')

    # Display loop
    running = True
    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get latest filtered data and plot it
        with data_lock:
            if latest_filtered_data is not None and len(latest_info.get('channels', [])) > 0:
                try:
                    # Create MNE Raw object from filtered data
                    raw = mne.io.RawArray(
                        latest_filtered_data,
                        mne.create_info(
                            latest_info['channels'],
                            latest_info['sampling_rate'],
                            ch_types='eeg'
                        )
                    )

                    # Plot filtered data
                    if len(raw.times) > 3 * latest_info['sampling_rate']:
                        duration = min(args.max_seconds, len(raw.times) / raw.info['sfreq'])
                        fig = plot_raw_eeg(raw, duration, start_offset=1.5, end_offset=1.5)

                        # Convert matplotlib figure to pygame surface
                        plot_surface = plot_to_pygame(agg, fig)

                        # Clear screen and draw plot
                        screen.fill((255, 255, 255))
                        screen.blit(plot_surface, (20, 20))
                        pygame.display.flip()

                        plt.close(fig)
                except Exception as e:
                    print(f"Error displaying plot: {e}")

        time.sleep(args.pull_interval)

    pygame.quit()
    print("\nShutting down...")

if __name__ == '__main__':
    main()
