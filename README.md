# EEG live streaming

Stream EEG data from an LSL source (hardware or replayed file) and expose it over HTTP, WebSocket, and OSC for real-time visualization.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Replaying example data

`replay_xdf.py` loads a `.xdf` recording and republishes it as an LSL stream, which `EEG_server.py` then picks up like any live source.

```bash
# terminal 1 — play back a recording
python replay_xdf.py path/to/recording.xdf

# terminal 2 — serve it
python EEG_server.py
```

## Running the server

```bash
python EEG_server.py [options]
```

## HTTP

Plain GET endpoints, JSON or PNG:

```text
GET http://localhost:5000/status          # stream info
GET http://localhost:5000/data/filtered   # data (JSON)
```

JSON shape:

```json
{
  "data": { "O1": [...samples...], "O2": [...], ... },
  "sampling_rate": 250,
  "timestamp": 1712345678.9,
  "sample_timestamps": [...],
  "n_samples": 3500
}
```

Each request returns the full current buffer (minus the trim window).

## WebSocket

```text
ws://localhost:5001
```

Pushes only **new samples** since the last broadcast. Client accumulates them into a rolling buffer.

Message shape:

```json
{
  "data": { "O1": [...new samples...], "O2": [...], ... },
  "sampling_rate": 250
}
```

Browser example:

```javascript
const ws = new WebSocket("ws://localhost:5001");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  // msg.data[channel] = array of fresh samples
};
```

## OSC (TouchDesigner, Max, etc.)

Opt-in — pass `--osc-host` to enable:

```bash
# send to same machine
python EEG_server.py --osc-host 127.0.0.1 --osc-port 7000

# broadcast to whole LAN
python EEG_server.py --osc-host 255.255.255.255

# broadcast to a specific subnet
python EEG_server.py --osc-host 192.168.1.255
```

Sends one message per channel at `--osc-interval` (default 50 Hz):

```text
/eeg/<channel_name>  <latest_float_value>
```

In **TouchDesigner**: drop an `OSC In CHOP`, set **Network Port** to `7000`. You'll see one CHOP channel per electrode, each carrying the latest sample value.

## Typical workflow

1. Plug in your EEG headset (or start `replay_xdf.py`)
2. Run `python EEG_server.py --osc-host 127.0.0.1`
3. Open a browser / TouchDesigner / Hydra and consume whichever endpoint fits
