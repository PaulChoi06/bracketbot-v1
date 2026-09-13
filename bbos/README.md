# bbos

## Quickstart
1. Clone

```
git clone https://github.com/raghavauppuluri13/bbos.git
```

2. Install

```
cd bbos
```

```
# takes around 5-10min
./install
```

3. Calibrate

```
calibrate drive
```

4. Run
Open https://<HOSTNAME>.local:8001 to see the dashboard and toggle apps

OR: 

```
uv run apps/<apps>.py
```

## Debug
- Run `clean` to reset bracketbot to a fresh state
- `debug-daemons [daemon_name]` to view logs for all daemons or a specific daemon
- `debug-service <service>` to debug the installed system services
