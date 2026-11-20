# The noise server, for a Linux host that is always on.
#
#   docker build -t noisemachine .
#   docker run --rm --network host noisemachine
#
# Host networking is required, which makes this Linux only:
#
# - The server tells a speaker which address to fetch the stream from, and it
#   uses its own. On Docker's default bridge that is an address only Docker
#   knows (172.17.x.x), and nothing inside the container can find the host's.
# - Speakers are found by room name with a multicast search, which the bridge
#   does not pass to the LAN.
#
# Publishing the port (-p 8000:8000) fixes neither. Docker Desktop on macOS
# and Windows runs containers in a virtual machine, so host networking there
# is the virtual machine's network, not the LAN: run serve_noise.py directly
# on those. A network that puts the container on the LAN (macvlan, an incus
# app) works like host networking.

FROM python:3.13-slim AS build
# pyflac has no wheel: it is compiled here, and the compiler stays behind.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libffi-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /venv
COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim
COPY --from=build /venv /venv
WORKDIR /app
COPY control.py cover.py mdns.py noise.py serve_noise.py sonos_play.py streams.py ./
USER nobody
EXPOSE 8000
ENTRYPOINT ["/venv/bin/python", "serve_noise.py"]
