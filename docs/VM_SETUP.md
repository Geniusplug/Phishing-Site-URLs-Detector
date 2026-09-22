# Optional VM setup

## Windows Hyper-V / VMware / VirtualBox

Recommended guest: Ubuntu 24.04 LTS, 4 CPU, 8 GB RAM, 30 GB disk.

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline 100
```

The system is CPU-friendly. No GPU is required.
