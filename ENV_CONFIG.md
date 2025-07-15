# TRELLIS Environment Variable Configuration

## Overview

The TRELLIS apps now support environment variable configuration for Gradio server settings.

## Environment Variables

### Gradio Server Configuration
These are handled directly in `app.py` and `app_text.py`:

- `GRADIO_SERVER_NAME` (default: `0.0.0.0`) - Server binding address
- `GRADIO_SERVER_PORT` (default: `7860`) - Server port  
- `GRADIO_SHARE` (default: `false`) - Enable public sharing

## Usage

```bash
# Custom configuration
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=8080
export GRADIO_SHARE=true
python3 app.py
```
