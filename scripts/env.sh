# Source this: . scripts/env.sh
# Console address and the ports the jailbreak's payloads listen on.
export PS5_HOST=${PS5_HOST:-192.168.1.50}
export PS5_FTP_PORT=${PS5_FTP_PORT:-2121}
export PS5_API_PORT=${PS5_API_PORT:-8081}
# Where models live on the console (PS5LM's library folder).
export PS5_MODEL_DIR=${PS5_MODEL_DIR:-/data/PS5LM/models}
# Local download cache for GGUFs.
export TS5_MODEL_CACHE=${TS5_MODEL_CACHE:-$HOME/.cache/tokenstation5/models}
