#!/bin/sh
set -eu

if [ -n "${KIMI_SHARE_DIR:-}" ]; then
  mkdir -p "${KIMI_SHARE_DIR}"
  if [ -d /run/kimi-seed ] && [ -z "$(ls -A "${KIMI_SHARE_DIR}" 2>/dev/null || true)" ]; then
    cp -R /run/kimi-seed/. "${KIMI_SHARE_DIR}/"
  fi
fi

if [ -d /run/claude-seed ] && [ -z "$(ls -A /home/agent 2>/dev/null || true)" ]; then
  cp -R /run/claude-seed/. /home/agent/
fi

exec "$@"
