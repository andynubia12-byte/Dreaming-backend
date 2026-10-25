#!/usr/bin/env bash
# 一键启动：启动本地服务并打开浏览器
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v node >/dev/null 2>&1; then
  echo "未找到 node，请先安装 Node.js 18+（推荐 20+）。"
  exit 1
fi

exec node server.mjs --open
