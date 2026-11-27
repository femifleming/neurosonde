#!/usr/bin/env bash
# Collects the built plug-ins, apps and the manual into dist/Neurosonde-<os>/<product>/
# usage: ci/package.sh <macOS|Windows|Linux> <build dir>
set -euo pipefail
OS="$1"; B="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/dist/Neurosonde-$OS"
rm -rf "$OUT"; mkdir -p "$OUT"

art() {   # artefact folder of a target (single- or multi-config generators)
    local tgt="$1"
    for d in "$B/${tgt}_artefacts/Release" "$B/${tgt}_artefacts"; do
        [ -d "$d/VST3" ] && { echo "$d"; return; }
    done
    echo "missing artefacts for $tgt" >&2; exit 1
}

product() {   # folder target file-name
    local name="$1" tgt="$2" file="$3"
    local A; A="$(art "$tgt")"
    local P="$OUT/$name"; mkdir -p "$P"
    cp -R "$A/VST3/$file.vst3" "$P/"
    case "$OS" in
        macOS)
            cp -R "$A/AU/$file.component" "$A/Standalone/$file.app" "$P/"
            for f in "$P"/*.app "$P"/*.component "$P"/*.vst3; do [ -e "$f" ] && codesign --force --deep -s - "$f"; done ;;
        Windows)
            cp "$A/Standalone/$file.exe" "$P/" ;;
        Linux)
            cp "$A/Standalone/$file" "$P/"
            [ -d "$A/LV2/$file.lv2" ] && cp -R "$A/LV2/$file.lv2" "$P/" ;;
    esac
}

product "Neurosonde" Neurosonde "Neurosonde"
product "Neurosonde Space" NeurosondeSpace "Neurosonde Space"
cp "$ROOT/README.md" "$ROOT/LICENSE" "$OUT/"
[ -f "$ROOT/docs/Neurosonde-Manual.pdf" ] && cp "$ROOT/docs/Neurosonde-Manual.pdf" "$OUT/"

cd "$ROOT/dist"
case "$OS" in
    macOS)   ditto -c -k --keepParent "Neurosonde-$OS" "Neurosonde-$OS.zip" ;;
    Windows) 7z a -tzip -r "Neurosonde-$OS.zip" "Neurosonde-$OS" > /dev/null ;;
    Linux)   tar -czf "Neurosonde-$OS.tar.gz" "Neurosonde-$OS" ;;
esac
ls -la "$ROOT/dist"
