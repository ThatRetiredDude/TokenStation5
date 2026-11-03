#!/bin/sh
# What Steam starts (Steamworks > Installation > General: executable
# "launch.sh"). Steam runs it inside Steam Linux Runtime 3.0 (sniper), a
# Debian 11 container, which is what the Steam build is built against.
#
# Models, settings, keys and chats go in ~/.local/share/tokenstation, as with
# the .deb and the portable build, never in the Steam folder: Steam replaces
# that folder on every update and deletes it on uninstall.
here=$(cd -- "$(dirname -- "$0")" && pwd)
export TS5_STEAM=1

# Big Picture, Game Mode and the Steam Deck: fill the screen. On the desktop
# the app is full screen too; F11 makes it a window.
# Steam Input hands the app a virtual controller, which SDL reads like any
# other: nothing to set here for the DualSense, the Deck or an Xbox pad.

exec "$here/bin/tokenstation" "$@"
