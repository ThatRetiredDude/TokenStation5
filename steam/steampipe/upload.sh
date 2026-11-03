#!/usr/bin/env bash
# Uploads build/steam/linux/depot (steam/linux/build-depot.sh) to Steam as a new
# build, and sets it live on a beta branch, never on the default one: that
# stays a click in Steamworks (App Admin > SteamPipe > Builds), after testing.
#
#   STEAM_APP_ID=1234560 STEAM_DEPOT_ID=1234561 STEAM_USER=builder \
#       steam/steampipe/upload.sh [BRANCH]
#
# BRANCH defaults to "beta"; make that branch first in Steamworks
# (SteamPipe > Builds > Manage Branches). Use "" to upload without setting it
# live anywhere. Needs steamcmd (https://developer.valvesoftware.com/wiki/SteamCMD)
# on the PATH or in STEAMCMD. The first login asks for the Steam Guard code;
# use a separate Steam account that only has build rights, not your own.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$here/../.." && pwd)"
app=${STEAM_APP_ID:?your app ID, from Steamworks}
depotid=${STEAM_DEPOT_ID:?the Linux depot ID, from Steamworks > SteamPipe > Depots}
user=${STEAM_USER:?the Steam account that uploads builds}
branch=${1-beta}
[ "$branch" = default ] && { echo "upload: set the default branch live in Steamworks, not from here" >&2; exit 2; }
steamcmd=${STEAMCMD:-steamcmd}
command -v "$steamcmd" > /dev/null || { echo "upload: steamcmd was not found (set STEAMCMD)" >&2; exit 2; }

depot="$root/build/steam/linux/depot"
[ -x "$depot/launch.sh" ] || { echo "upload: no depot; run steam/linux/build-depot.sh first" >&2; exit 2; }
version=$(cat "$root/build/steam/linux/version")
commit=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)
work="$root/build/steam/steampipe"
rm -rf "$work"
mkdir -p "$work/output"
fill() {
    sed -e "s|@APPID@|$app|g" -e "s|@DEPOTID@|$depotid|g" -e "s|@VERSION@|$version|g" \
        -e "s|@COMMIT@|$commit|g" -e "s|@DEPOT@|$depot|g" -e "s|@OUTPUT@|$work/output|g" \
        -e "s|@BRANCH@|$branch|g" "$1"
}
fill "$here/app_build.vdf.in" > "$work/app_build.vdf"
fill "$here/depot_build.vdf.in" > "$work/depot_build.vdf"
# No branch: no SetLive line at all.
[ -n "$branch" ] || sed -i '/"SetLive"/d' "$work/app_build.vdf"

"$steamcmd" +login "$user" +run_app_build "$work/app_build.vdf" +quit
