#!/usr/bin/env bash
# Builds the G2P (ps5/tts) on Linux with its lexicon linked in, as the title
# would, and runs its host test: tricky inputs against expected phonemes,
# the chat cleaner and sentence splitter, and every output checked against
# Kokoro's vocabulary. With sanitizers unless G2P_SANITIZE=0.
#
#   ps5/g2p-test.sh                      # the checks
#   ps5/g2p-test.sh --lines < text.txt   # phonemes for each line
#   ps5/g2p-test.sh --words < text.txt   # each word: text, phonemes, source
#   ps5/g2p-test.sh --lts < words.txt    # the letter-to-sound rules alone
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work="$root/build/g2p-test"
mkdir -p "$work"
flags=(-std=c++17 -O2 -g -Wall -Wextra -pedantic)
bin="$work/g2p_test"
if [ "${G2P_SANITIZE:-1}" = 1 ]; then
    flags+=(-fsanitize=address,undefined -fno-sanitize-recover=all)
    bin="$work/g2p_test_sanitized"
fi
src=("$root"/ps5/tts/{g2p,text,numbers,spelling,g2p_test}.cpp)
newest=$(ls -t "${src[@]}" "$root"/ps5/tts/*.h "$root"/ps5/tts/*.inc "$root/ps5/tts/lexicon.bin" "$0" | head -1)
if [ ! -x "$bin" ] || [ "$newest" -nt "$bin" ]; then
    "${CXX:-c++}" "${flags[@]}" -I "$root/ps5/tts" "${src[@]}" \
        -x assembler-with-cpp -Wa,-I"$root/ps5/tts" "$root/ps5/tts/lexicon_data.S" -x none -o "$bin"
fi
exec "$bin" "$@"
