#pragma once

namespace ts5 {

// A line for the log (stderr), written by a thread of its own, so the caller
// never waits on the log file: on the PS5 a write to it could take a second
// or more while the TV sat idle, and a Hugging Face search's lines made it
// take minutes (2026-10-24). Lines keep their order. The newline is added.
void log_line(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

// Waits (2 s at the most) for the queued lines to be written: before the
// title leaves, and at exit.
void log_drain();

}  // namespace ts5
