// Feedback reports (Linux): what went wrong, in the user's words, with the
// facts that help fix it (version, system, settings, the log's last lines),
// cleaned of anything personal or secret on this machine before it is shown
// or sent. ts5-tui's /feedback runs it, shows the user the exact text, and
// sends it only when the user says so.
//
//   ts5 feedback prepare [--summary TEXT] [--log FILE | --journal UNIT]
//                        [--settings FILE] [--chat FILE] [--no-logs]
//                        the report, cleaned, on stdout
//   ts5 feedback scrub [--keep-emails]
//                        stdin cleaned to stdout (the same cleaning)
//   ts5 feedback send FILE
//                        cleans FILE once more and sends it; JSON on stdout:
//                        {"ok":true,"id":"..."} or {"error":"..."}
//
// Cleaning takes out: private keys; ts5 access keys; GitHub, Tailscale,
// Hugging Face, Tavily and other API tokens; Bearer and Authorization
// values; settings named like a key, token, secret or password; passwords
// in addresses; email addresses, except in what the user wrote (so they
// can be answered); IP addresses other than 127.0.0.1 (a
// tailnet's 100.x and a home network's addresses say which they were);
// MAC addresses; Tailscale names (*.ts.net); this machine's name; the
// user's login name and home folder. A chat goes in only with --chat.
//
// Reports go to TokenStation's feedback service (a Cloudflare Worker that
// keeps them for the developers to read; no GitHub account or token is
// involved). TS5_FEEDBACK_URL points at another one, for tests.
#pragma once
#include <string>
#include <vector>

namespace ts5 {

// Words to take out besides the patterns: this machine's name, the user's
// login names, the home folder. feedback_scrub_words() reads them here.
struct ScrubWords {
    std::vector<std::string> users;  // login names
    std::string host;                // this machine's name
    std::string home;                // home folder, such as /home/ps5
};
ScrubWords feedback_scrub_words();

// keep_emails: the user's own words, where an email address is how they
// can be answered.
std::string feedback_scrub(const std::string &text, const ScrubWords &w, bool keep_emails);

int feedback_main(int argc, char **argv);

}  // namespace ts5
