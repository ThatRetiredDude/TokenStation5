// The GitHub connector (Linux): ts5 signs in to GitHub as the TokenStation5
// GitHub App, with GitHub's device flow (a code the user enters at
// github.com/login/device, on a phone or anywhere), keeps the user's token,
// and calls GitHub's REST API with it. ts5-tui runs it: its Connectors page
// signs in and out, and the model's GitHub tool makes its calls through it.
//
//   ts5 github sign-in             a line each, as they come: "code WDJB-MJHT",
//                                  "url https://github.com/login/device",
//                                  "expires 900"; then, once the user said yes,
//                                  "signed-in LOGIN" and "installations N"; or
//                                  "error WHY" (also "denied", "expired")
//   ts5 github status [--repos]    JSON: signed in, as whom, the app's install
//                                  page; --repos asks GitHub which repositories
//                                  the app may use
//   ts5 github sign-out            forgets the token
//   ts5 github api METHOD PATH [BODY] [--lean]
//                                  a REST call (PATH such as /user/repos; BODY
//                                  JSON, or - for stdin): "status N" (and
//                                  "next PATH" when there are more pages), then
//                                  the answer. --lean: for a model (files as
//                                  text, without the API's addresses; a file
//                                  written with "text" instead of base64)
//
// The token is the app's user token: it reaches only the repositories the
// user installed the app on, with the app's permissions, and it lasts 8
// hours; ts5 renews it with its refresh token (6 months) when it is used.
// It is kept in FILE (TS5_GITHUB_FILE, else connectors/github.json in the
// data folder, beside access/keys.tsv), readable by its owner only, and is
// never printed or logged. The app is TokenStation5's own: its Client ID is
// public, built in (TS5_GITHUB_CLIENT_ID and TS5_GITHUB_APP, the app's name
// in its address, for another app; TS5_GITHUB_WEB and TS5_GITHUB_API point
// at a server of one's own, for tests).
#pragma once
#include <string>

namespace ts5 {

// The token's file.
std::string github_file();
// A GitHub answer made shorter for a model to read: a file's text instead
// of its base64, the API's own addresses left out, a user as its login.
// An answer that isn't JSON comes back as it was.
std::string github_lean(const std::string &body);

int github_main(int argc, char **argv);

}  // namespace ts5
