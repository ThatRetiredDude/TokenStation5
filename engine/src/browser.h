// ts5's browser for a model: `ts5 browser`.
//
// A model can't look at a screen, so a page is given to it as text with
// every link, text box and button numbered in place ("the Volga[2]",
// "[4: text box "Search"]", "[5: button "Go"]"), and the model acts on the
// numbers. The program keeps one page open and takes one request a line on
// its standard input, each a JSON object, answering each with one line:
//
//   {"do":"open","url":"https://..."}        opens a page
//   {"do":"click","n":5}                     follows link 5, presses button 5, ticks box 5
//   {"do":"type","n":4,"text":"...","enter":true}
//                                            writes into text box 4; enter (the default)
//                                            then sends its form as the Enter key does
//   {"do":"more"}                            the page's next part (a long page comes in parts)
//   {"do":"back"}                            the page before this one
//   {"do":"what","act":"click","n":5}        what a request would do, without doing it:
//                                            "line" says it in words for the user, "asks"
//                                            whether the user ought to be asked first
//                                            (anything but reading: a button, typing, a
//                                            page on this machine or the home network)
//   {"do":"status"}   {"do":"close"}
//
//   {"ok":true,"text":"...","url":"...","title":"...","mode":"text"}
//   {"ok":false,"text":"what went wrong, in words for the model"}
//
// How the page is fetched and run is the user's choice (--program):
//   text        ts5's own fetch and page reader: no scripts run, so a page
//               built by scripts shows little; links and forms sent with GET work.
//   a browser   Chromium (or Chrome, or any browser with its debugging
//               port) or Obscura, started without a window and driven over
//               the Chrome DevTools Protocol on 127.0.0.1: the page's
//               scripts run and any button works. Nothing else is needed:
//               ts5 speaks the protocol itself.
//   auto        the first of those that is installed, else text.
// Only http:// and https:// pages are opened. A password box is never
// numbered: only the user fills those in. The browser gets a profile folder
// of its own under ~/.cache/tokenstation, removed when the program ends,
// and ends with it; it is stopped after ten idle minutes and started again
// when it is next needed.
#pragma once

namespace ts5 {

// ts5 browser [--program auto|text|PATH] [--kind chromium|obscura] [--chars N] [--idle SECONDS]
int browser_main(int argc, char **argv);

}  // namespace ts5
