# xterm.js, vendored

[xterm.js](https://github.com/xtermjs/xterm.js) 5.5.0 (npm `@xterm/xterm`,
tarball SHA-256 `bd954fa721872170188cc5d7e83e88db3c83c9a18a4e8d24c2783d26491f59d2`) and its fit addon 0.10.0 (npm `@xterm/addon-fit`,
SHA-256 `917ac44972453d5eed52edc1e50260c76398ce48cf2290c2e60671102bba0b33`). MIT: [LICENSE](LICENSE) and
[LICENSE-addon-fit](LICENSE-addon-fit). NOTICE.md at the top of this repo lists them.

ts5 serve's browser terminal (`engine/src/web_serve.cpp`, Linux) shows
ts5-tui in a web page with them; they are built into `ts5` and served from
`/ts5/web/`. Unchanged; `scripts/vendor-xterm.sh` copies them.

Kept: `lib/xterm.js`, `css/xterm.css`, the addon's `lib/addon-fit.js`, the
licenses. Left out: the source maps, the type definitions, the sources.
