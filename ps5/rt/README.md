# Our own PS5 title runtime

The PS5 title is GPL-3.0 today only because of its runtime: the start code,
libc stand-ins, heap, threads, AGC link stubs and `libc.prx` come from
PS5_Vulkan and the PS5 payload SDK (see [LICENSING.md](../../LICENSING.md)).
This folder replaces them with our own code, so the title can ship under the
same license as the engine.

`ps5/build.sh` picks the runtime with `TS5_RUNTIME`:

```sh
ps5/build.sh                    # gpl: PS5_Vulkan's runtime (the default)
TS5_RUNTIME=own ps5/build.sh    # ours, from this folder
```

The build tools stay as they are: the prospero clang wrapper, LLD, and
`ps5-native-tool`, which converts and signs the title. They don't end up in
the title. RADV (Mesa, MIT) stays too.

## Clean room

Everything here is written from `SPEC.md`, a description of what the runtime
has to do (the interfaces, symbols and behaviour the console and RADV
expect), written separately from the code. Whoever writes code in this
folder does not read PS5_Vulkan, PS5_PayloadSDK or other GPL PS5 sources, or
the GPL-derived files elsewhere in `ps5/`. Code taken from elsewhere is only
code with a permissive license, listed in [NOTICE.md](../../NOTICE.md):

- dlmalloc (public domain), for the heap
- musl's regex (MIT)

This folder is our own code under PolyForm Small Business 1.0.0
([LICENSE.md](../../LICENSE.md)).

## Status

Scaffolding only: `runtime.sh` has the hooks `build.sh` calls, and each one
stops the build until it's written. `own` becomes the default once a title
built with it launches cleanly on the console, passes the GPU check, and
matches the GPL build's pp512/tg128 speeds.
