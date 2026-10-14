# Our own PS5 title runtime, for ps5/build.sh with TS5_RUNTIME=own.
#
# build.sh sources this file and calls these hooks in place of the GPL
# runtime's pieces. Each one is written from ps5/rt/SPEC.md alone (see
# README.md); until a hook is written it stops the build and says so.
#
#   rt_check                 our sources and inputs are all there
#   rt_crt OBJ               the start code, compiled to OBJ
#   rt_stubs DIR             libSceAgc.so and libSceAgcDriver.so link stubs
#   rt_link_recipe ARCHIVE   sets radv_linker_script, radv_link_flags,
#                            radv_link_inputs and symbols_map
#   rt_libc_prx DEST         the runtime module the title carries
#
# build.sh's cc, root, sdk and out are in scope.

rt="$root/ps5/rt"

rt_todo() {
    echo "build: TS5_RUNTIME=own: $1 is not written yet (ps5/rt/README.md)" >&2
    exit 2
}

rt_check() { rt_todo "the runtime"; }
rt_crt() { rt_todo "the start code"; }
rt_stubs() { rt_todo "the AGC link stubs"; }
rt_link_recipe() { rt_todo "the link recipe"; }
rt_libc_prx() { rt_todo "libc.prx"; }
