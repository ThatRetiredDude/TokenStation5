// Sharing this machine with a user over Tailscale (`ts5 share`; ts5-tui's
// Access page runs it). Two ways, chosen per invite:
//   ps5      Tailscale's machine sharing: the user keeps their own
//            Tailscale account and reaches this machine only, nothing
//            else on the owner's tailnet (and the owner's devices can't
//            reach theirs)
//   tailnet  a plain invite to join the owner's tailnet: the user can
//            reach every device on it, unless the owner's own Tailscale
//            rules say otherwise
// Both are made with the Tailscale API (api.tailscale.com, or
// TS5_TAILSCALE_API) and an API access token the owner makes in
// Tailscale's admin console (Settings, Keys) and gives once
// (`ts5 share token set`, the token on stdin; ts5-tui's Connectors page).
// It is kept beside the keys (tailscale-api.key, 0600), never shown again.
// Each invite made goes to users/ID/tailscale.tsv (id, way, link, made), so
// removing a user (`ts5 users remove`) takes their invites back:
// a machine share ends; an invite to the tailnet not yet taken ends, one
// taken leaves the person on the tailnet (Tailscale's admin console,
// Users, removes them), which the answer says.
#pragma once
#include <string>
#include <vector>

namespace ts5 {

struct ShareInvite {
    std::string id;    // Tailscale's
    std::string way;   // "ps5" or "tailnet"
    std::string url;   // the invite's link
    long long made = 0;
};

// The token's file beside the keys' file.
std::string share_token_file(const std::string &keys_path);
// The invites made for a user (its id), from users/ID/tailscale.tsv.
std::vector<ShareInvite> share_invites(const std::string &keys_path, const std::string &user);
// Takes back every invite made for a user (its id) and forgets them. note:
// what is left for the owner to do, if anything ("" none). false with err
// when an invite couldn't be taken back (those are kept, to try again).
bool share_revoke(const std::string &keys_path, const std::string &user, std::string &note, std::string &err);

int share_main(int argc, char **argv);

}  // namespace ts5
