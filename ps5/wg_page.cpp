#include "wg_page.h"

#include <cstdlib>

namespace app {

namespace {

std::string esc(const std::string &s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&#39;"; break;
            default: out += c;
        }
    }
    return out;
}

// A form field's value from an application/x-www-form-urlencoded body.
std::string field(const std::string &body, const std::string &name) {
    size_t at = 0;
    while (at <= body.size()) {
        size_t amp = body.find('&', at);
        if (amp == std::string::npos) amp = body.size();
        const std::string pair = body.substr(at, amp - at);
        const size_t eq = pair.find('=');
        if (eq != std::string::npos && pair.compare(0, eq, name) == 0 && eq == name.size()) {
            std::string out;
            for (size_t i = eq + 1; i < pair.size(); i++) {
                if (pair[i] == '+') out += ' ';
                else if (pair[i] == '%' && i + 2 < pair.size()) {
                    out += (char)strtol(pair.substr(i + 1, 2).c_str(), nullptr, 16);
                    i += 2;
                } else out += pair[i];
            }
            return out;
        }
        at = amp + 1;
    }
    return "";
}

std::string page(const std::string &main) {
    return "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
           "<title>TokenStation5: WireGuard</title><style>"
           "body{font:16px/1.5 system-ui,sans-serif;max-width:40em;margin:0 auto;padding:16px;"
           "background:#101114;color:#e8e8ea}"
           "h1{font-size:1.4em}h2{font-size:1.1em;margin-top:1.6em}a{color:#7cb4ff}"
           "input{font:inherit;width:100%;box-sizing:border-box;padding:.5em;margin:.2em 0 1em;"
           "background:#1b1d22;color:inherit;border:1px solid #3a3d45;border-radius:6px}"
           "button{font:inherit;padding:.5em 1.2em;border:0;border-radius:6px;background:#3d7be0;color:#fff}"
           "pre{background:#1b1d22;padding:12px;border-radius:6px;overflow-x:auto;white-space:pre}"
           ".bad{color:#ff8a80}.note{color:#a0a3ab}"
           "</style></head><body>" +
           main + "</body></html>";
}

std::string form(const std::string &problem, const std::string &name, const std::string &key) {
    std::string m =
        "<h1>Add a device to this PS5's WireGuard</h1>"
        "<p>Do this on the device itself (a computer, phone or tablet) while it is on your home network.</p>"
        "<ol><li>Install WireGuard on this device, from <a href=\"https://www.wireguard.com/install/\">"
        "wireguard.com/install</a> or its app store.</li>"
        "<li>In WireGuard, make a new tunnel: <b>Add empty tunnel</b> on a computer, <b>Create from scratch</b> "
        "on a phone or tablet. The app makes this device's own key pair. Its private key stays in the app: "
        "the PS5 never sees it.</li>"
        "<li>Copy the <b>public key</b> the app shows, and paste it here with a name for this device.</li></ol>";
    if (!problem.empty()) m += "<p class=\"bad\">" + esc(problem) + "</p>";
    m += "<form method=\"post\" action=\"/wireguard\">"
         "<label>Name<input name=\"name\" maxlength=\"40\" placeholder=\"Laptop\" value=\"" +
         esc(name) +
         "\" required></label>"
         "<label>This device's public key<input name=\"key\" maxlength=\"60\" autocomplete=\"off\" "
         "spellcheck=\"false\" placeholder=\"44 characters, ending with =\" value=\"" +
         esc(key) +
         "\" required></label>"
         "<button>Ask to join</button></form>"
         "<p class=\"note\">Nothing joins until someone approves it on the TV, in Settings on the PS5.</p>";
    return page(m);
}

}  // namespace

bool wg_page(const std::string &method, const std::string &path, const std::string &body, const std::string &client,
             const WgPageInfo &info, WgStore &store, ts5::Server::Page &out) {
    if (path != "/wireguard" && path != "/wireguard/") return false;
    if (client.rfind("10.55.0.", 0) == 0) {
        out.status = 403;
        out.body = page("<h1>Not from the tunnel</h1><p>Add devices from your home network, not through "
                        "WireGuard.</p>");
        return true;
    }
    if (!info.on || info.public_key.empty()) {
        out.body = page("<h1>WireGuard is off</h1><p>Turn it on on the PS5 first: Settings, WireGuard. Then "
                        "load this page again.</p>");
        return true;
    }
    if (method != "POST") {
        out.body = form("", "", "");
        return true;
    }
    const std::string name = field(body, "name"), key = field(body, "key");
    WgRequest req;
    std::string err;
    if (!store.ask(name, key, client, req, err)) {
        out.status = 400;
        out.body = form(err, name, key);
        return true;
    }
    const std::string port = std::to_string(info.port);
    std::string m =
        "<h1>Now approve " + esc(req.name) +
        " on the TV</h1>"
        "<p>On the PS5: Settings, WireGuard. Press cross on <b>" +
        esc(req.name) +
        " asks to join</b>. Until then this device can't connect. The request lapses after 10 minutes.</p>"
        "<h2>Then finish the tunnel on this device</h2>"
        "<p>On a computer, the tunnel's text already has its <code>PrivateKey</code> line: add the rest under it, "
        "so it reads like this.</p><pre id=\"conf\">[Interface]\n"
        "PrivateKey = (the key your app made: leave it as it is)\n"
        "Address = " +
        req.address +
        "/32\n\n[Peer]\nPublicKey = " + esc(info.public_key) +
        "\nAllowedIPs = 10.55.0.1/32\nEndpoint = <span id=\"home\">YOUR-HOME-ADDRESS</span>:" + port +
        "\nPersistentKeepalive = 25</pre>"
        "<p>On a phone or tablet, fill in the same: Addresses <code>" +
        req.address +
        "/32</code>; then Add peer: Public key as above, Allowed IPs <code>10.55.0.1/32</code>, Endpoint, and "
        "Persistent keepalive 25.</p>"
        "<p><code>AllowedIPs</code> is the PS5 alone, so the tunnel reaches TokenStation5 and nothing else on "
        "your network.</p>"
        "<h2>Endpoint: your home's address on the internet</h2>"
        "<p>Your router's public address (or a dynamic DNS name for it, if your router has one), and on the "
        "router, forward UDP port " +
        port +
        " to this PS5. WireGuard doesn't answer anything without a key it knows, so the open port gives "
        "nothing away.</p>"
        "<p><button id=\"look\" type=\"button\">Look up my home's public address</button> "
        "<span class=\"note\">(asks api.ipify.org from this device)</span></p>"
        "<p class=\"note\">Some internet providers share one public address among many homes (CGNAT), and then a "
        "forwarded port can't be reached. If the tunnel never connects, ask your provider or use Tailscale "
        "instead.</p>"
        "<h2>In your apps</h2><p>Away from home, with the tunnel on, the base URL is <code>http://10.55.0.1:" +
        std::to_string(info.api_port) +
        "/v1</code>. Use one of your API keys too.</p>"
        "<script>document.getElementById('look').onclick=function(){fetch('https://api.ipify.org?format=json')"
        ".then(function(r){return r.json()}).then(function(j){document.getElementById('home').textContent=j.ip})"
        ".catch(function(){document.getElementById('home').textContent='(could not look it up)'})};</script>";
    out.body = page(m);
    return true;
}

}  // namespace app
