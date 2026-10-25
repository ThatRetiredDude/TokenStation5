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

// s as a JSON string, quotes and all.
std::string json_str(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\', out += c;
        else if ((unsigned char)c < 0x20) out += ' ';
        else if (c == '<') out += "\\u003c";  // never a </script> in a page
        else out += c;
    }
    return out + "\"";
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
           "textarea{font:13px/1.4 ui-monospace,monospace;width:100%;box-sizing:border-box;padding:.5em;"
           "background:#1b1d22;color:inherit;border:1px solid #3a3d45;border-radius:6px}"
           "details{margin-top:2em}summary{cursor:pointer;color:#7cb4ff}[hidden]{display:none}"
           ".bad{color:#ff8a80}.note{color:#a0a3ab}.ok{color:#7ee2a8}"
           "</style></head><body>" +
           main + "</body></html>";
}

// The other way: the device's WireGuard app makes the key pair, and its
// public key is pasted here. Posts to /wireguard.
std::string manual(const std::string &problem, const std::string &name, const std::string &key) {
    std::string m =
        "<ol><li>In WireGuard, make a new tunnel: <b>Add empty tunnel</b> on a computer, <b>Create from scratch</b> "
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
         "<button>Ask to join</button></form>";
    return m;
}

std::string form(const std::string &problem, const std::string &name, const std::string &key) {
    return page("<h1>Add a device to this PS5's WireGuard</h1>"
                "<p>Do this on the device itself (a computer, phone or tablet) while it is on your home network. "
                "Install WireGuard on it first, from <a href=\"https://www.wireguard.com/install/\">"
                "wireguard.com/install</a> or its app store.</p>" +
                manual(problem, name, key) +
                "<p class=\"note\">Nothing joins until someone approves it on the TV, in Settings on the PS5.</p>");
}

// X25519 (RFC 7748) for the page, so the browser makes the device's key pair:
// crypto.subtle needs HTTPS, and this page is plain HTTP on the home network.
// x25519(32 bytes) is that scalar times the base point, as 32 bytes.
const char *kX25519 =
    R"JS(function x25519(k){var P=(1n<<255n)-19n,m=function(a){a%=P;return a<0n?a+P:a},s=0n,i,
x1=9n,x2=1n,z2=0n,x3=9n,z3=1n,sw=0n,t,A,AA,B,BB,E,C,D,DA,CB;
for(i=31;i>=0;i--)s=(s<<8n)|BigInt(k[i]);
for(i=254;i>=0;i--){var b=(s>>BigInt(i))&1n;sw^=b;if(sw){t=x2;x2=x3;x3=t;t=z2;z2=z3;z3=t}sw=b;
A=m(x2+z2);AA=m(A*A);B=m(x2-z2);BB=m(B*B);E=m(AA-BB);C=m(x3+z3);D=m(x3-z3);DA=m(D*A);CB=m(C*B);
x3=m((DA+CB)*(DA+CB));z3=m(x1*m((DA-CB)*(DA-CB)));x2=m(AA*BB);z2=m(E*(AA+121665n*E))}
if(sw){x2=x3;z2=z3}
var r=1n,e=P-2n,q=z2;while(e>0n){if(e&1n)r=m(r*q);q=m(q*q);e>>=1n}
var u=m(x2*r),o=[];for(i=0;i<32;i++){o.push(Number(u&255n));u>>=8n}return o})JS";

// The page with one tap: the browser makes the key pair, keeps the private
// key (in this tab only), asks to join with the public one, waits for the
// TV and then hands over a finished tunnel file.
std::string one_tap(const WgPageInfo &info) {
    const std::string port = std::to_string(info.port), api = std::to_string(info.api_port);
    std::string m =
        "<h1>Add this device to the PS5's WireGuard</h1>"
        "<p>Do this on the device itself (a computer, phone or tablet) while it is on your home network. "
        "Install WireGuard on it first, from <a href=\"https://www.wireguard.com/install/\">wireguard.com/install</a> "
        "or its app store.</p>"
        "<div id=\"s1\"><label>A name for this device<input id=\"name\" maxlength=\"40\" placeholder=\"Laptop\" "
        "autocomplete=\"off\"></label><button id=\"ask\" type=\"button\">Ask to join</button>"
        "<p class=\"note\">This page makes the device's key pair here, in this browser. Only the public half goes "
        "to the PS5; the private half stays on this device.</p></div>"
        "<div id=\"s2\" hidden><h2>Now let <span class=\"nm\"></span> in on the TV</h2>"
        "<p>On the PS5: Settings, WireGuard. Press cross on <b><span class=\"nm\"></span> asks to join</b>.</p>"
        "<p class=\"note\">Keep this page open: it moves on by itself. The request lapses after 10 minutes.</p></div>"
        "<div id=\"s3\" hidden><h2 class=\"ok\"><span class=\"nm\"></span> is in. Now get its tunnel</h2>"
        "<label>Your home's address on the internet<input id=\"home\" autocomplete=\"off\" spellcheck=\"false\" "
        "placeholder=\"Looking it up...\"></label>"
        "<p class=\"note\">Looked up from this device (api.ipify.org). If your router has a dynamic DNS name, "
        "put that here instead. On your router, forward UDP port " +
        port +
        " to this PS5.</p>"
        "<p><button id=\"dl\" type=\"button\">Download the tunnel</button></p>"
        "<p>Then open it in WireGuard: <b>Import tunnel(s) from file</b> on a computer; on a phone or tablet, "
        "<b>+</b>, then <b>Create from file or archive</b>, and pick TokenStation5.conf from Downloads or Files.</p>"
        "<p>Away from home, turn the tunnel on. Apps then use <code>http://10.55.0.1:" +
        api +
        "/v1</code> and one of your API keys.</p>"
        "<p class=\"note\">The file holds this device's private key. Once WireGuard has it, delete the "
        "downloaded copy. If the tunnel never connects, your internet provider may share one public address "
        "among many homes (CGNAT): then use Tailscale instead.</p>"
        "<details><summary>Or copy the tunnel as text</summary><textarea id=\"conf\" rows=\"9\" readonly>"
        "</textarea><p><button id=\"copy\" type=\"button\">Copy</button></p></details></div>"
        "<p class=\"bad\" id=\"err\"></p>"
        "<details id=\"other\"><summary>Rather have your WireGuard app make the key?</summary>" +
        manual("", "", "") +
        "</details>"
        "<p class=\"note\">Nothing joins until someone approves it on the TV, in Settings on the PS5.</p>"
        "<script>\n" +
        std::string(kX25519) +
        "\nvar PS5=" + json_str(info.public_key) + ",PORT=" + port +
        R"JS(;
function $(i){return document.getElementById(i)}
function show(n){['s1','s2','s3'].forEach(function(i){$(i).hidden=i!=n})}
function names(t){Array.prototype.forEach.call(document.querySelectorAll('.nm'),function(e){e.textContent=t})}
function b64(a){return btoa(String.fromCharCode.apply(null,a))}
function load(){try{return JSON.parse(sessionStorage.getItem('ts5wg'))}catch(e){return null}}
function keep(d){try{if(d)sessionStorage.setItem('ts5wg',JSON.stringify(d));else sessionStorage.removeItem('ts5wg')}catch(e){}}
function post(path,body){return fetch(path,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},
 body:body}).then(function(r){return r.json()})}
var dev=null,timer=0;
function conf(){return '[Interface]\nPrivateKey = '+dev.priv+'\nAddress = '+dev.address+'/32\n\n[Peer]\nPublicKey = '+PS5+
 '\nAllowedIPs = 10.55.0.1/32\nEndpoint = '+($('home').value.trim()||'YOUR-HOME-ADDRESS')+':'+PORT+
 '\nPersistentKeepalive = 25\n'}
function finish(){clearInterval(timer);names(dev.name);show('s3');$('conf').value=conf();
 if(!$('home').value)fetch('https://api.ipify.org?format=json').then(function(r){return r.json()})
 .then(function(j){if(!$('home').value)$('home').value=j.ip;$('conf').value=conf()})
 .catch(function(){$('home').placeholder='Type it here: your router shows it'})}
function check(){post('/wireguard/status','key='+encodeURIComponent(dev.pub)).then(function(j){
 if(j.state=='in'||j.state=='off')finish();
 else if(j.state=='none'){clearInterval(timer);keep(null);show('s1');
  $('err').textContent=dev.name+' wasn\'t let in: it was refused on the TV, or the request lapsed. Ask again.';dev=null}
}).catch(function(){})}
function wait(){names(dev.name);show('s2');clearInterval(timer);timer=setInterval(check,2000);check()}
$('home').oninput=function(){$('conf').value=conf()};
$('ask').onclick=function(){
 var n=$('name').value.trim();$('err').textContent='';
 if(!n){$('err').textContent='Give the device a name, such as Laptop.';return}
 var k=new Uint8Array(32);crypto.getRandomValues(k);k[0]&=248;k[31]&=127;k[31]|=64;
 var d={name:n,priv:b64(k),pub:b64(x25519(k))};
 $('ask').disabled=true;
 post('/wireguard/ask','name='+encodeURIComponent(n)+'&key='+encodeURIComponent(d.pub)).then(function(j){
  $('ask').disabled=false;
  if(j.error){$('err').textContent=j.error;return}
  d.name=j.name;d.address=j.address;dev=d;keep(d);wait()})
 .catch(function(){$('ask').disabled=false;$('err').textContent='The PS5 didn\'t answer. Is this device on your home network?'})};
$('dl').onclick=function(){
 if(!$('home').value.trim()){$('err').textContent='Put in your home\'s address on the internet first.';return}
 $('err').textContent='';
 var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([conf()],{type:'application/octet-stream'}));
 a.download='TokenStation5.conf';document.body.appendChild(a);a.click();document.body.removeChild(a)};
$('copy').onclick=function(){$('conf').select();try{document.execCommand('copy')}catch(e){}};
if(typeof BigInt=='undefined'||!window.crypto||!crypto.getRandomValues||!window.fetch){$('s1').hidden=true;
 $('err').textContent='This browser is too old to make the key here. Use the other way, below.';$('other').open=true}
else{dev=load();if(dev&&dev.address)wait();else{dev=null;show('s1')}}
</script>)JS";
    return page(m);
}

}  // namespace

bool wg_page(const std::string &method, const std::string &path, const std::string &body, const std::string &client,
             const WgPageInfo &info, WgStore &store, ts5::Server::Page &out) {
    const bool ask = path == "/wireguard/ask", status = path == "/wireguard/status";
    if (path != "/wireguard" && path != "/wireguard/" && !ask && !status) return false;
    // Why the page can't be used now: its heading, and what to do.
    const char *title = nullptr, *what = nullptr;
    if (client.rfind("10.55.0.", 0) == 0) {
        out.status = 403;
        title = "Not from the tunnel";
        what = "Add devices from your home network, not through WireGuard.";
    } else if (!info.on || info.public_key.empty()) {
        title = "WireGuard is off";
        what = "Turn it on on the PS5 first: Settings, WireGuard. Then load this page again.";
    } else if (!info.running) {
        out.status = 503;
        title = "WireGuard isn't running";
        what = "It's on, but it couldn't start: Settings, WireGuard on the PS5 says why. Then load this page again.";
    }
    if (title) {
        if (ask || status) {
            out.type = "application/json";
            out.body = "{\"error\":" + json_str(std::string(title) + ". " + what) + "}";
        } else {
            out.body = page("<h1>" + std::string(title) + "</h1><p>" + what + "</p>");
        }
        return true;
    }
    if (ask || status) {
        // The one-tap page's calls (one_tap), answered in JSON.
        out.type = "application/json";
        if (method != "POST") {
            out.status = 405;
            out.body = "{\"error\":\"POST only.\"}";
        } else if (status) {
            switch (store.state_of(field(body, "key"))) {
                case WgStore::State::Waiting: out.body = "{\"state\":\"waiting\"}"; break;
                case WgStore::State::In: out.body = "{\"state\":\"in\"}"; break;
                case WgStore::State::Off: out.body = "{\"state\":\"off\"}"; break;
                default: out.body = "{\"state\":\"none\"}";
            }
        } else {
            WgRequest req;
            std::string err;
            if (store.ask(field(body, "name"), field(body, "key"), client, req, err)) {
                out.body = "{\"name\":" + json_str(req.name) + ",\"address\":" + json_str(req.address) + "}";
            } else {
                out.status = 400;
                out.body = "{\"error\":" + json_str(err) + "}";
            }
        }
        return true;
    }
    if (method != "POST") {
        out.body = one_tap(info);
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
