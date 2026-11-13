// HTTPS for ts5 serve on its tailnet name (tailnet_serve.h): a TLS front on
// 127.0.0.1 with the name's certificate (acme.h). The tunnel passes the
// tailnet's port 443 to it; once a connection's TLS is done it is passed on,
// as plain HTTP, to the API's port. The API names it by the device behind
// it, as it names the tunnel's own connections, so every key check and
// tailnet-only page works the same over HTTPS: the front opens its
// connection to the API from a port of its own and tells the tunnel which
// device that port is for (note), before it connects.
//
// A connection that didn't come through the tunnel (a program on this
// machine) gets no name, as if it had reached the API directly.
//
// It answers only a connection that asks for one of the certificate's names
// (SNI): https://100.x.y.z/ asks for none and is ended before the
// certificate goes out, as if nothing answered HTTPS there.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace ts5 {

class HttpsFront {
public:
    // The device behind a connection to the front from peer_port on this
    // machine, "" if none (the tunnel's client_for).
    using NameOf = std::function<std::string(uint16_t peer_port)>;
    // Names the front's connection to the API from local_port (the tunnel's
    // note_client).
    using Note = std::function<void(uint16_t local_port, const std::string &name)>;

    HttpsFront();
    ~HttpsFront();
    // Listens on 127.0.0.1:port (0: a free one); connections go on to
    // 127.0.0.1:api_port. The port it listens on, or 0 with why.
    int start(int api_port, NameOf name_of, Note note, std::string &why, int port = 0);
    // The certificate chain (PEM, the certificate first) and its key it
    // answers with from now on; connections already open keep theirs. False
    // with why when they can't be used, and the earlier pair stays.
    bool set_certificate(const std::string &cert_pem, const std::string &key_pem, std::string &why);
    bool has_certificate();
    // Takes a connection that began TLS on another port (the API's, for an
    // address that starts https:// with the API's port), its handshake not
    // read yet, from device: served as one of the front's own, and closed
    // when done. False, with fd untouched, when there's no certificate or
    // too many connections.
    bool adopt(int fd, const std::string &device);

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace ts5
