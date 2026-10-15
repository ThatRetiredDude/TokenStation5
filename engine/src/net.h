// A small HTTP(S) client for the web search tool: one POST, read to the end.
// HTTPS is Mbed TLS, checking the server's certificate against the system's
// CA bundle, or TS5_CA_FILE, or the Mozilla bundle built in (the console has
// no CA files a title can read).
#pragma once
#include <string>
#include <vector>

namespace ts5 {

struct HttpResponse {
    int status = 0;
    std::string body;
};

// url: http://host[:port]/path or https://host[:port]/path. headers: "Name: value".
bool http_post(const std::string &url, const std::vector<std::string> &headers, const std::string &body,
               HttpResponse &resp, std::string &err, int timeout_s = 30);

}  // namespace ts5
