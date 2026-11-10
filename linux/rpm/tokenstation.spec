# Fedora package, built by linux/build-rpm.sh from the files the .deb has
# (one build for every distribution: see linux/build-deb.sh).
Name:           tokenstation
Version:        %{ts_version}
Release:        1
Summary:        Your PS5's own AI, on Linux: chat, voice and an OpenAI-style API
License:        LicenseRef-PolyForm-Small-Business-1.0.0
URL:            https://github.com/ThatRetiredDude/TokenStation5
Source0:        tokenstation-root.tar.gz
ExclusiveArch:  x86_64
Requires:       mesa-vulkan-drivers
Requires:       sudo
Requires:       shadow-utils
Recommends:     libnotify
Recommends:     tmux

# Already built; no debug packages or rebuilding here.
%global debug_package %{nil}
%global __strip /bin/true
%global __brp_mangle_shebangs %{nil}

%description
TokenStation5 runs GGUF language models on the GPU through Vulkan. The app
is the PS5 title's dashboard on the TV: pick and download models, chat with
the controller or a keyboard, talk to it, make API keys, and reach it from
anywhere with Tailscale. Apps on your network use its OpenAI-style API.
Built for a PS5 running Linux; it works on any Linux machine with a Vulkan GPU.

%prep

%build

%install
mkdir -p %{buildroot}
tar -C %{buildroot} -xzf %{SOURCE0}

%post
/usr/lib/tokenstation/system-setup configure

%preun
if [ "$1" = 0 ]; then /usr/lib/tokenstation/system-setup remove; fi

%postun
if [ -d /run/systemd/system ]; then systemctl daemon-reload || true; fi

%files
/usr/bin/tokenstation
/usr/bin/ts5
/usr/bin/ts5-tui
/usr/bin/ts5-tune
/usr/bin/tokenstation-cu
/usr/bin/tokenstation-memory
/usr/lib/tokenstation
/usr/lib/systemd/system/tokenstation.service
/usr/lib/systemd/system/tokenstation-tv.service
/usr/share/applications/tokenstation.desktop
/usr/share/icons/hicolor/512x512/apps/tokenstation.png
/usr/share/doc/tokenstation
%config(noreplace) /etc/default/tokenstation
%attr(0440,root,root) %config(noreplace) /etc/sudoers.d/tokenstation
