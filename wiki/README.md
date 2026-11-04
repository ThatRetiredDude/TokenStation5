# Wiki source

These files are the GitHub wiki's pages (https://github.com/ThatRetiredDude/TokenStation5/wiki). Edit them here, then publish:

```sh
git clone https://github.com/ThatRetiredDude/TokenStation5.wiki.git /tmp/ts5-wiki
cp wiki/*.md /tmp/ts5-wiki/ && rm /tmp/ts5-wiki/README.md
git -C /tmp/ts5-wiki add -A && git -C /tmp/ts5-wiki commit -m "Update the user guide" && git -C /tmp/ts5-wiki push
```

The file name is the page name (`Install-on-Linux.md` is the page "Install on Linux"); `_Sidebar.md` is the sidebar.
