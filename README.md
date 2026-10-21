# llm-sys

The llm-sys blog: a static site built with [Hugo](https://gohugo.io), hosted on GitHub Pages. No Node, no npm, no theme to update; every template, style, and script is in this repository.

## Run it locally

Install Hugo (extended, 0.160 or newer), then start the preview server:

```bash
brew install hugo
hugo server
```

Open <http://localhost:1313>. Pages reload as you save.

To build the production site into `public/`:

```bash
hugo --gc --minify
```

## Write a post

Each post is a folder under `content/blog/` holding `index.md` and the files it uses (figures, videos, the paper PDF).

```bash
hugo new content blog/my-post/index.md
```

This creates the folder with the front matter filled in from `archetypes/blog.md`. New posts start as `draft: true`; `hugo server -D` shows drafts locally, and the published site never does. Set `draft: false` (or delete the line) to publish.

Every component, with the Markdown that produces it, is shown at <http://localhost:1313/styleguide/> while the server runs: math, numbered figures, side-by-side panels, video, tables, code, Mermaid diagrams, callouts, theorem blocks, footnotes, headline numbers, and interactive charts. The style guide is built but never listed or indexed.

The header of a post comes from its front matter:

```yaml
title: "Post title"
subtitle: "One sentence shown under the title and in previews."
date: 2026-10-20
authors:
  - name: Fanxu Meng
    affil: [1]          # index into affiliations
    marks: ["*"]        # extra superscripts
affiliations:
  - Dots Studio, Xiaohongshu Inc.
notes:
  - { mark: "*", text: "Equal contribution." }
links:                  # buttons under the authors
  - { label: Paper, url: paper.pdf, icon: pdf }   # icons: pdf, arxiv, github, video, data
  - { label: Code, url: "https://github.com/llm-sys/...", icon: github }
tags: [sparse attention]
accent: amber           # second color of the cover pattern: amber, coral, butter, sky, lilac
arxiv: "2610.13990"     # optional, quoted; see Citations below
arxiv_class: cs.LG      # optional
citekey: meng2026dosa   # BibTeX key; or give a full `bibtex:` block
```

### Citations

Every post ends with a citation block and BibTeX, chosen in this order:

1. `bibtex:` in front matter is used as written.
2. `arxiv:` set: the BibTeX cites the arXiv entry (`eprint`, `archivePrefix`, `primaryClass` from `arxiv_class`), and an arXiv button appears next to the paper link.
3. Neither: the BibTeX cites the blog post (`howpublished = {\url{...}}`).

Quote the arXiv ID. Unquoted, YAML reads `2610.13990` as a number, and an ID like `2610.13900` would lose its trailing zeros; the build stops with an error if that happens.

The newest post becomes the featured card on the home page automatically. Older posts move to the list below it.

## Remove or hide a post

- Unpublish but keep it: set `draft: true` in its front matter.
- Delete it: remove its folder, for example `rm -r content/blog/my-post`.

## Other pages

| What | Where |
| --- | --- |
| Site name, home headline, intro, GitHub, email | `hugo.toml`, `[params]` |
| Top navigation | `hugo.toml`, `[menus]` |
| Open-source projects (home page and /projects/) | `data/projects.yaml` |
| About page | `content/about/index.md` |
| Colors, spacing, type | `assets/css/site.css` (the `:root` block at the top) |
| Post typography and components | `assets/css/prose.css` |
| Logo | `layouts/_partials/glyph.html`, `static/favicon.svg`, `static/brand/` |

## Publish

Pushing to `main` builds and deploys the site through `.github/workflows/deploy.yml`. One-time setup on GitHub: repository **Settings → Pages → Source: GitHub Actions**.

The site is served at <https://llmsys.net>. The domain is set in `static/CNAME` and under Settings → Pages → Custom domain; its DNS points at GitHub Pages. To move to another domain, change both, plus `baseURL` in `hugo.toml`.

## Layout

```
hugo.toml               site settings, menus, Markdown and math options
content/                posts and pages, in Markdown
data/projects.yaml      project list
layouts/                HTML templates
  _partials/            header, footer, paper card, citation, cover pattern
  _shortcodes/          figure, row, video, stats, chart, screening
  _markup/              how Markdown images, code, math, tables, and quotes render
assets/css, assets/js   styles and scripts, bundled and fingerprinted by Hugo
static/                 fonts, KaTeX fonts, favicon, brand files, copied as-is
```

Math is rendered to HTML at build time with KaTeX, so pages run no math JavaScript. Fonts are self-hosted: Mona Sans, Mona Sans Mono, and Newsreader, all under the SIL Open Font License (see `static/fonts/LICENSE.md`).
