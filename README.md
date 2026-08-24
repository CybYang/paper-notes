# Paper Notes

Personal research paper reading notes on robotics, embodied AI, motion
generation, and related topics. Notes appear chronologically on the homepage
and can also be found through full-text search or tags.

Published site: <https://cybyang.github.io/paper-notes/>

## Local setup

Python 3.10 or newer is recommended.

On Ubuntu, install the standard-library virtual environment support once if it
is not already available:

```bash
sudo apt install python3-venv
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Preview locally

```bash
source .venv/bin/activate
mkdocs serve
```

Open <http://127.0.0.1:8000/paper-notes/>. Stop the development server with
<kbd>Ctrl</kbd>+<kbd>C</kbd>.

## Build

```bash
source .venv/bin/activate
mkdocs build --strict
```

The generated site is written to `site/`, which is intentionally ignored by
Git.

## Add a paper note

Copy `docs/templates/paper-template.md` to a flat file under `docs/papers/`,
rename it with a stable lowercase slug, and update its `title`, `date`,
`description`, `tags`, and content. Keep `<!-- more -->` after the introduction:
the text above it becomes the homepage excerpt. The homepage and Tags page
update automatically, so new files do not need to be added to `mkdocs.yml`.

Put overview figures under `docs/assets/papers/` and reference them before the
`<!-- more -->` marker with a path such as
`../assets/papers/my-paper-overview.png`. The figure then appears both in the
homepage entry and in the full article.

## GitHub Pages

Pushes to `main` run `.github/workflows/deploy-pages.yml`, build the site, and
deploy the generated artifact with GitHub Pages Actions. For the first
deployment, set **Settings → Pages → Build and deployment → Source** to
**GitHub Actions** in this repository.
