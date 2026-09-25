import html
import json
import yaml
import re
import unicodedata
from collections import defaultdict
from xml.sax.saxutils import escape
import glob
import os
import shutil
from datetime import datetime, timezone

from tagging import GROUPS, TAG_GROUP, TAG_ORDER, tag_paper

SITE_URL = 'https://weatherml.github.io/'
REPO_URL = 'https://github.com/weatherml/weatherml.github.io'
SUGGEST_URL = f'{REPO_URL}/issues/new?template=suggest-paper.yml'

# Category display order (categories not listed here appear at the end)
CATEGORY_ORDER = [
    'Global Models', 'Nowcasting', 'Downscaling',
    'Data Assimilation', 'Ensembles', 'Climate Modeling',
    'Extreme Weather', 'Ocean & Sea Ice', 'Air Quality & Composition',
    'Remote Sensing', 'Other',
]

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Words skipped when picking the title word for a cite key
CITE_KEY_STOPWORDS = {
    'a', 'an', 'the', 'on', 'of', 'in', 'for', 'to', 'and', 'with', 'via',
    'towards', 'toward', 'from', 'by', 'at', 'is', 'are', 'using', 'can',
}

FEED_SIZE = 50
PAGE_SIZE = 30     # cards per topic page
RECENT_COUNT = 24  # cards on the home page


def slugify(text):
    """Slugify a category/tag name for use in file names and URL anchors."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def ascii_word(text):
    """Lowercase ASCII letters/digits only (e.g. 'Rühling' -> 'ruhling')."""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', text.lower())


# Text-mode LaTeX found in arXiv titles/abstracts, outside of $...$ math.
# Each maps to (HTML replacement, plain-text replacement).
TEX_TEXT_COMMANDS = [
    # LaTeX quotes; backticks would otherwise start Markdown code spans
    (r"``(.*?)(?:''|\")", r'“\1”', r'“\1”'),
    (r"`([^`']*)'", r'‘\1’', r'‘\1’'),
    (r'`', '‘', '‘'),
    (r'\\textbf\{([^{}]*)\}', r'<strong>\1</strong>', r'\1'),
    (r'\\(?:textit|emph)\{([^{}]*)\}', r'<em>\1</em>', r'\1'),
    (r'\{\\(?:it|em)\s+([^{}]*)\}', r'<em>\1</em>', r'\1'),
    (r'\\texttt\{([^{}]*)\}', r'<code>\1</code>', r'\1'),
    (r'\\textsubscript\{([^{}]*)\}', r'<sub>\1</sub>', r'\1'),
    (r'\\href\{[^{}]*\}\{([^{}]*)\}', r'\1', r'\1'),
    (r'\\(?:url|gls|textrm|ul)\{([^{}]*)\}', r'\1', r'\1'),
    (r'\\(?:footnote|cite)\{[^{}]*\}', '', ''),
    (r'\\(?:textdegree|degree)\b\s*', '°', '°'),
    (r'\\([%&])', r'\1', r'\1'),
    (r'\\SI\{([^{}]*)\}\{([^{}]*)\}', r'\1 \2', r'\1 \2'),
    (r'\\si\{([^{}]*)\}', r'\1', r'\1'),
    (r'\\slash\b\s*', '/', '/'),
    (r'\\sim\b\s*', '∼', '∼'),
]

# Inline math: $...$, not preceded by a backslash (\$ is a literal dollar)
TEX_MATH = re.compile(r'(?<!\\)(\$[^$]+?(?<!\\)\$)')


def clean_tex(text, html=True):
    """Rewrite text-mode LaTeX outside $...$ so only real math is left for KaTeX."""
    parts = TEX_MATH.split(text)
    for i in range(0, len(parts), 2):  # even parts are outside math
        # Repeat so nested commands (\textbf{\ul{D}}) unwrap from the inside
        previous = None
        while previous != parts[i]:
            previous = parts[i]
            for pattern, as_html, as_text in TEX_TEXT_COMMANDS:
                parts[i] = re.sub(pattern, as_html if html else as_text, parts[i])
    return ''.join(parts)


def truncate(text, limit=200):
    """Cut at a word boundary without leaving a $...$ span open."""
    if len(text) <= limit:
        return text
    snippet = text[:limit].rsplit(' ', 1)[0]
    dollars = [m.start() for m in re.finditer(r'(?<!\\)\$', snippet)]
    if len(dollars) % 2:
        snippet = snippet[:dollars[-1]].rstrip()
    return snippet + '...'


def base_id(paper):
    """arXiv ID without the version suffix."""
    return re.sub(r'v\d+$', '', paper['arxiv'])


def paper_month(paper):
    """'Apr 2026' from the arXiv ID (YYMM.NNNNN), falling back to the year."""
    match = re.match(r'(\d{2})(\d{2})\.\d{4,5}', paper['arxiv'])
    if match:
        month = int(match.group(2))
        if 1 <= month <= 12:
            return f"{MONTHS[month - 1]} 20{match.group(1)}"
    return str(paper['year'])


def paper_date(paper):
    """First day of the arXiv submission month, used for feed timestamps."""
    match = re.match(r'(\d{2})(\d{2})\.\d{4,5}', paper['arxiv'])
    if match and 1 <= int(match.group(2)) <= 12:
        return f"20{match.group(1)}-{match.group(2)}-01T00:00:00Z"
    return f"{paper['year']}-01-01T00:00:00Z"


# Tag names that don't slugify into readable URLs
TAG_SLUGS = {'0.25°': 'quarter-degree', 'Coarse (≥1°)': 'coarse'}


def tag_slug(tag):
    return TAG_SLUGS.get(tag, slugify(tag))


def listing_page(folder, page=1):
    """Source path of one page of a listing (topic or tag), relative to docs/."""
    name = 'index' if page == 1 else str(page)
    return f"{folder}/{name}.md"


def listing_url(folder, page=1):
    """Site URL path of one page of a listing."""
    suffix = '' if page == 1 else f"{page}/"
    return f"{folder}/{suffix}"


def topic_folder(category):
    return f"papers/{slugify(category)}"


def tag_filter_url(tag):
    """The Explore page with this tag preselected (absolute: the site is served from /)."""
    return f"/explore/?t={tag_slug(tag)}"


def topic_page(category, page=1):
    return listing_page(topic_folder(category), page)


def paginate(items, size=PAGE_SIZE):
    return [items[i:i + size] for i in range(0, len(items), size)] or [[]]


def generate_pager(page, total):
    """Newer/older links plus page numbers, linking pages in the same folder."""
    def href(n):
        return 'index.md' if n == 1 else f'{n}.md'

    parts = []
    if page > 1:
        parts.append(f'[:material-arrow-left: Newer]({href(page - 1)}){{ .pager-step }}')
    for n in range(1, total + 1):
        parts.append(f'**{n}**' if n == page else f'[{n}]({href(n)})')
    if page < total:
        parts.append(f'[Older :material-arrow-right:]({href(page + 1)}){{ .pager-step }}')
    return f'<nav class="pager" markdown="span">{" ".join(parts)}</nav>\n\n'


def assign_cite_keys(papers):
    """Google Scholar–style keys (lastname + year + first title word), unique.

    Remaining clashes get a/b/c suffixes in arXiv ID order so keys stay
    stable as new papers are added.
    """
    keys = {}
    for paper in papers:
        first_author = paper['authors'].split(',')[0].strip()
        last_name = ascii_word(first_author.split()[-1]) if first_author else 'anon'
        words = [ascii_word(w) for w in paper['title'].split()]
        title_word = next((w for w in words if w and w not in CITE_KEY_STOPWORDS), '')
        keys[base_id(paper)] = f"{last_name}{paper['year']}{title_word}"

    by_key = defaultdict(list)
    for arxiv_id, key in keys.items():
        by_key[key].append(arxiv_id)
    for key, ids in by_key.items():
        if len(ids) > 1:
            for i, arxiv_id in enumerate(sorted(ids)):
                keys[arxiv_id] = f"{key}{chr(ord('a') + i)}"
    return keys


def generate_bibtex(paper, cite_key):
    """Generate a BibTeX entry for a paper."""
    arxiv_id = base_id(paper)
    bib = f"@article{{{cite_key},\n"
    bib += f"  title = {{{paper['title']}}},\n"
    bib += f"  author = {{{' and '.join(a.strip() for a in paper['authors'].split(','))}}},\n"
    bib += f"  year = {{{paper['year']}}},\n"
    bib += f"  eprint = {{{arxiv_id}}},\n"
    bib += f"  archivePrefix = {{arXiv}},\n"
    bib += f"  url = {{https://arxiv.org/abs/{arxiv_id}}},\n"
    bib += f"}}\n"
    return bib


def generate_paper_card(paper, root=''):
    """Generate a card list item with h4 heading for search indexing.

    The heading carries the arXiv ID as its anchor so each paper can be
    linked to directly. `root` is the relative path back to docs/.
    """
    arxiv_id = base_id(paper)
    lines = []
    lines.append(f"-   #### {clean_tex(paper['title'])} {{ #{arxiv_id} }}\n")
    lines.append(f"\n")

    # Authors (truncate if too many)
    authors = paper['authors']
    if len(authors) > 100:
        authors = authors[:100].rsplit(',', 1)[0] + ' et al.'
    lines.append(f"    *{authors}* · {paper_month(paper)}\n    {{ .paper-meta }}\n")
    lines.append(f"\n")

    # Abstract - truncated with expand button
    abstract = paper.get('abstract', '').replace('\n', ' ')
    if abstract:
        # The snippet is plain text so a cut can't split an HTML tag
        snippet = truncate(clean_tex(abstract, html=False))
        full = clean_tex(abstract)
        lines.append(f'    <span class="abstract-snippet" id="snip-{arxiv_id}">{snippet}</span>')
        lines.append(f'<span class="abstract-full" id="full-{arxiv_id}" hidden>{full}</span>')
        if len(abstract) > 200:
            lines.append(f' <span class="abstract-toggle" data-id="{arxiv_id}">more</span>')
        lines.append(f"\n")
        lines.append(f"\n")

    # Links (pinned to the bottom of the card together with the tags)
    links = [
        f"[:material-file-document-outline: arXiv](https://arxiv.org/abs/{paper['arxiv']})",
        f"[:material-file-pdf-box: PDF](https://arxiv.org/pdf/{paper['arxiv']})",
    ]
    if paper.get('github'):
        links.append(f"[:fontawesome-brands-github: Code]({paper['github']})")
    links.append(f"[:material-content-copy: BibTeX]({root}bibtex/{arxiv_id}.bib){{ .bibtex-link }}")
    lines.append(f"    {' · '.join(links)}\n    {{ .paper-links }}\n")
    lines.append(f"\n")

    # Tags, each opening the Explore page with that tag selected
    tags = paper.get('_tags', [])
    if tags:
        tag_links = ' '.join(
            f'<a class="md-tag" href="{tag_filter_url(tag)}" data-tag="{tag_slug(tag)}">{tag}</a>'
            for tag in tags
        )
        lines.append(f"    {tag_links}\n    {{ .paper-tags }}\n")
        lines.append(f"\n")

    return ''.join(lines)


def generate_link_grid(entries):
    """Compact grid of (label, link, count) cards."""
    md = '<div class="grid cards topics" markdown>\n\n'
    for label, link, count in entries:
        md += f'-   [{label}]({link}) <span class="topic-count">{count}</span>\n'
    md += '\n</div>\n\n'
    return md


def write_listing(folder, heading, meta, listing_papers, front_matter=''):
    """Write a paginated card listing to docs/<folder>/ (index.md, 2.md, ...).

    `meta` is Markdown shown under the heading, after the paper count.
    Returns the list of pages (each a list of papers).
    """
    pages = paginate(listing_papers)
    os.makedirs(os.path.join('docs', folder), exist_ok=True)
    for n, page_papers in enumerate(pages, start=1):
        title = heading if n == 1 else f"{heading} · Page {n}"
        page_info = f" · page {n} of {len(pages)}" if len(pages) > 1 else ''
        with open(os.path.join('docs', listing_page(folder, n)), 'w') as f:
            f.write(f"---\ntitle: '{title}'\n{front_matter}---\n\n")
            # Title on the left, count/page/BibTeX on the right (stacks on phones)
            f.write('<div class="listing-header" markdown>\n\n')
            f.write(f"# {heading}\n\n")
            f.write(f'<p class="page-meta" markdown="span">{len(listing_papers)} papers'
                    f'{page_info} · {meta}</p>\n\n')
            f.write('</div>\n\n')
            f.write('<div class="grid cards" markdown>\n\n')
            for paper in page_papers:
                f.write(generate_paper_card(paper, root='../../'))
            f.write('</div>\n\n')
            if len(pages) > 1:
                f.write(generate_pager(n, len(pages)))
    return pages


def nav_entries(label, folder, pages, indent):
    """Nav lines for a listing; later pages nest under it (hidden by CSS)."""
    pad = ' ' * indent
    if len(pages) == 1:
        return [f"{pad}- '{label}': {listing_page(folder)}\n"]
    lines = [f"{pad}- '{label}':\n", f"{pad}  - {listing_page(folder)}\n"]
    for n in range(2, len(pages) + 1):
        lines.append(f"{pad}  - 'Page {n}': {listing_page(folder, n)}\n")
    return lines


def generate_recent_papers(papers, n=RECENT_COUNT):
    """Generate markdown for the most recent papers."""
    sorted_papers = sorted(papers, key=lambda p: (p['year'], p['arxiv']), reverse=True)
    recent = sorted_papers[:n]

    md = "## Recent Additions\n\n"
    # Excluded from search: the same cards are indexed on the topic pages
    md += '<div class="grid cards" markdown data-search-exclude>\n\n'
    for paper in recent:
        md += generate_paper_card(paper)
    md += '</div>\n\n'

    return md


def generate_feed(papers, updated, paper_urls):
    """Atom feed of the most recently published papers."""
    recent = sorted(papers, key=lambda p: p['arxiv'], reverse=True)[:FEED_SIZE]
    xml = '<?xml version="1.0" encoding="utf-8"?>\n'
    xml += '<feed xmlns="http://www.w3.org/2005/Atom">\n'
    xml += '  <title>weatherml</title>\n'
    xml += '  <subtitle>New papers on machine learning for weather and climate</subtitle>\n'
    xml += f'  <link href="{SITE_URL}"/>\n'
    xml += f'  <link rel="self" href="{SITE_URL}feed.xml"/>\n'
    xml += f'  <id>{SITE_URL}</id>\n'
    xml += f'  <updated>{updated}</updated>\n'
    for paper in recent:
        arxiv_id = base_id(paper)
        category = paper['category']
        page_url = f"{SITE_URL}{paper_urls[arxiv_id]}#{arxiv_id}"
        xml += '  <entry>\n'
        xml += f'    <title>{escape(paper["title"])}</title>\n'
        xml += f'    <link href="https://arxiv.org/abs/{arxiv_id}"/>\n'
        xml += f'    <link rel="related" href="{page_url}"/>\n'
        xml += f'    <id>https://arxiv.org/abs/{arxiv_id}</id>\n'
        xml += f'    <updated>{paper_date(paper)}</updated>\n'
        xml += f'    <author><name>{escape(paper["authors"])}</name></author>\n'
        xml += f'    <category term="{escape(category)}"/>\n'
        xml += f'    <summary>{escape(paper.get("abstract", "").strip())}</summary>\n'
        xml += '  </entry>\n'
    xml += '</feed>\n'
    return xml


def math_html(text):
    """HTML-escape text and wrap $...$ the way pymdownx.arithmatex does,
    so katex.js renders it in cards built by the tag filter."""
    parts = TEX_MATH.split(clean_tex(html.escape(text, quote=False)))
    for i in range(1, len(parts), 2):
        parts[i] = f'<span class="arithmatex">\\({parts[i][1:-1]}\\)</span>'
    return ''.join(parts)


def paper_record(paper, paper_urls):
    """Compact entry for docs/explore/papers.json, read by tag-filter.js."""
    authors = paper['authors']
    if len(authors) > 100:
        authors = authors[:100].rsplit(',', 1)[0] + ' et al.'
    abstract = paper.get('abstract', '').replace('\n', ' ')
    record = {
        'id': base_id(paper),
        'v': paper['arxiv'],
        't': math_html(paper['title']),
        'a': html.escape(authors),
        'm': paper_month(paper),
        's': math_html(truncate(clean_tex(abstract, html=False))),
        'u': paper_urls[base_id(paper)],
        'c': slugify(paper['category']),
        'k': [tag_slug(t) for t in paper['_tags']],
    }
    if paper.get('github'):
        record['g'] = paper['github']
    return record


def tag_chip(slug, label, count):
    return (f'<button class="md-tag tag-chip" type="button" data-tag="{slug}">'
            f'{label} <span class="tag-chip-count">{count}</span></button>')


def write_if_changed(path, content):
    """Write a file, leaving it untouched if the content is identical."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            if f.read() == content:
                return
    with open(path, 'w') as f:
        f.write(content)


def build_pages():
    """Build all documentation pages from papers.yml."""
    with open('papers.yml', 'r') as f:
        papers = yaml.safe_load(f) or []

    # Sort papers by year descending, then by arxiv ID descending
    papers.sort(key=lambda p: (p['year'], p.get('arxiv', '')), reverse=True)

    papers_by_category = defaultdict(list)
    papers_by_tag = defaultdict(list)
    for paper in papers:
        papers_by_category[paper['category']].append(paper)
        paper['_tags'] = tag_paper(paper)
        for tag in paper['_tags']:
            papers_by_tag[tag].append(paper)

    ordered_categories = [c for c in CATEGORY_ORDER if c in papers_by_category]
    ordered_categories += [c for c in sorted(papers_by_category.keys())
                           if c not in CATEGORY_ORDER]

    os.makedirs('docs', exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')

    # BibTeX: one file per paper, one per category, and one for everything
    cite_keys = assign_cite_keys(papers)
    bibtex = {base_id(p): generate_bibtex(p, cite_keys[base_id(p)]) for p in papers}

    bibtex_dir = os.path.join('docs', 'bibtex')
    os.makedirs(bibtex_dir, exist_ok=True)
    for arxiv_id, bib in bibtex.items():
        write_if_changed(os.path.join(bibtex_dir, f"{arxiv_id}.bib"), bib)
    # Drop files for papers no longer in the collection
    for path in glob.glob(os.path.join(bibtex_dir, '*.bib')):
        if os.path.basename(path)[:-4] not in bibtex:
            os.remove(path)

    write_if_changed(os.path.join('docs', 'all_papers.bib'),
                     '\n'.join(bibtex[base_id(p)] for p in papers))

    bib_dir = os.path.join('docs', 'bib')
    os.makedirs(bib_dir, exist_ok=True)
    for path in glob.glob(os.path.join(bib_dir, '*.bib')):
        os.remove(path)
    for category in ordered_categories:
        with open(os.path.join(bib_dir, f"{slugify(category)}.bib"), 'w') as f:
            f.write('\n'.join(bibtex[base_id(p)] for p in papers_by_category[category]))

    # Split each topic into pages, newest first, and remember where each
    # paper ended up so the feed and tags page can link straight to it
    topic_pages = {c: paginate(papers_by_category[c]) for c in ordered_categories}
    paper_urls = {}
    for category, pages in topic_pages.items():
        for n, page_papers in enumerate(pages, start=1):
            for paper in page_papers:
                paper_urls[base_id(paper)] = listing_url(topic_folder(category), n)

    # Atom feed
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(os.path.join('docs', 'feed.xml'), 'w') as f:
        f.write(generate_feed(papers, now, paper_urls))

    # Home page
    with open('docs/index.md', 'w') as f:
        f.write("---\nhide:\n  - navigation\n  - toc\ntitle: weatherml\n---\n\n")
        f.write("A collection of papers on AI for weather forecasting, "
                "climate modelling and atmospheric science.\n\n")
        f.write(f'<p class="page-meta" markdown="span">{len(papers)} papers · updated {today} · '
                f'<a href="feed.xml">:material-rss: RSS</a> · '
                f'<a href="all_papers.bib" download>:material-download: BibTeX</a> · '
                f'<a href="{SUGGEST_URL}">:material-plus: Suggest a paper</a></p>\n\n')
        f.write("## Browse by Topic\n\n")
        f.write(generate_link_grid(
            (c, topic_page(c), len(papers_by_category[c])) for c in ordered_categories
        ))
        f.write(generate_recent_papers(papers))

    # Papers: each topic gets a folder of pages (index.md, 2.md, 3.md, ...)
    for old in ('docs/papers', 'docs/tags'):
        if os.path.exists(old):
            shutil.rmtree(old)
    for old in ('docs/papers.md', 'docs/tags.md'):
        if os.path.exists(old):
            os.remove(old)

    for category in ordered_categories:
        slug = slugify(category)
        write_listing(
            topic_folder(category), category,
            f'<a href="../../bib/{slug}.bib" download>:material-download: '
            f'BibTeX for this topic</a>',
            papers_by_category[category],
            front_matter="hide:\n  - toc\n",
        )

    # Explore: one page where tags can be combined. Chips are rendered here;
    # tag-filter.js reads papers.json and shows the papers matching all of
    # the selected ones.
    ordered_tags = [t for t in TAG_ORDER if t in papers_by_tag]
    topic_slugs = {slugify(c) for c in ordered_categories}
    clashes = topic_slugs & {tag_slug(t) for t in ordered_tags}
    assert not clashes, f"topic and tag slugs clash: {clashes}"

    if os.path.exists('docs/tags'):
        shutil.rmtree('docs/tags')
    os.makedirs('docs/explore', exist_ok=True)
    with open('docs/explore/papers.json', 'w') as f:
        json.dump({
            'tags': {tag_slug(t): t for t in ordered_tags},
            'papers': [paper_record(p, paper_urls) for p in papers],
        }, f, ensure_ascii=False, separators=(',', ':'))

    groups = [('Topic', [(slugify(c), c, len(papers_by_category[c]))
                         for c in ordered_categories])]
    for group in GROUPS:
        groups.append((group, [(tag_slug(t), t, len(papers_by_tag[t]))
                               for t in ordered_tags if TAG_GROUP[t] == group]))

    with open('docs/explore/index.md', 'w') as f:
        f.write("---\ntitle: Explore\nhide:\n  - navigation\n  - toc\n"
                "search:\n  exclude: true\n---\n\n")
        f.write('<div id="tag-filter" data-src="papers.json">\n')
        for group, chips in groups:
            f.write('<div class="tag-group">\n')
            f.write(f'<span class="tag-group-name">{group}</span>\n')
            f.write('<div class="tag-chips">\n')
            f.write('\n'.join(tag_chip(*chip) for chip in chips) + '\n')
            f.write('</div>\n</div>\n')
        f.write('<p class="tag-status page-meta"></p>\n')
        f.write('<div class="grid cards" id="tag-results"></div>\n')
        f.write('<p class="tag-more-wrap"><button class="md-button" id="tag-more" '
                'type="button" hidden>Show more</button></p>\n')
        f.write('</div>\n')

    # Update mkdocs.yml navigation (text-based to preserve !!python/name tags)
    with open('mkdocs.yml', 'r') as f:
        content = f.read()

    # Later pages sit under their topic so the topic stays highlighted in the
    # sidebar; CSS hides them there since the pager already links them
    # The home page is reached through the logo and site name, so it has no tab
    nav_lines = ["nav:\n", "  - All Papers:\n"]
    for category, pages in topic_pages.items():
        nav_lines += nav_entries(category, topic_folder(category), pages, indent=4)
    nav_lines.append("  - Explore: explore/index.md\n")

    # Replace everything from "nav:" to the end (nav is always last section)
    content = re.sub(r'^nav:.*', ''.join(nav_lines).rstrip(), content, flags=re.DOTALL | re.MULTILINE)

    with open('mkdocs.yml', 'w') as f:
        f.write(content)

    # Update README.md
    with open('README.md', 'r') as f:
        readme_content = f.read()

    if '<!-- PAPERS_START -->' in readme_content:
        readme_content = readme_content.split('<!-- PAPERS_START -->')[0]
    else:
        readme_content = readme_content.rstrip() + '\n\n'

    with open('README.md', 'w') as f:
        f.write(readme_content)
        f.write('<!-- PAPERS_START -->\n\n')
        f.write(f"## Paper Collection ({len(papers)} papers)\n\n")
        for category in ordered_categories:
            cat_papers = papers_by_category[category]
            f.write(f"### {category} ({len(cat_papers)})\n\n")
            for paper in cat_papers:
                f.write(f"- **{paper['title']}** ({paper['year']}) - "
                        f"[arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']})\n")
            f.write('\n')

    print(f"Built pages for {len(papers)} papers across {len(ordered_categories)} categories.")


if __name__ == '__main__':
    build_pages()
