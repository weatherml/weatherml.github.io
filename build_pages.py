import yaml
import re
from collections import defaultdict
import os
from datetime import datetime

# Category display order (categories not listed here appear at the end)
CATEGORY_ORDER = [
    'Global Models', 'Nowcasting', 'Downscaling',
    'Data Assimilation', 'Ensembles', 'Climate Modeling',
    'Extreme Weather', 'Other',
]


def generate_bibtex(paper):
    """Generate a BibTeX entry for a paper."""
    arxiv_id = re.sub(r'v\d+$', '', paper['arxiv'])
    # Create a cite key from first author last name + year
    first_author = paper['authors'].split(',')[0].strip()
    last_name = first_author.split()[-1].lower()
    cite_key = f"{last_name}{paper['year']}"

    bib = f"@article{{{cite_key},\n"
    bib += f"  title = {{{paper['title']}}},\n"
    bib += f"  author = {{{paper['authors']}}},\n"
    bib += f"  year = {{{paper['year']}}},\n"
    bib += f"  eprint = {{{arxiv_id}}},\n"
    bib += f"  archivePrefix = {{arXiv}},\n"
    bib += f"}}\n"
    return bib


def generate_paper_card(paper):
    """Generate a card list item with h4 heading for search indexing."""
    lines = []
    lines.append(f"-   #### {paper['title']}\n")
    lines.append(f"\n")
    lines.append(f"    ---\n")
    lines.append(f"\n")

    # Authors (truncate if too many)
    authors = paper['authors']
    if len(authors) > 100:
        authors = authors[:100].rsplit(',', 1)[0] + ' et al.'
    lines.append(f"    *{authors}* · {paper['year']}\n")
    lines.append(f"\n")

    # Abstract - truncated with expand button
    abstract = paper.get('abstract', '').replace('\n', ' ')
    if abstract:
        arxiv_id = re.sub(r'v\d+$', '', paper['arxiv'])
        snippet = abstract[:200]
        if len(abstract) > 200:
            snippet = snippet.rsplit(' ', 1)[0] + '...'
        lines.append(f'    <span class="abstract-snippet" id="snip-{arxiv_id}">{snippet}</span>')
        lines.append(f'<span class="abstract-full" id="full-{arxiv_id}" hidden>{abstract}</span>')
        if len(abstract) > 200:
            lines.append(f' <span class="abstract-toggle" data-id="{arxiv_id}">more</span>')
        lines.append(f"\n")
        lines.append(f"\n")

    # Links
    arxiv_display = re.sub(r'v\d+$', '', paper['arxiv'])
    lines.append(f"    [:material-file-document: {arxiv_display}](https://arxiv.org/abs/{paper['arxiv']})")
    if paper.get('github'):
        lines.append(f" · [:fontawesome-brands-github:]({paper['github']})")
    lines.append(f" · [:material-content-copy: BibTeX](bibtex/{arxiv_display}.bib){{ .bibtex-link }}")
    lines.append(f"\n")
    lines.append(f"\n")

    # Tags
    tags = paper.get('tags', [])
    display_tags = [t for t in tags if '.' not in t]
    if display_tags:
        tag_spans = ' '.join(f'<span class="md-tag">{tag}</span>' for tag in display_tags)
        lines.append(f"    {tag_spans}\n")
        lines.append(f"\n")

    return ''.join(lines)


def generate_stats(papers):
    """Generate statistics markdown for the index page."""
    papers_by_category = defaultdict(list)
    papers_by_year = defaultdict(int)
    all_tags = defaultdict(int)

    for paper in papers:
        papers_by_category[paper['category']].append(paper)
        papers_by_year[paper['year']] += 1
        for tag in paper.get('tags', []):
            if '.' not in tag:  # Skip arxiv categories
                all_tags[tag] += 1

    total = len(papers)

    md = "## Paper Statistics\n\n"
    md += '<div class="grid cards" markdown>\n\n'
    md += f"-   :material-file-document-multiple: **{total}**\n\n"
    md += f"    Total Papers\n\n"
    md += f"-   :material-folder-multiple: **{len(papers_by_category)}**\n\n"
    md += f"    Categories\n\n"

    years = sorted(papers_by_year.keys())
    if years:
        md += f"-   :material-calendar-range: **{years[0]}\u2013{years[-1]}**\n\n"
        md += f"    Year Range\n\n"

    md += f"-   :material-tag-multiple: **{len(all_tags)}**\n\n"
    md += f"    Unique Tags\n\n"
    md += "</div>\n\n"


    return md


def generate_recent_papers(papers, n=10):
    """Generate markdown for the most recent papers."""
    sorted_papers = sorted(papers, key=lambda p: (p['year'], p['arxiv']), reverse=True)
    recent = sorted_papers[:n]

    md = "## Recent Additions\n\n"
    md += '<div class="grid cards" markdown>\n\n'
    for paper in recent:
        md += generate_paper_card(paper)
    md += '</div>\n\n'

    return md


def build_pages():
    """Build all documentation pages from papers.yml."""
    with open('papers.yml', 'r') as f:
        papers = yaml.safe_load(f) or []

    # Sort papers by year descending, then by arxiv ID descending
    papers.sort(key=lambda p: (p['year'], p.get('arxiv', '')), reverse=True)

    papers_by_category = defaultdict(list)
    for paper in papers:
        papers_by_category[paper['category']].append(paper)

    if not os.path.exists('docs'):
        os.makedirs('docs')

    # Generate individual BibTeX files
    bibtex_dir = os.path.join('docs', 'bibtex')
    os.makedirs(bibtex_dir, exist_ok=True)

    all_bibtex = []
    for paper in papers:
        bib = generate_bibtex(paper)
        all_bibtex.append(bib)
        arxiv_id = re.sub(r'v\d+$', '', paper['arxiv'])
        with open(os.path.join(bibtex_dir, f"{arxiv_id}.bib"), 'w') as f:
            f.write(bib)

    # Full BibTeX file
    with open(os.path.join('docs', 'all_papers.bib'), 'w') as f:
        f.write('\n'.join(all_bibtex))

    # Generate index page with statistics
    with open('docs/index.md', 'w') as f:
        f.write("---\nhide:\n  - navigation\n---\n\n")
        f.write("# Deep Learning in Weather\n\n")
        f.write("A curated collection of papers on deep learning and machine learning ")
        f.write("applied to weather forecasting, climate modeling, and atmospheric science.\n\n")
        f.write(f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n")
        f.write(generate_recent_papers(papers))

    # Generate single papers page with all categories as sections
    nav = [{'Home': 'index.md'}]

    with open('docs/papers.md', 'w') as f:
        f.write("---\nhide:\n  - navigation\n---\n\n<style>.md-content h1 { display: none; }</style>\n\n")

        # Ordered categories
        for category in CATEGORY_ORDER:
            if category in papers_by_category:
                cat_papers = papers_by_category[category]
                f.write(f"## {category} ({len(cat_papers)})\n\n")
                f.write('<div class="grid cards" markdown>\n\n')
                for paper in cat_papers:
                    f.write(generate_paper_card(paper))
                f.write('</div>\n\n')

        # Any remaining categories
        for category in sorted(papers_by_category.keys()):
            if category not in CATEGORY_ORDER:
                cat_papers = papers_by_category[category]
                f.write(f"## {category} ({len(cat_papers)})\n\n")
                f.write('<div class="grid cards" markdown>\n\n')
                for paper in cat_papers:
                    f.write(generate_paper_card(paper))
                f.write('</div>\n\n')

    nav.append({'Papers': 'papers.md'})

    # Generate tags page
    all_tags = defaultdict(list)
    for paper in papers:
        for tag in paper.get('tags', []):
            if '.' not in tag:  # Skip arxiv categories
                all_tags[tag].append(paper)

    if all_tags:
        nav.append({'Tags': 'tags.md'})
        with open('docs/tags.md', 'w') as f:
            f.write("---\nhide:\n  - navigation\n---\n\n")
            f.write("# Tags\n\n")
            for tag in sorted(all_tags.keys()):
                tag_papers = all_tags[tag]
                f.write(f"## {tag} ({len(tag_papers)})\n\n")
                for paper in tag_papers:
                    f.write(f"- **{paper['title']}** ({paper['year']}) ")
                    f.write(f"- [{paper['category']}]({paper['category'].lower().replace(' ', '_')}.md) ")
                    f.write(f"- [arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']})\n")
                f.write("\n")

    # Update mkdocs.yml navigation (text-based to preserve !!python/name tags)
    with open('mkdocs.yml', 'r') as f:
        content = f.read()

    # Build nav YAML lines
    nav_lines = ["nav:\n"]
    for entry in nav:
        for label, value in entry.items():
            nav_lines.append(f"  - {label}: {value}\n")

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
        for category in CATEGORY_ORDER:
            if category in papers_by_category:
                cat_papers = papers_by_category[category]
                f.write(f"### {category} ({len(cat_papers)})\n\n")
                for paper in cat_papers:
                    f.write(f"- **{paper['title']}** ({paper['year']}) - "
                            f"[arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']})\n")
                f.write('\n')

        for category in sorted(papers_by_category.keys()):
            if category not in CATEGORY_ORDER:
                cat_papers = papers_by_category[category]
                f.write(f"### {category} ({len(cat_papers)})\n\n")
                for paper in cat_papers:
                    f.write(f"- **{paper['title']}** ({paper['year']}) - "
                            f"[arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']})\n")
                f.write('\n')

    print(f"Built pages for {len(papers)} papers across {len(papers_by_category)} categories.")


if __name__ == '__main__':
    build_pages()
