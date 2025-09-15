import yaml
from collections import defaultdict
import os

def generate_paper_markdown(paper):
    md = f"### {paper['title']}\n\n"
    md += f"**Authors:** {paper['authors']}\n\n"
    md += f"**Year:** {paper['year']}\n\n"
    md += f"**Abstract:**\n"
    md += f"> {paper['abstract']}\n\n"
    md += f"[**arXiv:{paper['arxiv']}**](https://arxiv.org/abs/{paper['arxiv']})\n\n"
    if 'tags' in paper:
        md += f"**Tags:** `{'`, `'.join(paper['tags'])}`\n\n"
    md += "---\n\n"
    return md

def build_pages():
    with open('papers.yml', 'r') as f:
        papers = yaml.safe_load(f)

    # Sort papers by year, descending
    papers.sort(key=lambda p: p['year'], reverse=True)

    papers_by_category = defaultdict(list)
    for paper in papers:
        papers_by_category[paper['category']].append(paper)

    # Create category pages
    nav = [{'Home': 'index.md'}]
    if not os.path.exists('docs'):
        os.makedirs('docs')

    for category, cat_papers in papers_by_category.items():
        cat_slug = category.lower().replace(' ', '_')
        nav_entry = {category: f"{cat_slug}.md"}
        nav.append(nav_entry)

        with open(f"docs/{cat_slug}.md", 'w') as f:
            f.write(f"# {category}\n\n")
            for paper in cat_papers:
                f.write(generate_paper_markdown(paper))

    # Update mkdocs.yml
    with open('mkdocs.yml', 'r') as f:
        mkdocs_config = yaml.safe_load(f)

    mkdocs_config['nav'] = nav

    with open('mkdocs.yml', 'w') as f:
        yaml.dump(mkdocs_config, f, default_flow_style=False, sort_keys=False)

    # Update README.md
    with open('README.md', 'r') as f:
        readme_content = f.read().split('<!-- PAPERS_START -->')[0]

    with open('README.md', 'w') as f:
        f.write(readme_content)
        f.write('<!-- PAPERS_START -->\n\n')
        f.write("## Paper Collection\n\n")
        for category, cat_papers in papers_by_category.items():
            f.write(f"### {category}\n\n")
            for paper in cat_papers:
                f.write(f"- **{paper['title']}** ({paper['year']}) - [arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']})\n")
            f.write('\n')

if __name__ == '__main__':
    build_pages()
