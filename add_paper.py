"""Add a single paper to papers.yml by arXiv ID or URL.

Usage:
    python add_paper.py 2604.09041 --star
    python add_paper.py "https://arxiv.org/abs/2604.09041"
    python add_paper.py "some text containing an arxiv link" --star

The first arXiv ID found in the argument is used, so the raw body of a
GitHub issue can be passed directly.
"""

import argparse
import re
import sys

import arxiv
import yaml

from find_papers import make_paper_entry, save_papers, base_arxiv_id

ARXIV_ID_PATTERN = re.compile(r'(\d{4}\.\d{4,5})(v\d+)?')


def add_paper(text, star=False):
    match = ARXIV_ID_PATTERN.search(text)
    if not match:
        print(f"error: no arXiv ID found in: {text[:200]!r}")
        sys.exit(1)
    arxiv_id = match.group(1)

    with open('papers.yml', 'r') as f:
        papers = yaml.safe_load(f) or []

    for paper in papers:
        if base_arxiv_id(paper['arxiv']) == arxiv_id:
            if star and not paper.get('starred'):
                paper['starred'] = True
                save_papers(papers)
                print(f"Starred existing paper {arxiv_id}: {paper['title']}")
            else:
                print(f"Paper {arxiv_id} already in collection: {paper['title']}")
            return paper

    client = arxiv.Client(num_retries=5)
    try:
        result = next(client.results(arxiv.Search(id_list=[arxiv_id])))
    except StopIteration:
        print(f"error: arXiv ID {arxiv_id} not found on arXiv")
        sys.exit(1)

    paper = make_paper_entry(result)
    if star:
        paper['starred'] = True
    papers.append(paper)
    save_papers(papers)
    print(f"Added [{paper['category']}] {paper['arxiv']}: {paper['title']}")
    return paper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add a paper to papers.yml by arXiv ID/URL.')
    parser.add_argument('text', help='arXiv ID, URL, or text containing one')
    parser.add_argument('--star', action='store_true',
                        help='mark the paper as starred')
    args = parser.parse_args()
    add_paper(args.text, star=args.star)
