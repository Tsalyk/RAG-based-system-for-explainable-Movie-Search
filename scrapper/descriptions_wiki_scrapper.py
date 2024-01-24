# Wikipedia movies data scrapper

import pandas as pd
import numpy as np

from pwiki.wiki import Wiki
import wikipedia
import wikipediaapi

import re

from tqdm import tqdm
tqdm.pandas()


def get_movie_description(title: str) -> str:
    """
    Retrieves movie plot from Wikipedia with a title
    """
    data_loaders = {'wiki': wikipedia_loader, 'pywiki': pywiki_loader, 'apiwiki': apiwiki_loader}
    titles = [title, f"{title.split(' (')[0]} (film)", title.split(' (')[0]]

    content, success = '', False
    for title in titles:
        if success:
            break
        for loader_name in data_loaders:
            if success:
                break
            loader = data_loaders[loader_name]
            try:
                content = loader(title)
                if len(content) > 100:
                    success = True
            except:
                pass

    if len(content) > 0:
        plot_pattern = re.compile(r'==\s*Plot\s*==\n(.*?)(?==|$)', re.DOTALL)
        plot_match = plot_pattern.search(content)

        if plot_match:
            return plot_match.group(1).strip()

    return None

def wikipedia_loader(title: str):
    return wikipedia.page(title).content

def pywiki_loader(title: str):
    wiki = Wiki()
    return wiki.page_text(title)

def apiwiki_loader(title: str) -> str:
    wiki = wikipediaapi.Wikipedia('Diploma (m.tsalyk@ucu.edu.ua)', 'en')
    return wiki.page(title).text

def scrap_descriptions(df: pd. DataFrame) -> pd.DataFrame:
    if 'description' not in df.columns:
        df['description'] = None
    df.loc[df['description'].isna(), 'description'] = df.loc[df['description'].isna()]['title'].progress_map(get_movie_description)
    return df


if __name__ == '__main__':
    df = pd.read_csv('movies.csv')
    df = scrap_descriptions(df)
