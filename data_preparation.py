import pandas as pd
from datetime import datetime
import numpy as np

_android = pd.read_csv('googleplaystore.csv')
_genres = set()
_digits = ['.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ]

def get_android():
    return _android


def find_median_of(df: pd.DataFrame, col_name: str) -> int:
    a = df[col_name][df[col_name] != 'Varies with device']
    a = sorted(a.apply(size_to_uniform_view))
    return a[int((len(a) + 1) / 2)]


def get_unique(df: pd.DataFrame, col_name: str) -> list:
    return df[col_name].unique()


def size_to_uniform_view(size: str) -> float:
    if size[-1].lower() == 'm':
        return float(size[:-1]) * 1024
    elif size[-1].lower() == 'k':
        return float(size[:-1])
    else:
        return 13312  # median of sorted size


def install(s: str) -> int:
    return int(s[:-1].replace(',', ''))


def type_to_int(t: str) -> int:
    if t == 'Free':
        return 0
    else:
        return 1


def price_to_float(price: str) -> float:
    if price == '0':
        return 0.0
    else:
        return float(price[1:])


def genre1_to_int(genre: str) -> int:
    return int(_genres[genre.split(';')[0]])


def genre2_to_int(genre: str) -> int:
    tmp = genre.split(';')
    if len(tmp) == 2:
        return _genres[tmp[1]]
    else:
        return 0


def hash(s: str) -> int:
    numbers = [int(n) for n in s.split(';')]
    mean = round(float(np.mean(numbers)))
    return int(s.replace(';', str(mean)))


def date_to_int(date: str) -> int:
    date_time = datetime.strptime(date, '%B %d, %Y')
    return int(round(date_time.timestamp()))


def android_ver_to_int(version) -> int:
    version = str(version)
    if version == 'Varies with device' or version == 'nan':
        return 41  # median of sorted android ver
    return int(version[:3].replace('.', ''))


def prepare_data() -> pd.DataFrame:

    # APP
    _android['App'] = _android['App'].apply(lambda s: len(s))
    _android.rename(columns={'App': 'App Name Length'}, inplace=True)

    # CATEGORY
    categories = get_unique(_android, 'Category')
    categories = dict(zip(categories, range(len(categories))))
    _android['Category'] = _android['Category'].apply(lambda c: categories[c])

    # RATING
    _android.dropna(axis=0, inplace=True)

    # REVIEWS
    _android['Reviews'] = _android['Reviews'].apply(lambda r: int(r))

    # SIZE
    _android['Size'] = _android['Size'].apply(lambda s: size_to_uniform_view(s))

    # INSTALLS
    _android['Installs'] = _android['Installs'].apply(install)

    # TYPE
    _android['Type'] = _android['Type'].apply(type_to_int)
    _android.rename(columns={'Type': 'Paid'}, inplace=True)

    # PRICE
    _android['Price'] = _android['Price'].apply(price_to_float)

    # CONTENT RATING
    ratings = get_unique(_android, 'Content Rating')
    ratings = dict(zip(ratings, range(len(ratings))))
    _android['Content Rating'] = _android['Content Rating'].apply(lambda r: ratings[r])

    # GENRES
    global _genres
    for g in get_unique(_android, 'Genres'):
        elements = g.split(';')
        for e in elements:
            _genres.add(e)

    _genres = dict(zip(sorted(_genres), range(len(_genres))))
    _android['Genres_1'] = _android['Genres'].apply(genre1_to_int)
    _android['Genres_2'] = _android['Genres'].apply(genre2_to_int)
    _android.drop('Genres', inplace=True, axis=1)

    # LAST UPDATED
    _android['Last Updated'] = _android['Last Updated'].apply(date_to_int)

    # CURRENT VER
    _android.drop('Current Ver', inplace=True, axis=1)  # too many variations

    # ANDROID VER
    _android['Android Ver'] = _android['Android Ver'].apply(android_ver_to_int)

    return _android
