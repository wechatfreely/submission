import pandas as pd
from datetime import datetime


def data_transfer(filename, start, end):
    with open(filename) as f:
        df = pd.read_csv(f, sep='\t')
    df['datetime'] = df[['date', 'time']].apply(lambda x: ' '.join(x), axis=1)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %I:%M:%S %p')
    mask = (df['datetime'] >= start) & (df['datetime'] < end) & (df['url'].str[-4:] == '.jpg')
    df = df.loc[mask]
    size = df['size'].sum()
    return size


if __name__ == "__main__":
    filename = 'log'
    start = datetime(2017, 8, 24)
    end = datetime(2017, 8, 26)
    size = data_transfer(filename, start, end)
    print(size)
