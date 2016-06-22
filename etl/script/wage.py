# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from ddf_utils.str import to_concept_id
from ddf_utils.index import create_index_file


# configuration of file path
source = '../source/bulk_GWR_EN.csv'
out_dir = '../../'


def extract_entities_country(data):
    country = data[['Country_Code', 'Country_Label']].copy()
    country = country.drop_duplicates()
    country.columns = ['country', 'name']
    country['country'] = country['country'].map(to_concept_id)

    return country


def extract_entities_sex(data):
    sex = data[['Sex_Item_Code', 'Sex_Item_Label']].drop_duplicates().dropna().copy()
    sex.columns = ['sex', 'name']
    sex['sex'] = sex['sex'].map(to_concept_id)

    return sex


def extract_concepts(data):
    conc = data[['Indicator_Code', 'Indicator_Label']].copy()
    conc = conc.drop_duplicates()
    conc.columns = ['concept', 'name']
    conc['concept'] = conc['concept'].map(to_concept_id)
    conc['concept_type'] = 'measure'

    disc = pd.DataFrame([['name', 'Name', 'string'],
                         ['year', 'Year', 'time'],
                         ['country', 'Country', 'entity_domain'],
                         ['sex', 'Sex', 'entity_domain']], columns=conc.columns)

    all_conc = pd.concat([disc, conc])

    return all_conc


def extract_datapoints(data):
    dps = data[['Country_Code', 'Indicator_Code', 'Sex_Item_Code', 'Time', 'Obs_Value']].copy()
    dps.columns = ['country', 'concept', 'sex', 'year', 'value']
    dps['country'] = dps['country'].map(to_concept_id)
    dps['sex'] = dps['sex'].map(to_concept_id)
    dps['concept'] = dps['concept'].map(to_concept_id)

    gps = dps.groupby('concept').groups

    for k, idxs in gps.items():
        df = dps.ix[idxs]
        df = df.rename(columns={'value': k})

        yield k, df


if __name__ == '__main__':
    print('reading source files...')
    data = pd.read_csv(source)

    print('creating concepts files...')
    concepts = extract_concepts(data)
    path = os.path.join(out_dir, 'ddf--concepts.csv')
    concepts.to_csv(path, index=False)

    print('creating entities files...')
    country = extract_entities_country(data)
    path = os.path.join(out_dir, 'ddf--entities--country.csv')
    country.to_csv(path, index=False)

    sex = extract_entities_sex(data)
    path = os.path.join(out_dir, 'ddf--entities--sex.csv')
    sex.to_csv(path, index=False)

    print('creating datapoints files...')
    for k, df in extract_datapoints(data):
        if np.all(df['sex'].isnull()):  # sex column are all empty
            df_ = df[['country', 'year', k]]
            path = os.path.join(out_dir,
                                'ddf--datapoints--{}--by--country--year.csv'.format(k))
        else:
            assert not df['sex'].hasnans  # assert sex column don't have nans
            df_ = df[['country', 'sex', 'year', k]]
            path = os.path.join(out_dir,
                                'ddf--datapoints--{}--by--country--sex--year.csv'.format(k))

        df_.sort_values(by=['country', 'year']).to_csv(path, index=False)

    print('creating index file...')
    create_index_file(out_dir)

