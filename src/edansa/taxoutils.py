from typing import Dict, Union, Optional, Type
from collections import Counter
from collections.abc import MutableMapping


def find_upper_taxo(taxo):

    if '.' in taxo:
        taxo_a = taxo.split('.')
    else:
        taxo_a = taxo[:]

    if set(taxo_a) == set('X'):
        return '.'.join(taxo_a)
    if 'X' in taxo_a:
        taxo_a = [x if x != 'X' else '0' for x in taxo_a]
    # -1 because we do not change first bit
    for i in range(len(taxo_a) - 1):
        if taxo_a[-(i + 1)] == '0':
            continue
        else:
            taxo_a[-(i + 1)] = '0'
            break

    taxo_a = '.'.join(taxo_a)
    return taxo_a


def get_root_taxos(org_taxo):
    root_taxos = []
    upper_taxo = find_upper_taxo(org_taxo)
    previous_taxo = org_taxo
    while upper_taxo != previous_taxo:
        root_taxos.append(upper_taxo)
        previous_taxo = upper_taxo
        upper_taxo = find_upper_taxo(previous_taxo)
    return root_taxos


assert ['1.1.0', '1.0.0'] == get_root_taxos('1.1.1')


def row2yaml_codev1(row, excell_names2code):
    if row['Specific Category'] in ['Songb', 'SongB']:
        row['Specific Category'] = 'Songbird'

    if row['Category'] in ['Mamm']:
        row['Category'] = 'Mam'
    # 'S4A10288_20190729_033000_unknown.wav', # 'Anthro/Bio': 'Uknown', no other data

    if row['Anthro/Bio'] in ['Uknown', 'Unknown']:
        row['Anthro/Bio'] = ''

    code = [row['Anthro/Bio'], row['Category'], row['Specific Category']]

    # place X for unknown topology
    # '0' is reserved for 'other'
    code = [i if i != '' else 'X' for i in code]

    # place X for unknown topology
    # '0' is reserved for 'other'
    # print(code)
    code = [i if i != '' else 'X' for i in code]
    for c in code:
        if '/' in c:
            raise NotImplementedError(
                f"row has wrong info about categories, '/' found: {row}")

    if code == ['X', 'X', 'X']:
        yaml_code = 'X.X.X'
    elif code[2] != 'X':
        yaml_code = excell_names2code[code[2].lower()]
    elif code[1] != 'X':
        yaml_code = excell_names2code[code[1].lower()]
    elif code[0] != 'X':
        yaml_code = excell_names2code[code[0].lower()]
    else:
        print(code)
        raise ValueError(f'row does not belong to any toplogy: {row}')

    return yaml_code


def row2yaml_codev2(row, excell_names2code):
    '''
        
    
    '''
    # we are hard coding this because this function for a specific template
    #
    yaml_codes = []
    # excell_class_names = {
    #     'whim', 'amgp', 'shorebird', 'spsa', 'paja', 'hola', 'sngo', 'lalo',
    #     'savs', 'wipt', 'nora', 'atsp', 'wisn', 'wfgo', 'sesa', 'glgu', 'bird',
    #     'mam', 'loon', 'helo', 'auto', 'jet', 'corv', 'dgs', 'flare', 'bio',
    #     'bear', 'crane', 'fly', 'airc', 'hum', 'truck', 'rain', 'bug', 'mosq',
    #     'geo', 'mach', 'woof', 'deer', 'water', 'anth', 'songb', 'woop', 'car',
    #     'rapt', 'sil', 'seab', 'wind', 'owl', 'meow', 'mous', 'prop', 'grous',
    #     'weas', 'hare', 'shrew'
    # }

    # make keys case insensitive
    row_lower = {}
    for k, v in row.items():
        k_lower = k.lower()
        row_lower[k_lower] = v
        if k_lower not in excell_names2code and k_lower not in INFO_COLUMNS:
            print(f'excell_names2code missing categories: {k_lower}')

    for excell_class_name in excell_names2code:
        class_exists_or_not = row_lower.get(excell_class_name, None)
        if class_exists_or_not == 'N/A' or class_exists_or_not == '':
            continue
        if class_exists_or_not is None:
            continue
        if float(class_exists_or_not) == 1.0:
            yaml_code = excell_names2code[excell_class_name]
            yaml_codes.append(yaml_code)
        elif float(class_exists_or_not) == 0.0:
            pass
        else:
            raise ValueError(  # pragma: no cover
                f'row has wrong info about categories not 1 or 0: {row}')

    if len(yaml_codes) != len(set(yaml_codes)):
        print(row)
        raise Exception(
            f'input excell have non-unique class names, yaml code counts: {Counter(yaml_codes)}'
        )

    yaml_codes.sort()
    return yaml_codes


def megan_excell_row2yaml_code(row: Dict,
                               excell_names2code: Union[Dict, None] = None,
                               version='V2'):
    '''Megan style labels to nna yaml topology V1.

    Row is a mapping, with 3 topology levels, function starts from most specific
    category and goes to most general one, when a mapping is found, returns
    corresponding code such as 0.2.0 for plane.

    Args:
        row = dictinary with following keys
                'Anthro/Bio','Category','Specific Category'
        excell_names2code = mapping from names to topology code
        version = Version of the function, 
            v1: single taxonomy per sample
            v2: multi taxonomy per sample, returns a list

    '''
    if excell_names2code is None:
        excell_names2code = EXCELL_NAMES2TAXO_CODE

    if version == 'V1':
        yaml_code = row2yaml_codev1(row, excell_names2code)
    elif version == 'V2':
        # returns a list
        yaml_code = row2yaml_codev2(row, excell_names2code)
    else:
        raise ValueError(
            f'This version is not implemented at megan_excell_row2yaml_code {version}'
        )

    return yaml_code


# taxonomy YAML have an issue that leafes has a different structure then previous
# orders, I should change that.
class Taxonomy(MutableMapping):
    """A dictionary that holds taxonomy structure.

    transforms edge keys from x.y.z to just last bit z 
    
    """

    def __init__(self, *args, **kwargs):
        self._init_end = False
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        self._edges = self.flatten(self._store)
        self.shorten_edge_keys(self._store)
        self._init_end = True

    @property
    def edges(self,):
        """Property of _edges."""
        return self._edges

    @edges.setter
    def edges(self, value):
        if self._init_end:
            raise NotImplementedError('Edges and taxonomy are Immutable')
        else:
            self._edges = value

    @edges.getter
    def edges(self,):
        return self._edges

    def __getitem__(self, key):
        key = self._keytransform(key)
        if isinstance(key, list):
            data = self._store
            for k in key:
                data = data[k]
            return data
        return self._store[key]

        # trying to implement general access by single key or multiple with dot
        # current_order = self._store[key[0]]
        # if len(key)==1:
        #     return current_order
        # keys = self._store[self._keytransform(key)]
        # for k in keys[:-1]:
        #     current_order = current_order[k]

        # return current_order[key]

    def __setitem__(self, key, value):
        if self._init_end:
            raise NotImplementedError('You cannot update after initilization.')
        else:
            self._store[key] = value

    def __delitem__(self, key):
        if self._init_end:
            raise NotImplementedError('You cannot update after initilization.')
        else:
            del self._store[key]
        # del self._store[self._keytransform(key)]
        # self.edges = self.flatten(self._store)

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def _keytransform(self, key):
        if isinstance(key, str):
            return key.split('.')
        elif isinstance(key, list):
            return key
        return key

    def flatten(self, d):
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                deeper = self.flatten(val).items()
                out.update(deeper)
            else:
                out[key] = val
        return out

    def shorten_edge_keys(self, d):
        for key, val in list(d.items()):
            del d[key]
            if isinstance(val, dict):
                d[key.split('.')[-1]] = self.shorten_edge_keys(val)
            else:
                d[key.split('.')[-1]] = val
        return d


INFO_COLUMNS = [
    'clip_path',
    'data_version',
    'annotator',
    'region',
    'location',
    'comments',
    'file_name',
    'date',
    'duration_sec',
    'reviewed',
    'extra_tags',
    'start_date_time',
    'end_date_time',
    'prev_clip_path',
]
excell_all_headers = [
    'clip_path',
    'data_version',
    'annotator',
    'region',
    'location',
    'comments',
    'file_name',
    'start_date_time',
    'duration_sec',
    'end_date_time',
    'reviewed',
    'extra_tags',
    # 'WHIM',
    # 'AMGP',
    # 'SHOREBIRD',
    # 'SPSA',
    # 'PAJA',
    # 'HOLA',
    # 'SNGO',
    # 'LALO',
    # 'SAVS',
    # 'WIPT',
    # 'NORA',
    # 'ATSP',
    # 'WISN',
    # 'WFGO',
    # 'SESA',
    # 'GLGU',  # new from John's labels
    'Anth',
    'Bio',
    'Geo',
    'Sil',
    'Auto',
    'Airc',
    'Mach',
    'Flare',
    'Bird',
    'Mam',
    'Bug',
    'Wind',
    'Rain',
    'Water',
    'Truck',
    'Car',
    'Prop',
    'Helo',
    'Jet',
    'Corv',
    'SongB',
    'DGS',
    'Grous',
    'Crane',
    'Loon',
    'SeaB',
    'Owl',
    'Hum',
    'Rapt',
    'Woop',
    'ShorB',
    'Woof',
    'Bear',
    'Mous',
    'Deer',
    'Weas',
    'Meow',
    'Hare',
    'Shrew',
    'Mosq',
    'Fly',
]
excell_label_headers = [
    # 'WHIM',
    # 'AMGP',
    # 'SHOREBIRD',
    # 'SPSA',
    # 'PAJA',
    # 'HOLA',
    # 'SNGO',
    # 'LALO',
    # 'SAVS',
    # 'WIPT',
    # 'NORA',
    # 'ATSP',
    # 'WISN',
    # 'WFGO',
    # 'SESA',
    # 'GLGU',  # new from John's labels
    'Anth',
    'Bio',
    'Geo',
    'Sil',
    'Auto',
    'Airc',
    'Mach',
    'Flare',
    'Bird',
    'Mam',
    'Bug',
    'Wind',
    'Rain',
    'Water',
    'Truck',
    'Car',
    'Prop',
    'Helo',
    'Jet',
    'Corv',
    'SongB',
    'DGS',
    'Grous',
    'Crane',
    'Loon',
    'SeaB',
    'Owl',
    'Hum',
    'Rapt',
    'Woop',
    'ShorB',
    'Woof',
    'Bear',
    'Mous',
    'Deer',
    'Weas',
    'Meow',
    'Hare',
    'Shrew',
    'Mosq',
    'Fly'
]
label_headers_noaffect_on_silence = [
    'Geo',
    'Wind',
    'Rain',
    'Water',
]

EXCELL_NAMES2TAXO_CODE = {
    'sngo': '4.1.10.0',
    'lalo': '4.1.10.1',
    'savs': '4.1.10.2',
    'wipt': '4.1.10.3',
    'nora': '4.1.10.4',
    'atsp': '4.1.10.5',
    'wisn': '4.1.10.6',
    'wfgo': '4.1.10.7',
    'sesa': '4.1.10.8',
    'glgu': '4.1.10.9',
    'whim': '4.1.10.10',
    'amgp': '4.1.10.11',
    'ampi': '4.1.10.12',
    'amro': '4.1.10.13',
    'blue': '4.1.10.14',
    'cang': '4.1.10.15',
    'cora': '4.1.10.16',
    'core': '4.1.10.17',
    'eywa': '4.1.10.18',
    'fosp': '4.1.10.19',
    'gcsp': '4.1.10.20',
    'caja': '4.1.10.21',
    'gcth': '4.1.10.22',
    'gcrf': '4.1.10.23',
    'gwfg': '4.1.10.24',
    'hore': '4.1.10.25',
    'hola': '4.1.10.26',
    'leye': '4.1.10.27',
    'lbdo': '4.1.10.28',
    'nowh': '4.1.10.29',
    'pesa': '4.1.10.30',
    'ropt': '4.1.10.31',
    'sacr': '4.1.10.32',
    'sepl': '4.1.10.33',
    'smlo': '4.1.10.34',
    'snbu': '4.1.10.35',
    'wata': '4.1.10.36',
    'wesa': '4.1.10.37',
    'wcsp': '4.1.10.38',
    'wiwa': '4.1.10.39',
    'yewa': '4.1.10.40',
    'duck': '4.1.10.41',
    'goose': '4.1.10.42',
    'jaeger': '4.1.10.43',
    'ocwa': '4.1.10.44',
    'deju': '4.1.10.45',
    'anth': '0.0.0',
    'auto': '0.1.0',
    'car': '0.1.2',
    'truck': '0.1.1',
    'prop': '0.2.1',
    'helo': '0.2.2',
    'jet': '0.2.3',
    'mach': '0.3.0',
    'bio': '1.0.0',
    'bird': '1.1.0',
    'crane': '1.1.11',
    'corv': '1.1.12',
    'hum': '1.1.1',
    'shorb': '1.1.2',
    'rapt': '1.1.4',
    'owl': '1.1.6',
    'woop': '1.1.9',
    'bug': '1.3.0',
    'dgs': '1.1.7',
    'flare': '0.4.0',
    'fox': '1.2.4',
    'geo': '2.0.0',
    'grouse': '1.1.8',
    'grous': '1.1.8',
    'loon': '1.1.3',
    'mam': '1.2.0',
    'bear': '1.2.2',
    'plane': '0.2.0',
    'ptarm': '1.1.8',
    'rain': '2.1.0',
    'seab': '1.1.5',
    'mous': '1.2.1',
    'deer': '1.2.3',
    'woof': '1.2.4',
    'weas': '1.2.5',
    'meow': '1.2.6',
    'hare': '1.2.7',
    'shrew': '1.2.8',
    'fly': '1.3.2',
    'silence': '3.0.0',
    'sil': '3.0.0',
    'songbird': '1.1.10',
    'songb': '1.1.10',
    'unknown': 'X.X.X',
    'water': '2.2.0',
    'x': 'X.X.X',
    'airc': '0.2.0',
    'mosq': '1.3.1',
    'wind': '2.3.0',
}

# EXCELL_NAMES2CODE
TAXO_CODE2EXCELL_NAMES = {
    code: name for name, code in EXCELL_NAMES2TAXO_CODE.items()
}

if not len(EXCELL_NAMES2TAXO_CODE) == len(TAXO_CODE2EXCELL_NAMES):
    # print missing taxonomies, keys and values are swapped
    # print repeating values from EXCELL_NAMES2TAXO_CODE

    error_flag = False
    value_counts = Counter(EXCELL_NAMES2TAXO_CODE.values())
    for k, v in value_counts.items():
        if v == 2:
            if k in ['X.X.X', '1.2.4', '0.2.0', '3.0.0', '1.1.10']:
                continue
        if v == 3:
            # grous, grouse, ptarm
            if k in ['1.1.8']:
                continue
        if v == 1:
            continue
        print(k, v)  # Added print for v == 2 cases that are not excluded
        error_flag = True

    if error_flag:
        raise ValueError('There are repeating values in EXCELL_NAMES2TAXO_CODE')
