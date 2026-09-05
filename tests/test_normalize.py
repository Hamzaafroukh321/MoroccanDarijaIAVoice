import json
from pathlib import Path
import pytest
from engine.normalize import normalize, phone, temporal


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/pizza.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('source,expected',[
    ('deux pizzas avec fromage','2 pizzas avec fromage'),
    ('vingt et un dirhams','21 MAD'),
    ('quatre-vingt-douze dh','92 MAD'),
    ('deux cent cinquante dirhams','250 MAD'),
    ('deux mille trois cents dh','2300 MAD'),
    ('06 12 34 56 78','0612345678'),
    ('+212 6 12 34 56 78','0612345678'),
    ('zero six douze trente quatre cinquante six soixante dix huit','0612345678'),
    ('٠٦١٢٣٤٥٦٧٨','0612345678'),
])
def test_normalize(config,source,expected):
    assert normalize(source,config)==expected


def test_owner_supplied_numeral_dictionary(config):
    # Synthetic token proves the mechanism without inventing a Darija translation.
    config['normalization']['darija_numbers']={'OWNER_ONE':1}
    assert normalize('OWNER_ONE large',config)=='1 large'


@pytest.mark.skip(reason='Native-speaker-reviewed Darija numeral fixtures have not been supplied.')
def test_reviewed_darija_numerals():
    pass


@pytest.mark.parametrize('source',['061234567','0812345678','+2120612345678','06abcdef78'])
def test_reject_invalid_phone(config,source):
    with pytest.raises(ValueError): phone(source,config['normalization'])


def test_dates_and_times(config):
    assert temporal('15/09/2026','date',config['normalization'])=='2026-09-15'
    assert temporal('9h30','time',config['normalization'])=='09:30'
    with pytest.raises(ValueError): temporal('31/02/2026','date',config['normalization'])
    config['normalization']['reference_date']='2026-09-04'
    assert normalize('rendez-vous demain',config)=='rendez-vous 2026-09-05'
