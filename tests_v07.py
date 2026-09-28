from __future__ import annotations

import json
import tempfile
from pathlib import Path

from portable_config import DEFAULT_SETTINGS, SettingsStore, read_bootstrap, resolve_data_root, write_bootstrap


def test_portable_root():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)/'app'; profile=Path(td)/'profile'; base.mkdir(); profile.mkdir()
        assert resolve_data_root('portable',base,profile)==base/'data'
        assert resolve_data_root('profile',base,profile)==profile
        custom=Path(td)/'custom'
        assert resolve_data_root('custom',base,profile,str(custom))==custom.resolve()


def test_bootstrap_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)
        p=write_bootstrap('custom',str(base/'library'),base)
        d=read_bootstrap(p)
        assert d['mode']=='custom'
        assert d['custom_root']==str(base/'library')


def test_settings_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'data'/'settings.json'
        s=SettingsStore(path)
        assert s.get('recursive') is True
        s.save({'workers':7,'theme':'light','last_source':'D:/Photos'})
        s2=SettingsStore(path)
        assert s2.get('workers')==7
        assert s2.get('theme')=='light'
        assert s2.get('last_source')=='D:/Photos'
        raw=json.loads(path.read_text(encoding='utf-8'))
        assert 'include_videos' in raw and 'recognition_threshold' in raw


def test_defaults_are_sane():
    assert DEFAULT_SETTINGS['include_images'] is True
    assert DEFAULT_SETTINGS['include_videos'] is True
    assert 1 <= int(DEFAULT_SETTINGS['workers']) <= 8
    assert .36 <= float(DEFAULT_SETTINGS['recognition_threshold']) <= .58


def run_all():
    tests=[test_portable_root,test_bootstrap_roundtrip,test_settings_roundtrip,test_defaults_are_sane]
    for fn in tests:
        fn(); print('PASS',fn.__name__)
    print('ALL V0.7 TESTS PASSED')
    return 0


if __name__=='__main__':
    raise SystemExit(run_all())
