import pytest
import pandas as pd

from mllabs import ExtDataProvider, Project


class TestExtDataProvider:
    def test_register_then_get_roundtrips(self, tmp_path):
        p = ExtDataProvider(tmp_path)
        p.register('kaggle_test', {'a': [1, 2, 3]})
        assert p.get('kaggle_test') == {'a': [1, 2, 3]}

    def test_get_unknown_name_raises(self, tmp_path):
        p = ExtDataProvider(tmp_path)
        with pytest.raises(KeyError, match='nope'):
            p.get('nope')

    def test_register_overwrites(self, tmp_path):
        p = ExtDataProvider(tmp_path)
        p.register('x', 1)
        p.register('x', 2)
        assert p.get('x') == 2

    def test_survives_reopen(self, tmp_path):
        ExtDataProvider(tmp_path).register('x', [1, 2, 3])
        assert ExtDataProvider(tmp_path).get('x') == [1, 2, 3]

    def test_nothing_cached_between_instances(self, tmp_path):
        """The whole point — a fresh provider over the same path reads what
        is actually on disk right now, not a stale in-memory copy."""
        first = ExtDataProvider(tmp_path)
        first.register('x', 'old')
        second = ExtDataProvider(tmp_path)
        first.register('x', 'new')
        assert second.get('x') == 'new'

    def test_remove_drops_the_file(self, tmp_path):
        p = ExtDataProvider(tmp_path)
        p.register('x', 1)
        p.remove('x')
        assert 'x' not in p
        with pytest.raises(KeyError):
            p.get('x')

    def test_remove_unknown_name_is_a_no_op(self, tmp_path):
        ExtDataProvider(tmp_path).remove('nope')  # does not raise

    def test_contains(self, tmp_path):
        p = ExtDataProvider(tmp_path)
        assert 'x' not in p
        p.register('x', 1)
        assert 'x' in p

    def test_names_sorted(self, tmp_path):
        p = ExtDataProvider(tmp_path)
        p.register('b', 1)
        p.register('a', 2)
        assert p.names() == ['a', 'b']

    def test_size_reports_bytes_written(self, tmp_path):
        p = ExtDataProvider(tmp_path)
        p.register('x', 'hello')
        assert p.size('x') > 0

    def test_size_of_unknown_name_is_none(self, tmp_path):
        assert ExtDataProvider(tmp_path).size('nope') is None

    def test_sizes_covers_every_name(self, tmp_path):
        p = ExtDataProvider(tmp_path)
        p.register('a', 1)
        p.register('b', 2)
        assert set(p.sizes()) == {'a', 'b'}

    def test_directory_created_if_missing(self, tmp_path):
        p = ExtDataProvider(tmp_path / 'nested' / 'ext_data')
        p.register('x', 1)
        assert p.get('x') == 1


class TestProjectExtData:
    def test_project_owns_an_ext_data_provider(self, tmp_path):
        project = Project(tmp_path / 'proj', data=pd.DataFrame({'a': [1]}))
        project.ext_data.register('kaggle_test', {'x': 1})
        assert project.ext_data.get('kaggle_test') == {'x': 1}

    def test_ext_data_survives_reopening_the_project(self, tmp_path):
        Project(tmp_path / 'proj', data=pd.DataFrame({'a': [1]})).ext_data.register('t', [1, 2])
        reopened = Project(tmp_path / 'proj', data=pd.DataFrame({'a': [1]}))
        assert reopened.ext_data.get('t') == [1, 2]
