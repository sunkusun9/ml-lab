import pytest
import pandas as pd

from mllabs import Resolver, ExtDataProvider, Collectors, Connector, Experimenter, PipelineBuilder


class TestResolverProcessorAndInstance:
    def test_processor_resolves_a_string_ref(self):
        from sklearn.tree import DecisionTreeClassifier
        assert Resolver().processor('sklearn.tree.DecisionTreeClassifier') is DecisionTreeClassifier

    def test_instance_resolves_a_ref_spec(self):
        inst = Resolver().instance('mllabs._connector.Connector')
        assert isinstance(inst, Connector)


class TestResolverParams:
    def test_plain_values_pass_through(self):
        assert Resolver().params({'a': 1, 'b': 'text', 'c': None}) == \
            {'a': 1, 'b': 'text', 'c': None}

    def test_empty_or_none_params_pass_through(self):
        assert Resolver().params(None) is None
        assert Resolver().params({}) == {}

    def test_callable_spec_resolves(self):
        from sklearn.metrics import accuracy_score
        resolved = Resolver().params({'metric_func': {'__callable__': 'sklearn.metrics.accuracy_score'}})
        assert resolved['metric_func'] is accuracy_score

    def test_ref_spec_instantiates(self):
        resolved = Resolver().params(
            {'connector': {'__ref__': 'mllabs._connector.Connector', '__params__': {'node_query': '^x'}}})
        assert isinstance(resolved['connector'], Connector)
        assert resolved['connector'].node_query == '^x'

    def test_ext_reference_resolves_via_provider(self, tmp_path):
        provider = ExtDataProvider(tmp_path)
        provider.register('kaggle_test', {'rows': 3})
        resolver = Resolver(ext_data=provider)
        assert resolver.params({'ext_data': '@ext:kaggle_test'}) == {'ext_data': {'rows': 3}}

    def test_ext_reference_without_provider_raises(self):
        with pytest.raises(ValueError, match='kaggle_test'):
            Resolver().params({'ext_data': '@ext:kaggle_test'})

    def test_ext_reference_nested_in_plain_dict_resolves(self, tmp_path):
        provider = ExtDataProvider(tmp_path)
        provider.register('x', [1, 2, 3])
        resolved = Resolver(ext_data=provider).params({'outer': {'inner': '@ext:x'}})
        assert resolved == {'outer': {'inner': [1, 2, 3]}}

    def test_ext_reference_nested_in_list_resolves(self, tmp_path):
        provider = ExtDataProvider(tmp_path)
        provider.register('x', 42)
        resolved = Resolver(ext_data=provider).params({'items': ['@ext:x', 'plain']})
        assert resolved == {'items': [42, 'plain']}

    def test_string_without_ext_prefix_is_not_touched(self, tmp_path):
        provider = ExtDataProvider(tmp_path)
        resolver = Resolver(ext_data=provider)
        assert resolver.params({'a': 'ext:not-a-match'}) == {'a': 'ext:not-a-match'}

    def test_unknown_ext_name_raises_key_error(self, tmp_path):
        resolver = Resolver(ext_data=ExtDataProvider(tmp_path))
        with pytest.raises(KeyError):
            resolver.params({'ext_data': '@ext:nope'})


class TestCollectorsWithResolver:
    def test_default_resolver_has_no_ext_data(self, tmp_path):
        reg = Collectors(tmp_path)
        with pytest.raises(ValueError, match='ExtDataProvider'):
            reg.set_collector('m', 'mock.EchoCollector', 'mllabs._connector.Connector',
                              params={'payload': '@ext:missing'})

    def test_injected_resolver_resolves_ext_data_on_registration(self, tmp_path):
        provider = ExtDataProvider(tmp_path / 'ext')
        provider.register('kaggle_test', {'x': 1})
        reg = Collectors(tmp_path / 'collectors', resolver=Resolver(ext_data=provider))
        obj = reg.set_collector('m', 'mock.EchoCollector', 'mllabs._connector.Connector',
                                params={'payload': '@ext:kaggle_test'})
        assert obj.payload == {'x': 1}

    def test_injected_resolver_resolves_ext_data_on_reopen(self, tmp_path):
        provider = ExtDataProvider(tmp_path / 'ext')
        provider.register('kaggle_test', {'x': 1})
        Collectors(tmp_path / 'collectors', resolver=Resolver(ext_data=provider)).set_collector(
            'm', 'mock.EchoCollector', 'mllabs._connector.Connector', params={'payload': '@ext:kaggle_test'})

        reopened = Collectors(tmp_path / 'collectors', resolver=Resolver(ext_data=provider))
        assert reopened.get_collector('m').payload == {'x': 1}

    def test_ext_data_change_is_picked_up_fresh_on_reopen(self, tmp_path):
        """Nothing is cached between Collectors instances — reopening re-resolves
        against whatever the provider currently holds."""
        provider = ExtDataProvider(tmp_path / 'ext')
        provider.register('x', 'first')
        Collectors(tmp_path / 'collectors', resolver=Resolver(ext_data=provider)).set_collector(
            'm', 'mock.EchoCollector', 'mllabs._connector.Connector', params={'payload': '@ext:x'})

        provider.register('x', 'second')
        reopened = Collectors(tmp_path / 'collectors', resolver=Resolver(ext_data=provider))
        assert reopened.get_collector('m').payload == 'second'


class TestResolverThroughExperimenterBuild:
    """The point of threading Resolver into _executor._process: an
    '@ext:name' param resolves for a Pipeline node exactly the same way it
    already does for a Collector."""

    def test_ext_reference_resolves_for_a_pipeline_node(self, tmp_path):
        provider = ExtDataProvider(tmp_path / 'ext')
        provider.register('greeting', 'hello')

        p = PipelineBuilder(path=tmp_path / 'pipeline')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('echo', processor='mock.EchoStage', method='fit_transform',
                  edges={'X': '{f1}'}, params={'payload': '@ext:greeting'})
        p.set_node('echo_node', grp='echo')
        pipeline = p.build()

        data = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0], 'target': [0, 1, 0, 1]})
        e = Experimenter(tmp_path / 'exp', 'run', data,
                         resolver=Resolver(ext_data=provider))
        e.set_pipeline(pipeline)
        e.build()

        obj, _ = e.get_objs('echo_node')
        assert obj.obj.payload == 'hello'

    def test_no_resolver_leaves_an_ext_reference_unresolved(self, tmp_path):
        """No resolver injected -> the default bare Resolver() has no
        ext_data, so an '@ext:' param raises instead of silently carrying
        the literal string through to the estimator."""
        p = PipelineBuilder(path=tmp_path / 'pipeline')
        p.set_datasource({'f1': 'numerical', 'target': 'binary'})
        p.set_grp('echo', processor='mock.EchoStage', method='fit_transform',
                  edges={'X': '{f1}'}, params={'payload': '@ext:greeting'})
        p.set_node('echo_node', grp='echo')
        pipeline = p.build()

        data = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0], 'target': [0, 1, 0, 1]})
        e = Experimenter(tmp_path / 'exp', 'run', data)
        e.set_pipeline(pipeline)
        e.build()

        errors = e.error_nodes(['echo_node'])
        assert len(errors) == 1
        assert 'ExtDataProvider' in errors[0]['message']
