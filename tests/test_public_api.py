import types

import mllabs


CORE = ('Project', 'Experimenter', 'Trainer', 'Inferencer',
        'Pipeline', 'PipelineBuilder', 'ProcessorSpec',
        'Trial', 'Predictor', 'Connector', 'Collectors')


def test_all_names_resolve():
    assert [n for n in mllabs.__all__ if not hasattr(mllabs, n)] == []


def test_core_classes_are_exported():
    """Trainer was reachable only as mllabs._trainer.Trainer while its three
    peers were top-level, which is the asymmetry this pins down."""
    assert [n for n in CORE if n not in mllabs.__all__] == []


def test_no_public_name_is_left_out_of_all():
    exported = {
        n for n in dir(mllabs)
        if not n.startswith('_') and not isinstance(getattr(mllabs, n), types.ModuleType)
    }
    assert exported == set(mllabs.__all__)
