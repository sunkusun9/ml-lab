# Trials & Predictors

A `Trial` is a candidate the [Experimenter](experimenter.md) evaluates; a
`Predictor` is one the [Trainer](trainer.md) commits to training. Both resolve
to the same `ProcessorSpec` — what differs is the question each answers.

::: mllabs.Trial

::: mllabs.GridTrials

::: mllabs.ListTrials

::: mllabs.Predictor

::: mllabs.PredictorStore

## Comparing specs

`compare_specs` diffs any `{name: ProcessorSpec}` mapping — most often Trial
specs (`{t.name: t.get_spec() for t in trials}`), but Pipeline node specs work
the same way since both resolve to `ProcessorSpec`.

::: mllabs.compare_specs
