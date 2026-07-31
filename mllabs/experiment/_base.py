class BaseExperiment:
    """Skeleton for an experiment: a source of :class:`Trial` objects plus the
    Collectors those trials report into.

    Subclasses implement the trial contract:

    - :meth:`get_trial_nums` — how many trials this experiment will yield
    - :meth:`get_next_trial` — the next one
    - :meth:`reset` — rewind, so the sequence can be drained again

    The pull shape is deliberate: it leaves room for a later experiment type
    that *decides* the next trial from results collected so far. For now
    :meth:`get_trials` drains the whole sequence up front, which keeps the
    executor's existing "target list is known before dispatch" assumptions
    (worker count capping, GPU/CPU job partitioning, progress totals) intact.

    **No dependency on Experimenter, and none on live Collectors.** An
    Experiment names the Collectors it reports into and nothing more; whatever a
    Collector needs from an Experimenter (fold counts, target values, index) it
    resolves itself through its own ``on_attach``/``_setup`` hooks. That keeps an
    Experiment usable — and inspectable — without any dataset, split, or
    collector storage attached to it.

    Args:
        name (str): Experiment name. Used as the default Trial ``label``.
        collectors (list[str], optional): Names of the Collectors this
            Experiment reports into, resolved against a
            :class:`~mllabs.Collectors` registry at run time.
        tags (list[str], optional): Selection tags applied to produced Trials.
    """

    def __init__(self, name, collectors=None, tags=None):
        self.name = name
        self.tags = list(tags or [])
        self.collector_names = list(collectors or [])

    # ------------------------------------------------------------------
    # trial contract — subclass responsibility
    # ------------------------------------------------------------------

    def get_trial_nums(self):
        """Total number of trials :meth:`get_next_trial` will yield."""
        raise NotImplementedError

    def get_next_trial(self):
        """Next :class:`Trial` in the sequence."""
        raise NotImplementedError

    def reset(self):
        """Rewind the trial sequence. Default is a no-op."""
        return None

    def get_trials(self):
        """Drain the whole sequence into a list.

        Rewinds first, so calling this twice yields the same trials rather than
        an empty second pass.
        """
        self.reset()
        return [self.get_next_trial() for _ in range(self.get_trial_nums())]

    # ------------------------------------------------------------------
    # collectors — names only
    # ------------------------------------------------------------------

    def use_collector(self, *names):
        """Record Collector name(s) this Experiment reports into.

        Only names are kept. The instances live in a :class:`~mllabs.Collectors`
        registry that the caller passes to ``Experimenter.exp``, so several
        Experiments can share one registry — and one metrics store — while each
        Experiment stays pure definition.
        """
        for name in names:
            if name not in self.collector_names:
                self.collector_names.append(name)
        return self

    def drop_collector(self, name):
        if name in self.collector_names:
            self.collector_names.remove(name)
        return self

    def __repr__(self):
        return (f"<{type(self).__name__} {self.name!r} "
                f"collectors={self.collector_names}>")
