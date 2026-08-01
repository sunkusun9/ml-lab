import pickle as pkl
from pathlib import Path

from ._data_wrapper import wrap, unwrap
from ._edge_dsl import parse, eval_expr
from ._flow import InferenceDataFlow


class Inferencer:
    """Applies trained processors to new data for inference.

    Created by :meth:`~mllabs._trainer.Trainer.to_inferencer`. Self-contained —
    no dependency on Experimenter or Trainer at serve time.

    Attributes:
        node_specs (dict): ``{name: ProcessorSpec}`` for every selected
            Stage and Trial. Only ``edges`` is actually needed at serve time,
            so the Inferencer carries these rather than a whole Pipeline.
        selected_stages (list[str]): Stage node names, in topological order.
        selected_heads (list[str]): Trial names producing the output.
        n_splits (int): Number of cross-validation splits.
        node_objs (dict): ``{name: [processor_split0, ...]}``.
        v: Output column filter applied to Trial outputs.
    """

    def __init__(self, node_specs, selected_stages, selected_heads, n_splits, node_objs, v=None):
        self.node_specs = node_specs
        self.selected_stages = selected_stages
        self.selected_heads = selected_heads
        self.n_splits = n_splits
        self.node_objs = node_objs
        self.v = v

    def _make_flow(self, split_idx):
        flow = InferenceDataFlow()
        for name in self.selected_stages + self.selected_heads:
            flow.add_node(name, self.node_objs[name][split_idx],
                          self.node_specs[name].edges)
        return flow

    def _resolve_heads(self, nodes):
        if nodes is None:
            return self.selected_heads
        if isinstance(nodes, str):
            nodes = [nodes]
        unknown = [n for n in nodes if n not in self.selected_heads]
        if unknown:
            raise ValueError(f"Unknown head node(s): {unknown}")
        return [n for n in self.selected_heads if n in set(nodes)]

    def process(self, data, agg='mean', nodes=None):
        """Run inference on new data and aggregate across splits.

        Args:
            data: Input dataset (pandas/polars DataFrame or numpy array).
            agg (str | callable | None): Aggregation strategy across splits.
                ``'mean'`` (default), ``'mode'``, a callable receiving a list of
                per-split DataFrames, or ``None`` (returns list).
                Ignored when ``n_splits == 1``.
            nodes (str | list[str] | None): Head node(s) to include.
                ``None`` (default) uses all selected heads.

        Returns:
            DataFrame | list: Aggregated predictions, or a list of per-split
            predictions when ``agg=None``.
        """
        target_heads = self._resolve_heads(nodes)
        data = wrap(data)
        results = []
        for split_idx in range(self.n_splits):
            flow = self._make_flow(split_idx)
            head_outputs = []
            for name in target_heads:
                output = flow._resolve(data, name)
                if output is None:
                    continue
                if self.v is not None:
                    obj = self.node_objs[name][split_idx]
                    cols = eval_expr(parse(self.v), output, processor=obj)
                    output = output.select_columns(cols)
                head_outputs.append(output)
            if head_outputs:
                result = (head_outputs[0] if len(head_outputs) == 1
                          else type(head_outputs[0]).concat(head_outputs, axis=1))
                results.append(result)

        if not results:
            return None
        if self.n_splits == 1:
            return unwrap(results[0])
        if agg is None:
            return [unwrap(r) for r in results]
        elif agg == 'mean':
            return unwrap(type(results[0]).mean(iter(results)))
        elif agg == 'mode':
            return unwrap(type(results[0]).mode(iter(results)))
        elif callable(agg):
            return unwrap(agg(results))
        else:
            raise ValueError(f"Unknown agg: {agg}")

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path):
        """Serialize the Inferencer to a single file.

        Args:
            path (str | Path): Directory to save into. Creates
                ``{path}/__inferencer.pkl``.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_data = {
            'node_specs': self.node_specs,
            'selected_stages': self.selected_stages,
            'selected_heads': self.selected_heads,
            'n_splits': self.n_splits,
            'node_objs': self.node_objs,
            'v': self.v,
        }
        with open(path / '__inferencer.pkl', 'wb') as f:
            pkl.dump(save_data, f)

    @classmethod
    def load(cls, path):
        """Load a saved Inferencer from disk.

        Args:
            path (str | Path): Directory containing ``__inferencer.pkl``.

        Returns:
            Inferencer: Restored inferencer.
        """
        path = Path(path)
        with open(path / '__inferencer.pkl', 'rb') as f:
            save_data = pkl.load(f)
        return cls(
            save_data['node_specs'],
            save_data['selected_stages'],
            save_data['selected_heads'],
            save_data['n_splits'],
            save_data['node_objs'],
            save_data.get('v'),
        )
