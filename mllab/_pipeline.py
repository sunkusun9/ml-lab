import re
import pandas as pd
from ._describer import desc_pipeline, desc_node
from .adapter  import get_adapter
class PipelineGroup:
    def __init__(
        self, name, role, processor=None, edges=None, method=None, parent=None, adapter=None, params=None
    ):
        self.name = name
        self.role = role  # 'stage' or 'head'
        self.processor = processor
        self.edges = edges if edges is not None else {}
        self.method = method
        self.parent = parent  # parent group name (str)
        self.adapter = adapter
        self.params = params if params is not None else {}
        self.children = []  # child group names
        self.nodes = []  # node names in this group
        self.attrs = None

    def get_attrs(self, grps):
        if self.attrs is not None:
            return self.attrs
        if self.parent is None:
            parent_attrs = {
                'edges': {},
                'params': {},
                'processor': None,
                'method': None,
                'adapter': None
            }
        else:
            parent_attrs = grps[self.parent].get_attrs(grps)
        edges = self.edges.copy()
        if parent_attrs['edges'] is not None:
            for k, v in parent_attrs['edges'].items():
                edges[k] = edges.get(k, []) + v
        params = self.params.copy()
        if parent_attrs['params'] is not None:
            for k, v in parent_attrs['params'].items():
                if k not in params:
                    params[k] = v
        processor = parent_attrs['processor'] if self.processor is None else self.processor
        if self.adapter is None:
            if parent_attrs['adapter'] is not None:
                adapter = parent_attrs['adapter']
            else:
                adapter = None
        else:
            adapter = self.adapter
        self.attrs = {
            'name': self.name,
            'edges': edges,
            'parent': self.parent,
            'adapter': adapter,
            'params': params,
            'children': self.children,
        }
        for i in ['role', 'processor', 'method']:
            self.attrs[i] = parent_attrs.get(i) if getattr(self, i) is None else getattr(self, i)

        return self.attrs

    def update_attrs(self):
        self.attrs = None

    def copy(self):
        ret = PipelineGroup(
            self.name, self.role, self.processor, self.edges.copy(),
            self.method, self.parent, self.adapter, self.params.copy()
        )
        ret.children = self.children.copy()
        ret.nodes = self.nodes.copy()
        return ret

class PipelineNode:
    def __init__(
        self, name, grp, processor=None, edges=None, method=None, adapter=None, params=None
    ):
        self.name = name
        self.grp = grp  # group name (str)
        self.processor = processor
        self.edges = edges if edges is not None else {}
        self.method = method
        self.adapter = adapter
        self.params = params if params is not None else {}

        self.output_edges = []  # 이 노드를 입력으로 사용하는 노드들의 이름
        self.attrs = None

    def copy(self):
        ret = PipelineNode(
            self.name, self.grp, self.processor, self.edges.copy(),
            self.method, self.adapter, self.params.copy()
        )
        ret.output_edges = self.output_edges.copy()
        return ret

    def get_attrs(self, grps):
        if self.attrs is not None:
            return self.attrs
        grp_attrs = grps[self.grp].get_attrs(grps)
        edges = self.edges.copy()
        if grp_attrs['edges'] is not None:
            for k, v in grp_attrs['edges'].items():
                edges[k] = edges.get(k, []) + v
        params = self.params.copy()
        if grp_attrs['params'] is not None:
            for k, v in grp_attrs['params'].items():
                if k not in params:
                    params[k] = v
        processor = grp_attrs['processor'] if self.processor is None else self.processor
        if self.adapter is None:
            if grp_attrs['adapter'] is None:
                adapter = get_adapter(processor)
            else:
                adapter = grp_attrs['adapter']
        else:
            adapter = self.adapter
        self.attrs = {
            'name': self.name,
            'grp': self.grp,
            'edges': edges,
            'processor': processor,
            'adapter': adapter,
            'params': params,
            'method': grp_attrs.get('method') if self.method is None else self.method,
        }

        return self.attrs

    def update_attrs(self):
        self.attrs = None


class Pipeline:
    def __init__(self):
        self.nodes = {}
        self.grps = {}
        self.nodes = {None: PipelineNode("Data_Source", None, None, None, None, None)}

    def copy(self):
        ret = Pipeline()
        ret.grps = {k: v.copy() for k, v in self.grps.items()}
        ret.nodes = {k: v.copy() for k, v in self.nodes.items()}
        return ret

    def copy_stage(self):
        ret = Pipeline()

        stage_grp_names = {name for name, grp in self.grps.items() if grp.role == 'stage'}

        for name in stage_grp_names:
            grp = self.grps[name].copy()
            grp.children = [c for c in grp.children if c in stage_grp_names]
            if grp.parent not in stage_grp_names:
                grp.parent = None
            ret.grps[name] = grp

        stage_node_names = set()
        for name, node in self.nodes.items():
            if name is None:
                continue
            if node.grp in stage_grp_names:
                stage_node_names.add(name)

        for name in stage_node_names:
            node = self.nodes[name].copy()
            node.output_edges = [e for e in node.output_edges if e in stage_node_names]
            ret.nodes[name] = node

        data_source = self.nodes[None].copy()
        data_source.output_edges = [e for e in data_source.output_edges if e in stage_node_names]
        ret.nodes[None] = data_source

        return ret

    def copy_nodes(self, node_names):
        needed_nodes = set()
        queue = list(node_names)

        while queue:
            name = queue.pop(0)
            if name is None or name in needed_nodes:
                continue
            if name not in self.nodes:
                continue
            needed_nodes.add(name)
            attrs = self.nodes[name].get_attrs(self.grps)
            for edge_list in attrs.get('edges', {}).values():
                for edge_name, _ in edge_list:
                    if edge_name is not None and edge_name not in needed_nodes:
                        queue.append(edge_name)

        needed_grps = set()
        for name in needed_nodes:
            grp_name = self.nodes[name].grp
            while grp_name is not None and grp_name not in needed_grps:
                needed_grps.add(grp_name)
                grp_name = self.grps[grp_name].parent

        ret = Pipeline()

        for name in needed_grps:
            grp = self.grps[name].copy()
            grp.children = [c for c in grp.children if c in needed_grps]
            grp.nodes = [n for n in grp.nodes if n in needed_nodes]
            if grp.parent not in needed_grps:
                grp.parent = None
            ret.grps[name] = grp

        for name in needed_nodes:
            node = self.nodes[name].copy()
            node.output_edges = [e for e in node.output_edges if e in needed_nodes]
            ret.nodes[name] = node

        data_source = self.nodes[None].copy()
        data_source.output_edges = [e for e in data_source.output_edges if e in needed_nodes]
        ret.nodes[None] = data_source

        return ret

    def _validate_name(self, name):
        if name is None:
            return

        if '__' in name:
            raise ValueError(f"Name '{name}' cannot contain '__'")

        invalid_chars = ['/', '\\', '\0', '<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            if char in name:
                raise ValueError(f"Name '{name}' cannot contain '{char}'")

    def _find_descendants(self, node_name):
        descendants = set()
        queue = [node_name]

        while queue:
            current = queue.pop(0)

            if current not in self.nodes:
                continue

            for child_name in self.nodes[current].output_edges:
                if child_name not in descendants:
                    descendants.add(child_name)
                    queue.append(child_name)

        return descendants

    def _check_cycle(self, node_name, new_edges):
        descendants = self._find_descendants(node_name)

        cycle_edges = []
        for key, edge_list in new_edges.items():
            for edge_name, _ in edge_list:
                if edge_name is None:
                    continue

                if edge_name not in self.nodes:
                    continue

                if edge_name in descendants:
                    cycle_edges.append(edge_name)

        if cycle_edges:
            return True, cycle_edges
        return False, []

    def _check_edges(self, edges):
        if edges is None or len(edges) == 0:
            return False
        for key, edge_list in edges.items():
            for name, _ in edge_list:
                if name is None:
                    continue
                if name not in self.nodes:
                    raise ValueError(f"Edge node '{name}' not found")
                node_grp = self.nodes[name].grp
                if self.grps[node_grp].role != 'stage':
                    raise ValueError(f"Edge node '{name}' must be a stage node, got '{self.grps[node_grp].role}'")
        return True

    def _get_all_nodes_in_grp(self, grp):
        result = list(grp.nodes)
        for child_name in grp.children:
            child_grp = self.grps[child_name]
            result.extend(self._get_all_nodes_in_grp(child_grp))
        return result

    def _get_affected_nodes(self, nodes):
        priorities = {}
        queue = []

        for node_name in nodes:
            priorities[node_name] = 1
            queue.append((node_name, 1))

        while queue:
            current_node, current_priority = queue.pop(0)

            descendants = self._find_descendants(current_node)

            for desc_node in descendants:
                new_priority = current_priority + 1
                if desc_node not in priorities or priorities[desc_node] < new_priority:
                    priorities[desc_node] = new_priority
                    queue.append((desc_node, new_priority))

        sorted_nodes = sorted(priorities.items(), key=lambda x: x[1])
        return [i[0] for i in sorted_nodes if i[0] is not None]

    def set_grp(
            self, name, role=None, processor=None, edges=None, method=None, parent=None, adapter=None, params=None, exist='skip'
        ):
        self._validate_name(name)
        if name in self.nodes:
            raise ValueError(f"Name '{name}' already exists as a node")
        if edges is None:
            edges = {}

        if parent is not None:
            if parent not in self.grps:
                raise ValueError(f"Parent group '{parent}' not found")
            if role is None:
                role = self.grps[parent].role
        if role not in ['stage', 'head']:
            raise ValueError(f"Role must be 'stage' or 'head', got '{role}'")

        if name not in self.grps:
            self._check_edges(edges)
            grp = PipelineGroup(
                name, role, processor=processor, edges=edges, method=method, parent=parent, adapter=adapter, params=params
            )

            if parent is not None:
                self.grps[parent].children.append(name)

            self.grps[name] = grp
            return {
                "result": "new", "grp": grp, "affected_nodes": list()
            }
        elif exist == 'skip':
            grp = self.grps[name]
            return {"result": "skip", "grp": grp, "affected_nodes": list()}
        elif exist == 'error':
            raise ValueError(f"Group '{name}' already exists.")

        old_grp = self.grps[name]
        if old_grp.role != role:
            raise ValueError(f"Cannot change role of group '{name}': existing '{old_grp.role}', requested '{role}'")
        grp = old_grp.copy()

        parent_changed = False
        old_parent = old_grp.parent

        if old_parent != parent:
            parent_changed = True
            if old_parent is not None:
                self.grps[old_parent].children.remove(name)
            grp.parent = parent
            if parent is not None:
                self.grps[parent].children.append(name)

        if processor is not None:
            grp.processor = processor
        if edges is not None and len(edges) > 0:
            grp.edges = edges
        if method is not None:
            grp.method = method
        if adapter is not None:
            grp.adapter = adapter
        if params is not None:
            grp.params.update(params)

        grp.update_attrs()
        attrs = grp.get_attrs(self.grps)
        new_edges = attrs['edges']
        if len(new_edges) > 0:
            affected_nodes = self._get_all_nodes_in_grp(grp)

            for node_name in affected_nodes:
                if node_name not in self.nodes:
                    continue

                node = self.nodes[node_name]
                node.update_attrs()
                node_attrs = node.get_attrs(self.grps)
                node_own_edges = node_attrs.get('edges', {})

                final_edges = {k: list(v) for k, v in new_edges.items()}
                for k, v in node_own_edges.items():
                    if k in final_edges:
                        final_edges[k].extend(v)
                    else:
                        final_edges[k] = list(v)

                has_cycle, cycle_edges = self._check_cycle(node_name, final_edges)
                if has_cycle:
                    cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
                    raise ValueError(f"Cannot update group '{name}': node '{node_name}' would create cycle through edge(s) {cycle_info}")
        else:
            affected_nodes = list()

        self.grps[name] = grp
        return {
            "result": "update", "affected_nodes": affected_nodes, "old_grp": old_grp, "grp": grp
        }
    
    def get_grp(self, name):
        return self.grps.get(name, None)
    
    def rename_grp(self, name_from, name_to):
        self._validate_name(name_to)

        if name_from not in self.grps:
            raise ValueError(f"Group '{name_from}' not found")
        if name_to in self.grps:
            raise ValueError(f"Group '{name_to}' already exists")

        old_grp = self.grps[name_from]
        grp = old_grp.copy()
        grp.name = name_to
        if grp.parent is not None:
            self.grps[grp.parent].children.remove(name_from)
            self.grps[grp.parent].children.append(name_to)

        for node_name in grp.nodes:
            self.nodes[node_name].grp = name_to
            self.nodes[node_name].update_attrs()

        for child_name in grp.children:
            self.grps[child_name].parent = name_to
            self.grps[child_name].update_attrs()

        del self.grps[name_from]
        self.grps[name_to] = grp

    def remove_grp(self, name):
        if name not in self.grps:
            raise ValueError(f"Group '{name}' not found")

        grp = self.grps[name]

        if len(grp.children) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.children)} child group(s)")

        if len(grp.nodes) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.nodes)} node(s)")

        if grp.parent is not None:
            self.grps[grp.parent].children.remove(name)

        del self.grps[name]

    def get_parents(self, node_name):
        if node_name not in self.nodes:
            return []

        node = self.nodes[node_name]
        if node.grp is None:
            return []

        result = []
        current_grp = self.grps.get(node.grp)

        while current_grp is not None:
            result.append(current_grp.name)
            current_grp = self.grps.get(current_grp.parent) if current_grp.parent else None

        return result

    def get_node_names(self, query):
        if query is None:
            node_names = list(self.nodes.keys())
        elif isinstance(query, list):
            node_names = [n for n in query if n in self.nodes]
        elif isinstance(query, str):
            pat = re.compile(query)
            node_names = [k for k in self.nodes.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"query must be None, list, or str, got {type(query)}")
        return node_names

    def remove_node(self, name):
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found")

        if name is None:
            raise ValueError("Cannot remove DataSource node")

        descendants = self._find_descendants(name)
        if descendants:
            descendants_list = sorted(descendants)
            raise ValueError(f"Cannot remove node '{name}': has {len(descendants)} dependent node(s): {descendants_list}")

        node = self.nodes[name]
        node_attr = node.get_attrs(self.grps)
        self._update_output_edges(name, node_attr['edges'], None)

        grp_name = node.grp
        if grp_name is not None and grp_name in self.grps:
            grp = self.grps[grp_name]
            if name in grp.nodes:
                grp.nodes.remove(name)

        del self.nodes[name]

    def _update_output_edges(self, node_name, old_edges, new_edges):
        if old_edges is not None:
            for key, edge_list in old_edges.items():
                for edge_name, _ in edge_list:
                    if edge_name in self.nodes:
                        parent_node = self.nodes[edge_name]
                        if node_name in parent_node.output_edges:
                            parent_node.output_edges.remove(node_name)

        if new_edges is not None:
            for key, edge_list in new_edges.items():
                for edge_name, _ in edge_list:
                    if edge_name in self.nodes:
                        parent_node = self.nodes[edge_name]
                        if node_name not in parent_node.output_edges:
                            parent_node.output_edges.append(node_name)

    def set_node(
        self, name, grp, processor=None, edges=None, method=None, adapter=None, params=None, exist='skip'
    ):
        self._validate_name(name)

        if name in self.grps:
            raise ValueError(f"Name '{name}' already exists as a group")

        if grp not in self.grps:
            raise ValueError(f"Group '{grp}' not found")

        if edges is None:
            edges = {}
        if params is None:
            params = {}

        self._check_edges(edges)

        is_update = name in self.nodes
        if is_update:
            if exist == 'skip':
                return {'result': 'skip', 'affected_nodes': [], 'old_obj': self.nodes[name], 'obj': self.nodes[name]}
            elif exist == 'error':
                raise ValueError(f"Node '{name}' already exists.")

        old_edges = None
        old_output_edges = None
        old_node = None
        if is_update:
            old_node = self.nodes[name]
            old_edges = old_node.get_attrs(self.grps)['edges']
            old_output_edges = old_node.output_edges

        node = PipelineNode(
            name, grp, processor, edges, method=method, adapter=adapter, params=params
        )

        grp_obj = self.grps[grp]
        attrs = node.get_attrs(self.grps)

        if attrs.get('processor') is None:
            raise ValueError(f"Cannot create node '{name}': processor is required")

        if attrs.get('method') is None:
            raise ValueError(f"Cannot create node '{name}': method is required")

        if len(attrs.get('edges', {})) == 0:
            raise ValueError(f"Cannot create node '{name}': edges is required")

        has_cycle, cycle_edges = self._check_cycle(name, attrs['edges'])
        if has_cycle:
            cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
            raise ValueError(f"Cannot add node '{name}': would create cycle through edge(s) {cycle_info}")

        self._update_output_edges(name, old_edges, attrs['edges'])

        if old_output_edges is not None:
            node.output_edges = old_output_edges

        if name not in grp_obj.nodes:
            grp_obj.nodes.append(name)

        if is_update:
            affected_nodes = list(self._find_descendants(name))
            old_grp_name = old_node.grp
            if old_grp_name != grp and old_grp_name in self.grps:
                old_grp = self.grps[old_grp_name]
                if name in old_grp.nodes:
                    old_grp.nodes.remove(name)
        else:
            affected_nodes = list()

        self.nodes[name] = node

        return {
            'result': 'update' if is_update else 'new',
            'affected_nodes': affected_nodes,
            'old_obj': old_node,
            'obj': node
        }

    def get_node(self, name):
        return self.nodes.get(name, None)

    def get_node_attrs(self, name):
        node = self.get_node(name)
        return node.get_attrs(self.grps)

    def desc_pipeline(self, max_depth=None, direction='TD'):
        """파이프라인 구조를 Mermaid Markdown으로 반환

        Args:
            max_depth: 최대 표시 깊이 (None이면 무제한)
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
        """
        return desc_pipeline(self, max_depth, direction)

    def compare_nodes(self, nodes):
        attrs_map = {n: self.get_node_attrs(n) for n in nodes}

        groups = {}
        for name in nodes:
            proc = attrs_map[name]['processor']
            proc_name = proc.__name__ if proc is not None else 'None'
            groups.setdefault(proc_name, []).append(name)

        result = {}
        for proc_name, group_nodes in groups.items():
            rows = {name: {} for name in group_nodes}

            # params
            all_param_keys = sorted({k for n in group_nodes for k in attrs_map[n]['params']})
            for name in group_nodes:
                params = attrs_map[name]['params']
                for k in all_param_keys:
                    rows[name][('params', k)] = params.get(k, None)

            # edges (X only) - stage node별 변수 비교
            stage_vars = {}
            for name in group_nodes:
                x_entries = attrs_map[name]['edges'].get('X', [])
                for sn, var_spec in x_entries:
                    if sn not in stage_vars:
                        stage_vars[sn] = {}
                    if name not in stage_vars[sn]:
                        stage_vars[sn][name] = []
                    if var_spec is None:
                        stage_vars[sn][name].append(None)
                    elif isinstance(var_spec, (list, tuple)):
                        stage_vars[sn][name].extend(var_spec)
                    else:
                        stage_vars[sn][name].append(var_spec)

            for sn, node_vars in stage_vars.items():
                sn_str = str(sn) if sn is not None else 'DataSource'
                for name in group_nodes:
                    if name not in node_vars:
                        node_vars[name] = []

                repr_map = {}
                var_sets = {}
                for name in group_nodes:
                    s = set()
                    for v in node_vars[name]:
                        r = repr(v)
                        s.add(r)
                        repr_map[r] = v
                    var_sets[name] = s

                if len({frozenset(s) for s in var_sets.values()}) <= 1:
                    continue

                non_empty = [s for s in var_sets.values() if s]
                common_reprs = set.intersection(*non_empty) if non_empty else set()
                common_vars = sorted([repr_map[r] for r in common_reprs], key=repr)
                col_2 = f"{sn_str} [{', '.join(str(v) for v in common_vars)}]" if common_vars else sn_str

                for name in group_nodes:
                    diff_reprs = var_sets[name] - common_reprs
                    diff_vars = sorted([repr_map[r] for r in diff_reprs], key=repr)
                    rows[name][('X', col_2)] = diff_vars if diff_vars else []

            df = pd.DataFrame.from_dict(rows, orient='index')
            if len(df.columns) > 0:
                df.columns = pd.MultiIndex.from_tuples(df.columns)
                diff_cols = [c for c in df.columns if len({repr(v) for v in df[c]}) > 1]
                df = df[diff_cols]
            result[proc_name] = df

        return result

    def desc_node(self, node_name, direction='TD', show_params=False):
        """특정 노드까지의 연결 구조를 Mermaid Markdown으로 반환

        Args:
            node_name: 대상 노드 이름
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
            show_params: True이면 노드의 파라미터 정보를 표시 (default: False)
        """
        return desc_node(self, node_name, direction, show_params)