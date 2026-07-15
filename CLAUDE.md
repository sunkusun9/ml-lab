# CLAUDE 동작
니가 구사한 코드는 왠만한 건 다 파악 가능해. 주석은 나중에 한꺼번에 만들꺼야, 만들지마
함수나 메소드 가이드도 나중에 할꺼야.

CLAUDE.md에서 불필요하게 토큰을 낭비 하지 않도록, 작업 내역의 개요를 확인해라
작업 관리는 GitHub Issues로 한다. TODO.md 같은 파일은 만들지 마라.
Git 관련 내용(커밋 메시지, PR, 이슈 코멘트)은 영어로 작성한다.
커밋 메시지에 "Co-Authored-By" 넣지 마라. PR에 "Generated with Claude Code" 같은 광고성 메시지 넣지 마라.

코드 검증은 `tests/`에서 적절한 `.py`를 찾아 테스트 케이스를 추가하여 진행한다. `python -c` 같은 임시 실행은 하지 않는다.

## CLI 버전
- git 2.43.0
- gh 2.45.0

## gh CLI 주의사항
- `gh issue view <num>` 는 Projects Classic 지원 deprecated 경고로 exit code 1 반환 → **반드시 `--json` 플래그 사용**
  - 예: `gh issue view 40 --json title,body,comments`
- `--repo` 플래그 없이도 현재 디렉토리의 remote origin에서 자동 추론됨

# modeler 모듈 요약

## 아키텍처 개요
- **Pipeline** (`_pipeline.py`): 노드 그래프 자료구조 (ML 관심사 분리)
- **Experimenter** (`_experimenter.py`): 실험 실행/관리 (Pipeline 사용)
- **Trainer** (`_trainer.py`): 학습 실행/관리 (split 기반)
- **Inferencer** (`_inferencer.py`): 학습된 파이프라인을 새 데이터에 적용
- **NodeStore** (`_store.py`): 노드 아티팩트 읽기/쓰기 (obj.pkl / result.pkl / info.pkl)
- **DataFlow / TrainDataFlow** (`_flow.py`): fold별 데이터 흐름 및 stage 빌드
- **_executor.py**: `_build_flow_single/multi`, `_experiment_single/multi` — 실제 빌드/실험 실행

## Node/Experiment 상태 모델

### Node 4-State
`init → built → finalized` / `init → error → (reset) → init`

| 상태 | Disk | 설명 |
|------|------|------|
| **init** | - | Pipeline에 정의만 된 상태 |
| **built** | O | 빌드 완료, 결과 추출 가능 |
| **finalized** | info only | 결과 추출 완료, obj/result 삭제 (Head 전용) |
| **error** | info only | 빌드/실험 중 에러 발생, 내역 보존 |

- Stage는 finalize 불가 (하위 노드에 데이터 지속 공급)

### Experiment 2-State
`open → closed`
- **open**: Stage/Head 객체 유지, Collector 데이터 유지
- **closed**: `close_exp` 호출 → Stage 객체까지 일괄 정리, Collector 데이터는 잔존

## 핵심 클래스

### Node 역할
- **DataSource** (`DataSourceNode`, key=`None`): 원본 데이터 스키마 및 target 정의
- **Stage**: 전처리/변환 (TransformProcessor) — 하위 노드에 데이터 공급
- **Head**: 모델링/예측 (PredictProcessor) — 최종 결과 생산

### Pipeline 계층 (`_pipeline.py`)
- `VAR_TYPES = frozenset({'numerical', 'ordinal', 'nominal', 'text', 'binary', 'datetime'})`
- **`_params_equal(a, b)`**: params dict 안전 비교 헬퍼
  - dict → key별 재귀 비교
  - `__dict__` 있는 객체 → `__dict__` 재귀 비교 (lgb_early_stopping 등 콜백 처리)
  - `__dict__` 없는 객체(primitive, C-ext) → `==` fallback (try/except)
  - 같은 타입이어야 equal 가능, 다른 타입 → False
- **Pipeline**: 노드 그래프 자료구조. Experimenter/Trainer를 소유/등록하지 않음 (아래 참조) — 순수 노드 그래프 정의 + serial 무결성 추적 역할만 담당
  - `nodes`: `{name: PipelineNode}` (`None` → `DataSourceNode`), `grps`: `{name: PipelineGroup}` (`'__datasource__'` 항상 존재)
  - `datasource`: `nodes[None]` 반환 property
  - `set_datasource(schema, targets=None)`: DataSource 스키마/target 설정, 변경 시 downstream serial 자동 bump
  - `set_grp(exist='diff'|'skip'|'error'|'replace')`, `set_node(exist=...)`, `rename_grp`, `remove_grp`, `remove_node`
  - `get_node_names(query)`, `get_node_attrs(name)`, `_get_affected_nodes(nodes)`
  - `_bump_serials(node_names)`: 지정 노드들의 serial을 새 UUID로 교체
  - `copy()`, `copy_stage()`, `copy_nodes(node_names)` — 선택적 복사
  - `compare_nodes(nodes)` → `{processor_name: DataFrame}` (params 차이 + edges['X'] stage별 변수 차이)
  - `check_data_compatibility(data)`: datasource schema 컬럼이 모두 `data`에 있는지 체크 (schema 미정의 시 no-op). `Experimenter.build/exp`, `Trainer.select_head/train` 시작 시 자동 호출
  - `desc_pipeline(max_depth, direction)`, `desc_node(node_name, direction, show_params)`: Mermaid 다이어그램
  - **Pipeline은 Experimenter/Trainer를 소유하지 않음** — `add_experiment`/`add_trainer`/`get_experiment`/`get_trainer` 등 registry 없음. `Experimenter`/`Trainer`는 사용자가 직접 생성해 변수로 들고 있다가, `build`/`exp`/`train` 등 호출 시마다 `pipeline`을 인자로 넘김 (아래 참조)

- **DataSourceNode** (`PipelineNode` 서브클래스):
  - `schema`: `{col: var_type}` — var_type은 VAR_TYPES 중 하나
  - `targets`: `list[str]` — 타겟 컬럼 목록 (타입과 별도)
  - `get_attrs(grps)`: role='datasource', serial, schema, targets 반환 (processor/edges/method/params 없음)

- **PipelineGroup**: 노드 그룹 (stage/head 역할)
  - 속성: `name`, `role`, `processor`, `edges`, `method`, `parent`, `adapter`, `params`, `desc`
  - `children`: 자식 그룹명 리스트, `nodes`: 소속 노드명 리스트
  - `get_attrs(grps)`: 상위 그룹 속성 병합하여 반환 (`desc`는 상속 안 됨, 각 요소 독립)
  - `diff(processor, edges, method, parent, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외 → desc-only 변경은 rebuild 미유발)

- **PipelineNode**: 개별 노드
  - 속성: `name`, `grp`, `processor`, `edges`, `method`, `adapter`, `params`, `desc`, **`serial`** (UUID str), **`tag`** (`list[str]`)
  - `serial`: 노드 정의가 변경될 때마다 `_bump_serials`에 의해 새 UUID로 교체 → 아티팩트 무결성 추적
  - `output_edges`: 이 노드를 입력으로 사용하는 노드명 리스트
  - `get_attrs(grps)`: 그룹 속성과 노드 속성 병합 (`serial` 포함)
  - `diff(grp, processor, edges, method, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외)
  - `set_grp`/`set_node`: `desc` 파라미터 수락; exist='diff' skip 경로에서도 `desc`는 업데이트됨

### Experimenter (`_experimenter.py`)
- 생성자: `(data, path, ..., cache_maxsize=4GB, logger, aug_data=None, tags=None)`
- **Pipeline을 보유하지 않음** — `self.pipeline`/`attach()` 없음. 노드 그래프가 필요한 메소드마다 `pipeline`을 인자로 명시적으로 전달하고, 매 호출마다 노드 `serial` 비교로 디스크 아티팩트를 그 pipeline과 동기화함
- `tags` (list[str]): `exp(pipeline)`를 `nodes=None`으로 호출하면 tag가 교집합인 head만 대상 (비어있으면 전체 head)
- `cache`: DataCache (LRU, 용량 기반) — `_cache.py`에 분리
- **pipeline 필요**:
  - `build(pipeline, nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None)` (stage), `exp(pipeline, nodes=None, finalize=False, n_jobs=1, gpu_id_list=None)` (head)
    - 시작 시 `pipeline.check_data_compatibility(self.data)` 호출 후 serial mismatch 노드 자동 감지 → `reset_nodes()` 후 재빌드
  - `collect(pipeline, collector, nodes=None, exist='skip')`: ad-hoc 수집 (빌드 완료된 head 노드 대상, nodes로 범위 제한 가능, progress 포함)
  - `get_collect_status(pipeline, collector, nodes=None)`: `{node: status}` 반환 — `'collected'`/`'not_collected'`/`'finalized'`/`'error'`
  - `reopen_exp(pipeline)`: closed→open, Stage 노드 초기화 후 `build(pipeline)` 재호출
  - `get_node_info(pipeline)`: 노드 요약 Markdown
- **pipeline 불필요** (디스크 상태만으로 동작): `get_status(node_name)`, `finalize(nodes)`, `reinitialize(nodes)`, `close_exp()`, `reset_nodes(nodes)`, `show_error_nodes(nodes=None, traceback=False)`, `get_objs(node_name, outer_idx=0, inner_idx=0)`
  - **주의**: 같은 fold의 `train_data_flows[j]`와 `artifact_stores[j]`는 디스크상 동일 디렉토리를 가리키지만 서로 독립적인 lazy info 캐시(`NodeStore._info_cache`)를 가짐. 양쪽을 합쳐 하나의 상태로 캐싱한 뒤 한쪽만 mutate(예: finalize)하면 다른 쪽 캐시가 stale해짐 — 그래서 `get_status`는 pipeline으로 role을 판별해 해당 store 하나만 조회함(`_reset_serial_stale_nodes`도 동일 패턴). `finalize`/`reinitialize`/`reset_nodes`/`close_exp`는 store별로 조회 직후 바로 그 store에 실행(check-and-act, 캐싱된 판단을 다른 store에 넘기지 않음)하는 구조라 pipeline 없이도 안전함
- `add_collector(collector, exist='skip')`: Collector **등록만** (path 설정, save) — pipeline 불필요, **자동 수집 안 함**. 이미 빌드된 head에서 즉시 수집하려면 별도로 `collect(pipeline, collector)` 호출
- `get_collector(name)`: Collector 반환 (없으면 None)
- `remove_collector(name)`: Collector 제거 후 `_save()`
- `get_train_data(edges, o_idx=0, i_idx=0)` / `get_valid_data(...)` / `get_test_data(...)`, `get_node_train_data(pipeline, node, o_idx=0, i_idx=0)` / `get_node_valid_data(...)` / `get_node_test_data(...)`: 노드 출력 추출 헬퍼
- `aug_data`: 외부 데이터를 DataSource 수준에서 inner train split에 append — 미퍼시스트, create/load 시 전달
- 저장/로드: `_save()`, `load(filepath, data, data_key)` — pipeline은 저장 대상 아님(직접 보유 안 하므로); 로드 후 매 호출 시 pipeline을 인자로 전달
  - collector_keys, tags 저장/복원

### DataCache (`_cache.py`)
- `cachetools.LRUCache` 기반, 용량(bytes) 단위 관리
- `get_data(node, typ, idx)`, `put_data(node, typ, idx, data)`
- `clear_nodes(nodes)`: 특정 노드들의 캐시 삭제

### NodeStore (`_store.py`)
- fold 경로 아래 노드별 아티팩트 관리: `{path}/{node_name}/`
  - `obj.pkl` — processor 객체
  - `result.pkl` — fit_transform/fit_predict 출력
  - `info.pkl` — `{status, build_id, node_serial, fit_time, edges, train_shape, ...}`
- `status(name)`: `None`(init) / `'built'` / `'finalized'` / `'error'`
- `get_info(name)`: info dict (lazy cache), `node_serial` 키로 serial 추적
- `finalize(name)`: obj/result 삭제, info status → 'finalized'
- `reset_node(name)`: 디렉토리 전체 삭제, cache 무효화

### DataFlow / TrainDataFlow (`_flow.py`)
- **DataFlow** (NodeStore 상속): 디스크에서 stage processor 로드, 소스 데이터를 stage 그래프로 변환
  - `node_objs`: `{name: (obj, result, info)}`, `_node_edges`: `{name: edges}`
  - `load()`: 초기화 시 디스크에서 built 노드 자동 로드
  - `get_data(source_data, edges)` → `{key: data}`
- **TrainDataFlow** (DataFlow 상속): stage 빌드 기능 추가
  - `data_source`: DataWrapperProvider (train/valid 제공)
  - `set_objs(name, obj, result, info)`: 빌드 완료 후 메모리에 등록
  - `get_train(edges)`, `get_valid(edges)`: train/valid 데이터 반환
  - `get_missing_stages(pipeline)`: 미빌드 stage 목록

### Trainer (`_trainer.py`)
- 생성자: `(name, data, path, splitter, splitter_params, cache, logger, tags=None, aug_data=None)` — **Pipeline을 생성자에서 받지 않음** (Experimenter와 동일하게 pipeline 미보유, 호출마다 전달)
- `tags` (list[str]): `select_head(pipeline)`를 `nodes=None`으로 호출하면 tag가 교집합인 head만 자동 선택 (비어있으면 전체 head)
- `train_folds`: `[TrainFold]` — split별 `(TrainDataFlow, NodeStore)` 쌍, 둘은 같은 fold 디렉토리를 공유 (Experimenter의 outer_fold와 동일 구조)
- `selected_stages`, `selected_heads`: `select_head(pipeline, nodes=None)`로 설정
- `cache`: Experimenter에서 전달받은 DataCache 공유
- `select_head(pipeline, nodes=None)`: `pipeline.check_data_compatibility(self.data)` 호출 후 head 노드 지정 + upstream stage 자동 수집, 위상 순서 정렬
- `train(pipeline, n_jobs=1, gpu_id_list=None)`: serial mismatch 자동 감지 후 미빌드 노드만 대상으로 학습
- `get_status(node_name)`: pipeline 불필요 — `artifact_stores`만 조회(`train_data_flows`와 동일 디렉토리를 공유하므로 role 구분 없이도 정확함)
- `process(data, v=None)`: generator, split마다 head output을 v로 필터 후 concat하여 yield
- `to_inferencer(pipeline, v=None)`: 학습된 Processor를 추출하여 Inferencer 생성
- `reset_nodes(pipeline, nodes)`: 하위 종속 노드 포함 초기화
- 저장/로드: `save()`, `_load(path, data, cache, logger, aug_data=None)` — pipeline 인자 없음

### Inferencer (`_inferencer.py`)
- 생성자: `(pipeline, selected_stages, selected_heads, n_splits, node_objs, v=None)`
- `node_objs`: `{name: [processor_split0, processor_split1, ...]}` — Processor 리스트 (Trainer 독립)
- `process(data, agg='mean', nodes=None)`: split 결과 자동 집계
  - `agg`: `'mean'`/`'mode'`/callable/`None`(list 반환). 단일 split이면 집계 없이 반환
  - `nodes`: str/list — 출력할 head 노드 선택 (None=전체). 미등록 노드 지정 시 ValueError
- 저장/로드: `save(path)`, `load(cls, path)` — 단일 `__inferencer.pkl`에 node_objs 포함

### Connector (`_connector.py`)
- `__init__(node_query=None, edges=None, processor=None, role=None)` — 4요소 선택적 매칭
- `match(node_name, node_attrs)`: 설정된 요소만 검사, 모두 충족 시 True
  - node_query: str(regex) 또는 list(in), edges: contain 기반 매칭, processor: 일치 검사, role: 'stage'/'head' 일치 검사 (None이면 무시)

### Collector (`collector/` 패키지)
- **Collector** (`_base.py`): 기본 클래스
  - `__init__(name, connector)`, `path`는 add_collector 시 설정
  - 라이프사이클: `_start(node)`, `_collect(node, idx, inner_idx, context)`, `_end_idx(node, idx)`, `_end(node)`
  - 에러 처리: `_collect`/`_end_idx`는 safe wrapper로 try/except 래핑; `_start`/`_end`는 직접 호출 — 에러 시 `warnings` 리스트에 저장 후 warning 로그
  - `on_attach(experimenter)`: `add_collector`/`collect` 호출 시 자동 실행 — experimenter identity 비교로 중복 재계산 방지; `_on_attach(experimenter)` no-op 훅을 subclass에서 override
  - `_experimenter`: pickle 제외 (save/load 시 None으로 초기화)
  - `has(node)`: 수집 결과 보유 여부 (has_node에 위임)
  - `has_node(node)`, `reset_nodes(nodes)`, `save()`, `load(cls, path)`
  - `_get_nodes(nodes, available)`: None/list/str(regex) 패턴 매칭
  - context: `{node_attrs, processor, spec, input, output_train, output_valid}`

- **MetricCollector** (`_metric.py`): 메트릭 수집
  - `output_var`, `metric_func`, `include_train`
  - target: `context['input']['y']`, 예측값: `resolve_columns(output_valid, output_var)`
  - `_on_attach`: `metric_func`에 `on_attach`가 있으면 자동 전파
  - 저장: `push()` 오버라이드로 inner fold 결과 발생 시 즉시 `metrics.db` INSERT (per-node pkl 없음)
  - 쿼리: `get_metric(node)`, `get_metrics(nodes)`, `get_metrics_agg(nodes, inner_fold, outer_fold, include_std)`

- **ProbToLabel** (`_metric.py`): predict_proba → label 변환 후 metric 적용
  - `__init__(metric_func, var, thresholds=None)` — `metric_func`를 래핑하는 callable class
  - `var`: edges 표현법 — str(`'target'`), tuple(`(None, 'target')`), list
  - `thresholds`: None=argmax, float=binary threshold, list=multiclass per-class threshold
  - `on_attach`에서 experimenter로부터 label classes 추출 (정렬 순서 = predict_proba 열 순서)
  - binary: 2D proba `(n, 2)` 자동 처리 (col 1 추출), 1D sigmoid도 지원
  - multiclass per-class threshold: threshold 초과 클래스 중 최대 확률 선택, 없으면 argmax fallback

- **StackingCollector** (`_stacking.py`): 스태킹 데이터 수집
  - `__init__(name, connector, output_var, method='mean')` — experimenter 불필요
  - `_on_attach`에서 experimenter로부터 `_index`, `_target`(ndarray), `_target_columns`, `_data_cls` 구축
  - `output_var`, `method`(mean/mode/simple)
  - `_aggregate()`: `DataWrapper` 대신 `_data_cls`(입력 데이터 타입)의 static 메서드 사용
  - 쿼리: `get_dataset(nodes=None, include_target=True)`

- **ModelAttrCollector** (`_model_attr.py`): 모델 속성 수집 (feature_importances 등)
  - `result_key`, `adapter`(default=None, `get_adapter(connector.processor)`로 자동 설정), `params`
  - `_is_mergeable()`: self.adapter에서 직접 판단
  - 쿼리: `get_attr(node, idx)`, `get_attrs(nodes)`, `get_attrs_agg(node, agg_inner, agg_outer)`

- **SHAPCollector** (`_shap.py`): SHAP value 수집 및 분석
  - `explainer_cls`(default=shap.TreeExplainer), `data_filter`(DataFilter 인스턴스)
  - train/valid 각각 필터 적용 → SHAP 계산 → raw output 저장
  - 결과: `results[node][(idx, inner_idx)] = {'train', 'valid', 'train_index', 'valid_index', 'columns'}`
  - 분석: `get_feature_importance(node, idx)` → inner fold별 `pd.Series` 리스트
  - 분석: `get_feature_importance_agg(node, agg_inner='mean', agg_outer='mean')` → agg_inner=None이면 MultiIndex, agg_outer=None이면 DataFrame, 둘 다 설정이면 Series
  - SHAP 3D array(multiclass) 지원: `(n_samples, n_features, n_classes)` → class축 평균 후 처리

- **OutputCollector** (`_output.py`): output_train/output_valid 원본 저장
  - `output_var`, `include_target`
  - 파일 저장: `{path}/{node}/{idx}_{inner_idx}.pkl`
  - 쿼리: `get_output(node, idx, inner_idx)`, `get_outputs(node)`

- **ProcessCollector** (`_process.py`): 외부(테스트) 데이터에 대한 예측 수집
  - `__init__(name, connector, ext_data, output_var=None, method='mean')`
  - `collect`: `context['output_ext']`에서 결과 추출 → `output_var`로 컬럼 필터
  - inner fold 결과는 `method`(mean/mode/simple)로 outer fold별 집계, 파일 저장: `{path}/{node}/{idx}.pkl`
  - 쿼리: `get_output(nodes=None, agg='mean')` — nodes 필터(None/list/regex) + outer fold 집계 후 column-wise concat 반환
  - save/load 시 `ext_data`는 미저장 (런타임 전달)

## edges 구조
- dict 형태: `{key: [(node_name, var_spec), ...], ...}`
- key: 변수 집합 이름 (예: 'X', 'y', 'sample_weight')
- node_name: stage 노드명 (None=DataSource)
- var_spec: 변수 선택 (stage 노드 대상: None=전체, str, list, callable, tuple)
- 같은 key의 데이터는 column 방향으로 concat
- 상위→하위 병합: 같은 key면 extend
- **DataSource edge (`node_name=None`)**: var_spec은 반드시 컬럼명 explicit list — 아니면 `set_grp`/`set_node`가 ValueError (`_check_edges`). datasource schema가 정의돼 있으면 리스트의 모든 컬럼이 schema에 존재해야 함 — 없으면 ValueError (schema 미정의 시 이 검사는 skip). [[feedback_datasource_edges_explicit_vars]]

## exist 파라미터 (set_grp, set_node, collect)
- `'diff'` (default, set_grp/set_node): 제공된 파라미터가 기존과 다를 때만 업데이트, 동일하면 skip
- `'skip'` (collect default): 이미 존재하면 무시하고 반환
- `'error'`: 이미 존재하면 ValueError
- `'replace'`: 기존 객체를 무조건 업데이트

### set_grp 업데이트 동작 (중요)
`exist='diff'`에서 변경이 감지되면 **제공된 모든 값으로 전체 필드를 대입**한다.
`None`/빈 값은 그대로 `None`/`{}`으로 덮어쓰므로, **유지하려는 필드도 반드시 명시**해야 한다.
```python
# 잘못된 예 — processor/edges/method가 None으로 덮어써짐
exp.pipeline.set_grp('scale', params={'with_std': False})

# 올바른 예
exp.pipeline.set_grp('scale', role='stage', processor=StandardScaler,
                     method='transform', edges={'X': [(None, cols)]},
                     params={'with_std': False})
```

## Serial 무결성 추적
- 노드 정의(`set_grp`/`set_node`/`set_datasource`) 변경 시 영향받는 노드들의 `serial`이 새 UUID로 자동 교체 (`_bump_serials`)
- 아티팩트 `info.pkl`에 `node_serial` 저장
- `build()`, `exp()`, `train()` 시작 시 현재 serial vs 저장된 serial 비교 → 불일치 노드 자동 reset 후 재빌드

## Processor (`_node_processor.py`)
- **TransformProcessor**: `fit`, `fit_process`, `process`
- **PredictProcessor**: `fit`, `fit_process`, `process`
- `adapter=None` 전달 시 `DefaultAdapter()` 로 fallback
- `fit`/`fit_process`에서 y 데이터를 `squeeze()` 후 전달 (sklearn DataConversionWarning 억제)
- `get_feature_names_out` 반환값은 `list()` 로 변환하여 사용 (list/ndarray 호환)
- `process()`: `adapter.get_process_data(data)` 로 입력 타입 변환 — polars 등 라이브러리별 호환성 처리
- `data_dict` (Experimenter): `{key: ((train, train_v), valid), ...}` 형태
- `data_dict` (Trainer): `{key: (train, valid), ...}` 형태 (inner fold 없음)
- **X-less 지원**: `edges`에 `'X'`가 없고 `'y'`만 있는 경우(e.g. `LabelEncoder`) `'y'`를 primary input으로 사용
  - `fit`/`fit_process`: `'X'` 없으면 `'y'` 데이터를 squeeze하여 전달, `output_vars`를 `y_columns`로 설정
  - `process`: `X_`가 비어 있으면 입력 데이터를 squeeze 후 transform
- `y_columns`가 str인 경우(polars Series 등) `[y_columns]` 로 wrap하여 처리

## Adapter 인터페이스
- `get_params(params, logger)`: 모델 생성 파라미터
- `get_fit_params(data_dict, params, logger)`: fit 파라미터 — base: X/y를 `unwrap()` 후 반환
- `get_process_data(data)`: `process()` 입력 데이터 변환 — base: `unwrap(data)`
  - `LightGBMAdapter`: polars→pandas 변환 (LightGBM polars 미지원); `early_stopping` dict 수락 → 내부에서 `lgb_early_stopping` 콜백으로 변환 (`_params_equal`이 plain dict 비교 가능해 false rebuild 방지)
  - `CatBoostAdapter`: `_catboost_supports_polars()` (>=1.3.0) 기반 분기 — 구버전이면 polars→pandas (`get_fit_params`도 동일 적용)
- `result_objs`: `{name: (callable, mergeable_bool)}`
- `__eq__`: `type(self) is type(other) and self.__dict__ == other.__dict__` — diff 모드에서 adapter 비교에 사용
- `__hash__`: `id(self)` — set/dict 키로 사용 가능

## Sampler (`sampler/` 패키지)
- **Sampler** (`_base.py`): 기본 클래스 — `sample(fit_params) → fit_params` 인터페이스
- **ImbLearnSampler** (`_imblearn.py`): imblearn `fit_resample` 래퍼
  - `__init__(sampler)`: imblearn sampler 인스턴스 주입
  - `sample(fit_params)`: `fit_params['X']`/`['y']`로 `fit_resample` 호출 후 X, y 교체하여 반환
- 사용법: node `params`에 `mllab_sampler` 키로 Sampler 인스턴스 지정 → `_node_processor`가 fit/fit_process 전에 `sample()` 호출; estimator에 전달 전 키 제거

## 보조 모듈
- **_data_wrapper.py**: DataWrapper (wrap/unwrap/squeeze/mean/mode/simple) — pandas/polars/cudf/numpy 통합
  - `PolarsWrapper.get_columns()`: `pl.DataFrame`이면 `.columns`, `pl.Series`이면 `.name` 반환
- **_describer.py**: desc_spec, desc_status, desc_pipeline, desc_node
- **_logger.py**: BaseLogger, DefaultLogger (start/update/end_progress, adhoc_progress, rename_progress)
- **col.py**: 컬럼 선택 유틸리티
- **_connector.py**: Connector (노드 매칭)
- **collector/**: Collector, MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector
- **filter/**: DataFilter, RandomFilter(n/frac/random_state), IndexFilter(index)
- **adapter/**: sklearn, xgboost, lightgbm, catboost, keras, `_nn.py` (NNAdapter)
- **processor/**: CatConverter, CatPairCombiner, CatOOVFilter, FrequencyEncoder, ColSelector, TypeConverter, CrossFitTransformer
  - `CatPairCombiner`: pair(2) → N-way 그룹 조합으로 확장. `pairs` 요소를 N개 컬럼 인덱스/이름 그룹으로 지정 가능
  - `TypeConverter`: 모든 컬럼을 지정 타입(`str`/`int`/`float`)으로 변환. pandas: `astype`, polars: cast, numpy: `astype`. `get_feature_names_out` 지원
  - `CrossFitTransformer`: sklearn-compatible stacking meta-feature 생성기
    - `__init__(estimator, cv=5, method='predict_proba', stratified=True)`
    - `fit_transform`: CV로 OOF 예측 생성 + 전체 데이터로 full estimator fit
    - `transform`: full estimator로 예측 (fit_transform 이후)
    - 출력 컬럼명: `{estimator_class_lower}_{class}` (predict_proba) / `{estimator_class_lower}_pred` (predict)
    - Stage 노드로 사용 시 Experimenter는 OOF, Trainer/Inferencer는 full model 경로로 동작
  - polars 설치 시: PolarsLoader, ExprProcessor, PandasConverter 추가
  - `_dproc.py`: `get_type_df` (수치형만 f32/i32/i16/i8 판정), `get_type_pl`, `get_type_pd`, `merge_type_df`

## 저장 구조
Pipeline/Experimenter/Trainer는 서로의 경로를 관리하지 않음 — 각자 생성 시 `path`를 직접 받음.
```
{pipeline.path}/
  pipeline.db                       # Pipeline 노드/그룹 정의 (SQLite)

{experimenter.path}/
  __exp.pkl                         # collector_keys, tags, 메타정보 (pipeline은 미보유 — 매 호출 시 인자로 전달)
  __collector/{name}/
    __config.pkl                    # Collector 설정
    metrics.db                      # MetricCollector 결과 (node, idx, inner_idx, split, value)
    {node}.pkl                      # StackingCollector 노드별 데이터
    {node}/{idx}_{inner_idx}.pkl    # OutputCollector fold별 데이터
  __folds/{outer_idx}/{inner_idx}/{node_name}/
    obj.pkl                         # processor 객체
    result.pkl                      # fit_transform/fit_predict 출력
    info.pkl                        # {status, build_id, node_serial, fit_time, edges, ...}
    # 주의: 이 {inner_idx} 디렉토리는 Stage(train_data_flows)와 Head(artifact_stores)가
    # 공유함 — 같은 경로를 서로 다른 NodeStore 인스턴스(독립된 info 캐시)로 감싸고 있을 뿐임

{trainer.path}/
  __trainer.pkl                     # name, splitter, tags, selected_stages/heads, split_indices
  {split_idx}/{node_name}/
    obj.pkl / result.pkl / info.pkl # 마찬가지로 train_data_flows/artifact_stores가 {split_idx} 디렉토리를 공유

{inferencer_path}/
  __inferencer.pkl                  # pipeline, selected_stages/heads, n_splits, node_objs, v (단일 파일)
```

## 패키지 정보
- PyPI 패키지명: `ml-labs`, Python 패키지: `mllabs/`
- `pyproject.toml`: setuptools 기반, Python >=3.10
- optional deps: `xgboost`, `lightgbm`, `catboost`, `shap`, `polars`, `tensorflow`, `all`, `dev`
- 릴리즈: `v*` 태그 push → GitHub Actions (`publish.yml`) → 테스트(3.10/3.11/3.12) → build → PyPI 자동 배포 (OIDC)

## mllabs.nn 패키지
- `NNClassifier`, `NNRegressor`: sklearn-compatible TF/Keras 기반 추정기
  - pandas `Categorical` / polars `Categorical`/`Enum` dtype 자동 감지 → embedding 자동 생성
  - `embedding_dims`: `{col: dim}` dict로 per-column override
  - `head`: head factory 클래스 (default=`SimpleConcatHead`), `head_params`: head factory에 전달할 kwargs dict
  - `hidden`: `DenseHidden` 인스턴스 또는 dict (kwargs로 전달) 또는 None(기본값)
  - `fit(X, y, eval_set=None, callbacks=None)`: constructor callbacks + fit callbacks + early stopping 순서로 합산
  - `evals_result_`: `{'train': {metric: [...]}, 'valid': {metric: [...]}}` (history 저장)
  - Pickle: `__getstate__`/`__setstate__` — weights만 저장, `col_info_` 기반 architecture 재빌드
- 컴포넌트: `SimpleConcatHead`, `FTTransformerHead`, `DenseHidden`, `LogitOutput`, `BinaryLogitOutput`, `RegressionOutput`
  - `FTTransformerHead`: Feature Tokenizer + Transformer head
    - cat embedding → d_model projection, cont feature → per-feature learned (w, b) tokenization
    - CLS token prepend + N × FTBlock (pre-LN, MHA + FFN/GELU, residual dropout) → CLS token 반환
    - 파라미터: `d_model=192`, `n_heads=8`, `n_layers=3`, `ffn_factor=4/3`, `attention_dropout=0.2`, `ffn_dropout=0.1`, `residual_dropout=0.0`
- `NNAdapter` (`adapter/_nn.py`): eval_set 전달 + `_ProgressCallback` (epoch 진행률 로깅) + `evals_result` result_obj
