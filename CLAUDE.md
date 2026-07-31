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
- **PipelineBuilder / Pipeline** (`_pipeline.py`): 가변 빌더 + `build()`가 만드는 불변 노드 그래프
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
- **DataSource** (`_DataSourceNode`, key=`None`): 원본 데이터 스키마 및 target 정의
- **Stage**: 전처리/변환 (TransformProcessor) — 하위 노드에 데이터 공급
- **Head**: 모델링/예측 (PredictProcessor) — 최종 결과 생산

### PipelineBuilder / Pipeline 분리 (`_pipeline.py`)
```
PipelineBuilder  — 가변. grps 계층, SQLite(pipeline.db), set_grp/set_node, serial bump
  └─ .build() ──► Pipeline  — 불변 스냅샷. grp 상속 해소 완료, 순수 데이터
```
- **Experimenter/Trainer/Inferencer는 `Pipeline`만 보유** — builder를 넘기면 `TypeError`(`_run_common.require_built_pipeline`). 즉 `e.set_pipeline(p.build())`. builder를 나중에 수정해도 진행 중인 실행에 새어 들어가지 않으며, 새 정의를 반영하려면 다시 `build()` 후 `set_pipeline` 재호출
- grp는 build를 넘어가지 않음 — 구조적으로 의미 있는 건 `role`('stage'/'head')뿐이고, 원래 그룹명은 표시용 `label`로만 남음 (`get_node_attrs`의 키도 `'grp'` → `'label'`)

#### PipelineBuilder
- `VAR_TYPES = frozenset({'numerical', 'ordinal', 'nominal', 'text', 'binary', 'datetime'})`
- **`_params_equal(a, b)`**: `a == b` 한 줄 — params가 순수 데이터/ref spec만 담도록 강제되므로 `__dict__` 재귀 비교 같은 우회가 불필요해짐
- `nodes`: `{name: _PipelineNode}` (`None` → `_DataSourceNode`), `grps`: `{name: _PipelineGroup}` (`'__datasource__'` 항상 존재)
- `datasource`: `nodes[None]` 반환 property
- `set_datasource(schema, targets=None)`: DataSource 스키마/target 설정, 변경 시 downstream serial 자동 bump
- `set_grp(exist='diff'|'skip'|'error'|'replace')`, `set_node(exist=...)`, `rename_grp`, `remove_grp`, `remove_node`
  - **`processor`/`adapter`/`params` 스펙 검증** (`_validate_processor`/`_validate_adapter`/`_validate_params`) — 산 객체를 넘기면 `TypeError`. 아래 "Lazy resolution" 참조
- `build()` → `Pipeline`
- `get_node_names(query)`, `get_node_attrs(name)`, `_get_affected_nodes(nodes)`, `_find_descendants(name)`
- `_bump_serials(node_names)`: 지정 노드들의 serial을 새 UUID로 교체
- `copy()`, `copy_stage()`, `copy_nodes(node_names)` — 선택적 복사 (builder→builder)
- `compare_nodes(nodes)` → `{processor_name: DataFrame}` (params 차이 + edges['X'] stage별 변수 차이)
- `desc_pipeline(max_depth, direction)`, `desc_node(node_name, direction, show_params)`: Mermaid 다이어그램 — grp 계층이 필요하므로 **builder 전용**

#### Pipeline (빌드 결과)
- `nodes`: `{name: _BuiltNode}` — `None` 키는 `_BuiltDataSource` (builder와 동일한 관례)
- `_BuiltNode` 속성(`__slots__`): `name`, `label`, `role`, `processor`, `edges`, `method`, `adapter`, `params`, `serial`, `tag`, `desc`, `output_edges`
- `pipeline_id`(builder 신원) / `build_id`(이 빌드의 신원)
- `get_node(name)`, `get_node_attrs(name)`, `get_node_names(query=None)`
- `topo_order()`: DataSource에서 내려오는 깊이순 노드명 (DataSource 제외) — 빌드 시 1회 계산해 캐시
- `descendants(name)`, `check_data_compatibility(data)`
- `subset(node_names)`: 지정 노드 + 조상만 담은 새 Pipeline (구 `copy_nodes`, Inferencer용)
- **불변성의 한계**: `params`/`edges`는 shallow copy — 중첩 값은 builder와 공유. "수정하지 않는다"는 관례로 지킴

- **`_DataSourceNode`** (`_PipelineNode` 서브클래스):
  - `schema`: `{col: var_type}` — var_type은 VAR_TYPES 중 하나
  - `targets`: `list[str]` — 타겟 컬럼 목록 (타입과 별도)
  - `get_attrs(grps)`: role='datasource', serial, schema, targets 반환 (processor/edges/method/params 없음)

- **`_PipelineGroup`**: 노드 그룹 (stage/head 역할) — builder 내부 전용
  - 속성: `name`, `role`, `processor`, `edges`, `method`, `parent`, `adapter`, `params`, `desc`
  - `children`: 자식 그룹명 리스트, `nodes`: 소속 노드명 리스트
  - `get_attrs(grps)`: 상위 그룹 속성 병합하여 반환 (`desc`는 상속 안 됨, 각 요소 독립)
  - `diff(processor, edges, method, parent, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외 → desc-only 변경은 rebuild 미유발)

- **`_PipelineNode`**: 개별 노드 — builder 내부 전용
  - 속성: `name`, `grp`, `processor`, `edges`, `method`, `adapter`, `params`, `desc`, **`serial`** (UUID str), **`tag`** (`list[str]`)
  - `serial`: 노드 정의가 변경될 때마다 `_bump_serials`에 의해 새 UUID로 교체 → 아티팩트 무결성 추적
  - `output_edges`: 이 노드를 입력으로 사용하는 노드명 리스트
  - `get_attrs(grps)`: 그룹 속성과 노드 속성 병합 (`serial` 포함)
  - `diff(grp, processor, edges, method, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외)
  - `set_grp`/`set_node`: `desc` 파라미터 수락; exist='diff' skip 경로에서도 `desc`는 업데이트됨

- **ColSelector** (`_pipeline.py`): processor params(예: `cat_features`, `cat_cols`)에 쓰는 지연(lazy) 컬럼 선택자
  - `__init__(dsl_string='*')` — DSL 문자열 하나만 보유(정의 시점엔 데이터 불필요, `edges[key]`와 동일한 원칙)
  - **params에는 인스턴스가 아니라 ref-dict로 지정**: `{"__ref__": "mllabs.ColSelector", "__params__": {"dsl_string": "*@categorical"}}` (인스턴스는 `set_grp`/`set_node`가 `TypeError`로 거부)
  - `_node_processor`가 Processor 생성 시 `resolve_ref_values()`로 인스턴스화하고, fit 시점에 `_resolve_col_selectors`가 `eval_expr(parse(v.dsl_string), data)`로 컬럼 확정

### Experiment / Trial (`experiment/` 패키지)
Head를 Pipeline에서 떼어내 Experiment가 소유하도록 하는 방향의 1단계. **아직 `Experimenter.exp()`와 연결되지 않았고, Pipeline에도 Head가 그대로 남아 있음** — 새 레이어만 추가된 상태.

- **`Trial`** (`_trial.py`): 평가할 구성 하나 = 기존 Head 노드에 해당. `name`, `processor`, `method`, `adapter`, `params`, `edges`, `label`, `tag`
  - `get_attrs()`: `Pipeline.get_node_attrs()`와 같은 모양(`role='head'` 고정, `serial` 없음) → Connector/executor/Collector가 Head 노드와 동일하게 취급 가능
  - **신원 2중 구조**:
    - `name` — 사람이 읽는 식별자이자 디스크 아티팩트 디렉토리명 (기존 Head 노드명 역할)
    - `trial_id(pipeline)` — 정의(`content_key()`)+ **참조하는 stage 노드들의 serial**의 sha256
  - **stage serial을 해시에 넣는 이유**: Head가 Pipeline을 떠나면 `_bump_serials`가 `output_edges`를 타고 Head까지 전파하던 연쇄가 끊김. 이 해시가 그 연쇄를 대체 — 없으면 "전처리를 바꿨는데 모델이 재실행 안 되는" 조용한 오류 발생
  - `upstream_serials(pipeline)`: 직접 참조 stage만 수집 (stage 자신의 serial이 이미 상위 변경 시 bump되므로 전이적으로 커버됨)
  - `content_key()`에 `name`/`label` 제외 — 이름을 바꿔도 계산 결과는 같음. **params가 순수 데이터로 강제된 덕에** 안정적 해시가 가능 (산 객체였다면 불가)

- **`BaseExperiment`** (`_base.py`): Trial 공급원 + Collector 소유
  - Trial 계약(서브클래스 구현): `get_trial_nums()`, `get_next_trial()`, `reset()`
  - `get_trials()`: `reset()` 후 `get_trial_nums()`만큼 `get_next_trial()` 호출해 리스트로 드레인 — pull 인터페이스를 유지하되 executor의 "대상 목록 사전 확정" 전제(worker 상한, GPU/CPU 사전 분배, progress 분모)를 건드리지 않음. 적응적 제안은 나중에 이 계약 위에서 구현
  - `set_collector(name, collector, connector, path, params=None, exist='skip')` / `get_collector` / `remove_collector` / `match_collectors(trial)`
  - **Experimenter에 대한 의존성 없음** — Collector에게 저장 경로만 지정하고, fold 수/target 등 Experimenter가 필요한 부분은 Collector가 자기 `on_attach`/`_setup`으로 해결. 덕분에 데이터/split 없이도 Experiment를 만들고 들여다볼 수 있음
  - **Collector가 자기 경로를 소유** — `{experimenter.path}/__collector/` 하위가 아님. 한 데이터셋에 여러 Experiment를 붙여도 결과가 섞이지 않음

- **`SimpleExperiment`** (`_simple.py`): processor/method/adapter/edges 고정 + `param_grid` 카테시안 곱
  - `params`(전 trial 공통 고정값) + `param_grid`(`{param: [values]}`) → grid 키 정렬 기준 결정적 순서
  - trial 이름: 단일이면 `{name}`, 복수면 `{name}_{idx}` (0 패딩)
  - 생성 시 `_validate_processor`/`_validate_adapter`/`_validate_params`로 spec 검증 (Pipeline과 동일 규칙)

### Experimenter (`_experimenter.py`)
- 생성자: `(data, path, ..., cache_maxsize=4GB, logger, aug_data=None, tags=None, pipeline=None)`
- **빌드된 `Pipeline`을 보유** — `self.pipeline` (constructor `pipeline=` 또는 `set_pipeline(pipeline)`으로 설정). `build`/`exp`/`collect` 등 노드 그래프가 필요한 메소드는 `pipeline` 인자를 받지 않고 `self.pipeline` 사용
- `set_pipeline(pipeline)`: **빌드된 `Pipeline`만 수락**(builder를 넘기면 `TypeError`). 기존에 pipeline이 설정돼 있었다면 노드 `serial` mismatch를 먼저 감지해 reset 후 교체, `{path}/pipeline.pkl`에 저장. **스냅샷이므로 이후 `p.set_grp`/`p.set_node` 수정은 반영되지 않음** — 새 정의를 반영하려면 `e.set_pipeline(p.build())`를 다시 호출
- `set_status(status)`: `self.status` 설정 + meta의 status row만 갱신 (전체 meta 재저장 X). `open()`/`close()`/`close_exp()`/`reopen_exp()`가 이걸 사용
- **OS log capture** (`open_os_log`/`close_os_log`/`os_log`) — `open()`/`close()`(experiment status)와는 **무관한 별개 기능**:
  - `open_os_log(log_path=None)`: 이 프로세스의 OS-level stdout/stderr(fd 1/2)를 `{path}/__worker_logs/master.log`(기본값)로 dup2 리다이렉트 시작 — `self._os_log_state`에 원본 fd/`sys.stdout`·`stderr` 백업 보관. 이미 open이면 에러
  - `close_os_log()`: 리다이렉트 원복(`sys.stdout`/`stderr` 및 fd 1/2 복구). open 안 된 상태에서 호출하면 no-op
  - `os_log(log_path=None)`: 위 둘을 감싼 컨텍스트 매니저 — `with e.os_log(): e.build(n_jobs=1); e.exp(n_jobs=4)`
  - open~close 구간 동안: `n_jobs=1`인 `build`/`exp`는 같은 프로세스에서 돌기 때문에 마스터 리다이렉트가 그대로 캡처(별도 처리 불필요). `n_jobs>1`이면 그 구간에 한해 `log_dir`이 전달되어 워커별 리다이렉트도 같이 동작(위 `build`/`exp` 항목 참조)
  - `sys.stdout`/`stderr`는 원본 fd의 dup으로 rebind되므로, capture가 열려 있어도 `DefaultLogger`의 진행률 표시 등 Python 레벨 출력은 그대로 콘솔에 보임 — dup2로 fd 1/2만 로그 파일로 돌리기 때문에 native(C-level) 직접 write만 잡힘
- `tags` (list[str]): `exp()`를 `nodes=None`으로 호출하면 tag가 교집합인 head만 대상 (비어있으면 전체 head)
- `cache`: DataCache (LRU, 용량 기반) — `_cache.py`에 분리
- **pipeline 필요** (`self.pipeline`, `_require_pipeline()`로 미설정 시 에러):
  - `build(nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None, logger=None)` (stage), `exp(nodes=None, finalize=False, n_jobs=1, gpu_id_list=None, logger=None)` (head)
    - 시작 시 `pipeline.check_data_compatibility(self.data)` 호출 후 serial mismatch 노드 자동 감지 → `reset_nodes()` 후 재빌드
    - `n_jobs`는 실제 작업 수(`total = folds × target_nodes`)로 상한 처리됨 (`min(n_jobs, total)`) — 유휴 워커/progress bar 생성 방지 (`Trainer.train`도 동일)
    - `n_jobs > 1`이고 OS log capture가 open 상태(`open_os_log`/`os_log` 참조)일 때만 각 워커의 stdout/stderr(fd 1/2)를 `{path}/__worker_logs/worker_{i}.log`로 os.dup2 리다이렉트 → 네이티브 라이브러리 출력(TF/LightGBM/CatBoost 등) 캡처, 콘솔 오염 방지. capture가 열려있지 않으면 `log_dir=None`이 전달되어 워커는 리다이렉트를 아예 안 함(기존 기본 동작과 동일)
  - `collect(collector, nodes=None, exist='skip', logger=None)`: ad-hoc 수집 (빌드 완료된 head 노드 대상, nodes로 범위 제한 가능, progress 포함)
  - `collect_missing(collector=None, nodes=None, logger=None)`
  - `get_collect_status(collector, nodes=None)`: `{node: status}` 반환 — `'collected'`/`'not_collected'`/`'finalized'`/`'error'`
  - `reopen_exp()`: closed→open, Stage 노드 초기화 후 `build()` 재호출
  - `get_node_info()`: 노드 요약 Markdown
- **pipeline 불필요** (디스크 상태만으로 동작): `get_status(node_name)`, `finalize(nodes)`, `reinitialize(nodes)`, `close_exp()`, `reset_nodes(nodes)`, `show_error_nodes(nodes=None, traceback=False)`, `get_objs(node_name, outer_idx=0, inner_idx=0)`
  - **주의**: 같은 fold의 `train_data_flows[j]`와 `artifact_stores[j]`는 디스크상 동일 디렉토리를 가리키지만 서로 독립적인 lazy info 캐시(`NodeStore._info_cache`)를 가짐. 양쪽을 합쳐 하나의 상태로 캐싱한 뒤 한쪽만 mutate(예: finalize)하면 다른 쪽 캐시가 stale해짐 — 그래서 `get_status`는 pipeline으로 role을 판별해 해당 store 하나만 조회함(`_reset_serial_stale_nodes`도 동일 패턴). `finalize`/`reinitialize`/`reset_nodes`/`close_exp`는 store별로 조회 직후 바로 그 store에 실행(check-and-act, 캐싱된 판단을 다른 store에 넘기지 않음)하는 구조라 pipeline 없이도 안전함
- `set_collector(name, collector, connector, params=None, exist='skip')`: Collector를 **부품에서 조립하여 등록** (구 `add_collector` 대체) — pipeline 불필요, **자동 수집 안 함**. 이미 빌드된 head에서 즉시 수집하려면 별도로 `collect(collector)` 호출
  - `collector`: Collector 클래스 또는 `"module.ClassName"` 문자열 ref (`resolve_processor`)
  - `connector`: `Connector` 인스턴스 또는 `{"__ref__": ..., "__params__": {...}}` (`resolve_instance`)
  - `params`: name/connector 이후 생성자 인자 dict — 값에 `resolve_ref_values` 적용(`{__ref__}` 인스턴스화 / `{__callable__}` 참조)
  - 내부적으로 `cls(name, connector, **params)` 조립 후 등록, 조립된 인스턴스 반환. collectors 테이블에 클래스 ref 기록
- `get_collector(name)`: Collector 반환 (없으면 None)
- `remove_collector(name)`: Collector 제거 + collectors 테이블 row 삭제
- `get_worker_logs(worker=None)`: OS log capture가 open이었던 동안 캡처된 네이티브 stdout/stderr 반환 — `{worker_idx: text, 'master': text}` 또는 `worker`(int 또는 `'master'`) 지정 시 문자열. 매 실행마다 덮어씀
- `get_train_data(edges, o_idx=0, i_idx=0)` / `get_valid_data(...)` / `get_test_data(...)`, `get_node_train_data(pipeline, node, o_idx=0, i_idx=0)` / `get_node_valid_data(...)` / `get_node_test_data(...)`: 노드 출력 추출 헬퍼
- `aug_data`: 외부 데이터를 DataSource 수준에서 inner train split에 append — 미퍼시스트, create/load 시 전달
- 저장/로드: `_save()`(생성 시 1회 full meta), `load(filepath, data, data_key)` — `pipeline.pkl`이 있으면 `self.pipeline`으로 복원(단순 대입, staleness 체크 없음). 로드 후 로컬에서 새로 구성한 pipeline 객체를 다시 붙이려면 `set_pipeline(p)` 호출(staleness 체크 발생)
  - **SQLite 저장** (`ExperimenterStore`, `_experimenter_store.py`, `{path}/__exp.db`):
    - `meta` 테이블 (key→JSON): 단순 ref-직렬화 가능 값만 (`data_key, title, cache_maxsize, exp_id, tags, status`)
    - `collectors` 테이블 (name→`module.QualName` ref): 클래스 정보를 ref로 저장 → load 시 `_ref_to_obj`로 복원, **COLLECTOR_TYPES 매핑 불필요**
    - splitter 객체(`sp, sp_v, splitter_params`)는 ref-직렬화 불가라 `{path}/__splitters.pkl`에 별도 pickle

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
- 생성자: `(name, data, path, splitter, splitter_params, cache, logger, tags=None, aug_data=None, pipeline=None)` — Experimenter와 동일하게 빌드된 `Pipeline`만 보유(`set_pipeline`/constructor `pipeline=`, builder는 `TypeError`)
- `select_head` 없음 — `set_pipeline(pipeline)`이 tags 교집합 head + upstream stage를 자동 선택(`_select_from_pipeline`), staleness 체크 포함
- `tags` (list[str]): 비어있으면 전체 head 대상
- `train_folds`: `[TrainFold]` — split별 `(TrainDataFlow, NodeStore)` 쌍, 둘은 같은 fold 디렉토리를 공유 (Experimenter의 outer_fold와 동일 구조)
- `selected_stages`, `selected_heads`: `set_pipeline(pipeline)` 호출 시 자동 설정
- `cache`: Experimenter에서 전달받은 DataCache 공유
- `train(n_jobs=1, gpu_id_list=None)`: serial mismatch 자동 감지 후 미빌드 노드만 대상으로 학습
- `get_status(node_name)`: pipeline 불필요 — `artifact_stores`만 조회(`train_data_flows`와 동일 디렉토리를 공유하므로 role 구분 없이도 정확함)
- `process(data, v=None)`: generator, split마다 head output을 `v`(DSL 문자열)로 필터 후 concat하여 yield
- `to_inferencer(v=None)`: 학습된 Processor를 추출하여 Inferencer 생성
- `reset_nodes(nodes)`: 하위 종속 노드 포함 초기화
- 저장/로드: `save()`, `_load(path, data, cache, logger, aug_data=None)` — `pipeline.pkl` 있으면 복원 후 선택 재계산

### Inferencer (`_inferencer.py`)
- 생성자: `(pipeline, selected_stages, selected_heads, n_splits, node_objs, v=None)`
- `node_objs`: `{name: [processor_split0, processor_split1, ...]}` — Processor 리스트 (Trainer 독립)
- `process(data, agg='mean', nodes=None)`: split 결과 자동 집계
  - `agg`: `'mean'`/`'mode'`/callable/`None`(list 반환). 단일 split이면 집계 없이 반환
  - `nodes`: str/list — 출력할 head 노드 선택 (None=전체). 미등록 노드 지정 시 ValueError
- 저장/로드: `save(path)`, `load(cls, path)` — 단일 `__inferencer.pkl`에 node_objs 포함

### Connector (`_connector.py`)
- `__init__(node_query=None, edges=None, processor=None, role=None)` — 4요소 선택적 매칭
- `processor`: **`"module.ClassName"` 문자열만** (클래스 인스턴스 아님) — resolve 안 하고 그대로 저장
- `match(node_attrs)`: 설정된 요소만 검사, 모두 충족 시 True
  - node_query: str(regex) 또는 list(in), processor: `node_attrs['processor']`(문자열, Pipeline도 항상 문자열로 저장)와 **문자열 그대로 비교**(정규화 없음 — `set_grp`/`set_node`에 준 것과 같은 문자열 형태를 넘겨야 매칭), role: 'stage'/'head' 일치 검사 (None이면 무시)
  - edges: `{key: dsl_string}` — 각 key에 대해 노드의 resolved `edges[key]` 문자열과 **정확히 일치**해야 함 (contain 기반 아님)

### Collector (`collector/` 패키지)
- **Collector** (`_base.py`): 기본 클래스
  - `__init__(name, connector)`, `path`는 `set_collector` 시 설정
  - 라이프사이클: `_start(node)`, `_collect(node, idx, inner_idx, context)`, `_end_idx(node, idx)`, `_end(node)`
  - 에러 처리: `_collect`/`_end_idx`는 safe wrapper로 try/except 래핑; `_start`/`_end`는 직접 호출 — 에러 시 `warnings` 리스트에 저장 후 warning 로그
  - `on_attach(experimenter)`: `set_collector`/`collect` 호출 시 자동 실행 — experimenter identity 비교로 중복 재계산 방지; `_on_attach(experimenter)` no-op 훅을 subclass에서 override
  - `_experimenter`: pickle 제외 (save/load 시 None으로 초기화)
  - `has(node)`: 수집 결과 보유 여부 (has_node에 위임)
  - `has_node(node)`, `reset_nodes(nodes)`(base: `self._buf`에서 해당 노드 제거 — 서브클래스는 `super().reset_nodes(nodes)` 먼저 호출 후 자신의 disk/cache 정리), `save()`, `load(cls, path)`
  - `_get_nodes(nodes, available)`: None/list/str(regex) 패턴 매칭
  - context: `{node_attrs, processor, spec, input, output_train, output_valid}`

- **MetricCollector** (`_metric.py`): 메트릭 수집
  - `output_var`(DSL 문자열 또는 None), `metric_func`, `include_train`
  - target: `context['input']['y']`, 예측값: `eval_expr(parse(output_var), output_valid, processor=context['processor'])`로 컬럼 선택
  - `_on_attach`: `metric_func`에 `on_attach`가 있으면 자동 전파
  - 저장: `push()` 오버라이드로 inner fold 결과 발생 시 즉시 `metrics.db` INSERT (per-node pkl 없음)
  - 쿼리: `get_metric(node)`, `get_metrics(nodes)`, `get_metrics_agg(nodes, inner_fold, outer_fold, include_std)`

- **ProbToLabel** (`_metric.py`): predict_proba → label 변환 후 metric 적용
  - `__init__(metric_func, var, thresholds=None)` — `metric_func`를 래핑하는 callable class
  - `var`: DSL 문자열 (예: `'{target}'`) — `on_attach`에서 `experimenter.get_test_data({'_y': var})`로 resolve
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

## edges 구조 — DSL 문자열 (`_edge_dsl.py`)
- dict 형태: `{key: dsl_string}` — key는 변수 집합 이름(예: 'X', 'y', 'sample_weight'), 값은 **항상 순수 문자열**
- `edges[key]`는 정의/상속/직렬화/비교 어디서나 문자열 그대로 유지되며, **Processor 실행 시점에 실제 데이터를 대상으로만** lazily 컬럼 리스트로 확장됨(`eval_expr`, `_flow.py.get_data`에서 호출). `set_grp`/`set_node`는 DSL *구조*(문법 + namespace 참조)만 검증(`validate_edges`)하고 컬럼/schema는 절대 만들어보지 않음
- 문법:
  ```
  expr        := term (op term)*            -- op는 '+'/'-'/'&', 공백으로 양쪽이 분리된 독립 토큰이어야 함
  term        := ('*' | set_literal | pattern) ['@' NAME ['(' ')']]
                  | slice | namespace | '(' expr ')'
  slice       := [INT] ':' [INT]             -- 파이썬 slice, 예: '-1:' == slice(-1, None)
  set_literal := '{' [NAME (',' NAME)*] '}'  -- 명시적 컬럼명 목록
  namespace   := NAME ':' '(' expr ')'       -- NAME은 stage 노드명, 그 노드의 출력을 가리킴
  pattern     := REGEX                       -- re.match 로 컬럼명 매칭
  ```
  - `*`/`set_literal`/`pattern` 뒤에 바로(공백 없이) `@name`/`@name()`을 붙이면 `col.py`에 등록된 column-selector 적용 (예: `*@numeric`, `{a, b}@int`, `A.*@ohe_drop_first`)
  - 값을 직접 명시하지 않은 top-level(namespace 밖) 항목은 DataSource를 가리킴; `name:(...)` 블록은 그 stage 노드의 출력을 가리킴 (namespace는 top-level에서 `+`로만 결합 가능 — `-`/`&`는 namespace 내부/괄호 안에서만)
  - **DataSource 참조는 반드시 명시적 컬럼명 리스트(`{a, b}`)** — 패턴/callable 아님. [[feedback_datasource_edges_explicit_vars]]
- 그룹/노드 상속: 자기 값이 `+`/`-`로 시작하면 부모의 이미 resolve된 문자열에 이어붙임(`f"{parent} {own}"`); 그 외 일반 문자열은 완전히 override(상속 안 함). 자기 값이 없으면(`{}`) 부모 값을 그대로 상속
- 같은 key의 여러 segment(`+`로 연결)는 column 방향으로 concat됨

## Edge DSL 관련 함수 (`_edge_dsl.py`)
- `parse(dsl_string)` → AST (`Star`/`SetLiteral`/`Pattern`/`Namespace`/`BinOp`/`slice`)
- `eval_expr(node, data, processor=None)`: AST를 실제 `data`(`DataWrapper`, `get_columns()`/`select_by_dtype()` 노출)에 대해 평가 → 컬럼명 리스트. `data`에서 `columns = data.get_columns()`를 내부적으로 유도하므로 호출부는 컬럼 리스트를 따로 넘기지 않음
- `validate_edges(dsl_string, pipeline)`: 구조만 검증(문법 + namespace가 존재하는 stage 노드를 가리키는지) — 컬럼/schema는 절대 건드리지 않음
- `iter_segments(dsl_string)` → `(node_name, expr)` 이터레이터 (top-level `+` 체인 분해)
- `referenced_nodes(dsl_string)` → 참조하는 노드명 집합 (`None`=DataSource 포함)
- `unparse(node)` → AST를 다시 DSL 문자열로 렌더링

## col.py — `@name` column-selector 레지스트리
- `col_selector(*processor_classes, name=None)` 데코레이터로 등록. 모든 selector는 동일 시그니처 `(data, processor=None) -> mask` — `data`는 이미 패턴 등으로 좁혀진 후보 컬럼만 담은 `DataWrapper`
- `processor_classes`가 지정되면 해당 processor 타입에서만 유효(`resolve_selector`가 불일치 시 ValueError); 비워두면(`()`) processor 없이도 사용 가능
- `name=`으로 등록 키를 함수명과 다르게 지정 가능 (파이썬 builtin과 겹치는 `float`/`int`/`string` 등에 사용)
- **`ohe_drop_first`** (OneHotEncoder 전용), **`subset_poly`** (PolynomialFeatures 전용, degree/interaction/bias 일관된 전체 조합으로 스냅)
- **dtype 기반 builtin selector** (processor 불필요, `data.select_by_dtype(kind)` 사용): `@numeric`, `@categorical`, `@binary`(bool dtype만), `@float`, `@int`, `@string`
  - DataSource 최상위(`*@numeric` 등)에 바로 걸면 schema에 없는 raw 컬럼(id/target/sample_weight 등)까지 포함될 수 있어 위험 — 이미 확정된 stage 노드 출력 namespace 안에서 쓰는 게 안전

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
p.set_grp('scale', params={'with_std': False})

# 올바른 예
p.set_grp('scale', role='stage', processor=StandardScaler,
          method='transform', edges={'X': '{' + ', '.join(cols) + '}'},
          params={'with_std': False})
```
[[feedback_pipeline_direct_reference]]: `p`가 스코프에 있으면 `p.set_grp`/`p.set_node`를 직접 호출 (`e.pipeline.set_grp`처럼 우회하지 않음)

## Serial 무결성 추적
- 노드 정의(`set_grp`/`set_node`/`set_datasource`) 변경 시 영향받는 노드들의 `serial`이 새 UUID로 자동 교체 (`_bump_serials`)
- 아티팩트 `info.pkl`에 `node_serial` 저장
- `build()`, `exp()`, `train()` 시작 시 현재 serial vs 저장된 serial 비교 → 불일치 노드 자동 reset 후 재빌드

## Lazy resolution: processor / adapter / params (edges DSL과 동일한 지연 원칙)
- `set_grp`/`set_node`는 `processor`/`adapter`/`params`를 **스펙 형태만 검증하고 절대 resolve/instantiate하지 않음** — 실제 값으로의 변환은 전부 사용 시점(`_node_processor.py`)으로 미룸
- **산 객체는 정의 시점에 `TypeError`로 거부됨** (`_validate_processor`/`_validate_adapter`/`_validate_params`, `_pipeline.py`). 에러 메시지가 써야 할 ref 형태를 그대로 안내함
- **processor**: `"module.ClassName"` **문자열만** (클래스 객체 거부). 실제 클래스로 resolve되는 유일한 지점은 `_node_processor.py`(`TransformProcessor`/`PredictProcessor.__init__`의 `resolve_processor()` 호출). `Connector`, `_describer.py`, `_executor.py._needs_gpu`는 전부 이 문자열을 그대로 다룸
  - **클래스를 허용하면 안 되는 이유**: `Connector.match`는 `node_attrs['processor']`를 **문자열 그대로 비교**하므로, 클래스로 정의된 노드는 문자열 ref로 설정한 Connector와 **영영 매칭되지 않음** — 에러 없이 collector가 조용히 아무것도 수집하지 않음. `serialize_value`가 클래스를 `{"__type__":"class"}`로 저장하고 리로드 시 클래스로 되돌리므로 재시작해도 해소되지 않음
- **adapter**: `None` / `"module.ClassName"` 문자열 / `{"__ref__":...,"__params__":{...}}` dict — 인스턴스 거부. `resolve_node_adapter(processor, adapter_spec)`(`adapter/__init__.py`)가 사용 시점에 resolve
- **params**: 순수 데이터(스칼라/numpy 스칼라/list/tuple/dict)와 ref spec만 허용 — `{'__ref__':...}`(예: ColSelector, `mllab_sampler`)/`{'__callable__':...}`(예: metric 함수). 중첩 값까지 재귀 검증하며 에러가 경로(`params['a']['b'][0]`)를 표시함. `_node_processor.py`의 `_resolve_params()`가 Processor 생성 시점에 `resolve_ref_values()`로 해제
- **왜**: (1) `mllabs.nn.NNClassifier`처럼 무거운 import(TensorFlow)를 유발하는 processor/adapter가 파이프라인 "정의" 시점에 실수로 로드되는 걸 방지 — 실제 `build`/`exp` 실행 시점까지 미뤄짐. (2) 파이프라인 전체가 직렬화 가능해짐(declarative config 방향). (3) diff 비교가 raw spec 비교라 `_params_equal`이 `==` 한 줄로 축소됨 — 인스턴스 `__eq__` 신뢰성 이슈 자체가 사라짐
- **테스트**: `tests/mock.py`에 여러 테스트 파일에 흩어져 있던 더미 processor 클래스 중 string-ref로 참조돼야 하는 것들을 모아둠 — `tests/`엔 `__init__.py`가 없어 pytest가 bare module(`import mock`)로 수집하므로 `processor='mock.DummyStage'`식으로 참조 가능

## Processor (`_node_processor.py`)
- **TransformProcessor**: `fit`, `fit_process`, `process`
- **PredictProcessor**: `fit`, `fit_process`, `process`
- **processor(=transformer/estimator)/adapter/params가 실제 값으로 resolve되는 유일한 지점** — Pipeline은 이 셋을 절대 resolve 안 하고 spec 그대로 넘김(아래 "Lazy resolution" 섹션 참조)
  - `__init__`: `self.transformer`/`self.estimator = resolve_processor(transformer/estimator)`(`"module.ClassName"` 문자열 → 클래스, `_serialize.py`) — 이 라인이 processor가 클래스가 되는 유일한 곳
  - `self.adapter = resolve_node_adapter(transformer/estimator, adapter)` — **resolve 전의 raw(문자열) processor**를 넘김(resolve된 클래스가 아님 — `get_adapter`가 문자열/클래스 둘 다 처리하므로 순서 무관하지만, "processor는 인스턴스 생성 시점에만 클래스로" 원칙에 맞춰 문자열째로 전달)
  - `self.params = _resolve_params(params)` — `params` 내 `{'__ref__':...}`/`{'__callable__':...}` 항목을 `resolve_ref_values()`로 해제(ColSelector 인스턴스화 등). `mllab_sampler` 값도 여기서 같이 resolve됨
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
  - `LightGBMAdapter`: polars→pandas 변환 (LightGBM polars 미지원); `early_stopping` dict 수락 → 내부에서 `lgb_early_stopping` 콜백으로 변환 (params에 콜백 인스턴스를 넣을 수 없으므로 이 dict 형태가 유일한 지정 방법)
  - `CatBoostAdapter`: `_catboost_supports_polars()` (>=1.3.0) 기반 분기 — 구버전이면 polars→pandas (`get_fit_params`도 동일 적용)
- `result_objs`: `{name: (callable, mergeable_bool)}`
- `__eq__`: `type(self) is type(other) and self.__dict__ == other.__dict__`
- `__hash__`: `id(self)` — set/dict 키로 사용 가능
- **adapter 지정 방식** (`set_grp`/`set_node`의 `adapter=`): `"module.ClassName"` 문자열 / `{"__ref__": ..., "__params__": {...}}` / `None`만 허용(**인스턴스는 `TypeError`**) — **저장 시점엔 resolve 안 함**, `_node_processor.py`가 인스턴스 생성 시 `resolve_node_adapter(processor, adapter)`로 resolve(`adapter.resolve_node_adapter`, `adapter/__init__.py`)
  - `resolve_node_adapter(processor, adapter_spec)`: `adapter_spec` 있으면 `resolve_instance(adapter_spec)`, 없으면 `get_adapter(processor)`(processor 클래스명 기반 디폴트 — `get_adapter`는 문자열이면 `rpartition('.')[-1]`로 bare/`"module.ClassName"` 둘 다 처리, 클래스/인스턴스면 `.__name__`/`.__class__.__name__`)
  - `_executor.py._needs_gpu`(멀티 워커 GPU 디스패치 스케줄링)도 이 함수로 resolve하되, 노드별 결과를 `_gpu_cache`(node name → bool)에 캐싱해 매 dispatch tick 재계산 방지
- **레지스트리** (`adapter/__init__.py`): `MODEL_ADAPTERS`(모델명→인스턴스), `get_adapter(model_or_name)`. `NNAdapter`는 TF를 top-level import하므로 **지연 로드** — `_LAZY_ADAPTERS`(`NNClassifier`/`NNRegressor`)로 first-use 시 인스턴스화·캐시, 모듈 `__getattr__`로 `NNAdapter` 심볼 노출 → `import mllabs`가 TF를 끌어오지 않음

## Sampler (`sampler/` 패키지)
- **Sampler** (`_base.py`): 기본 클래스 — `sample(fit_params) → fit_params` 인터페이스
- **ImbLearnSampler** (`_imblearn.py`): imblearn `fit_resample` 래퍼
  - `__init__(sampler)`: imblearn sampler 인스턴스 주입
  - `sample(fit_params)`: `fit_params['X']`/`['y']`로 `fit_resample` 호출 후 X, y 교체하여 반환
- 사용법: node `params`에 `mllab_sampler` 키로 Sampler 인스턴스 지정 → `_node_processor`가 fit/fit_process 전에 `sample()` 호출; estimator에 전달 전 키 제거

## 보조 모듈
- **_data_wrapper.py**: DataWrapper (wrap/unwrap/squeeze/mean/mode/simple) — pandas/polars/cudf/numpy 통합
  - `PolarsWrapper.get_columns()`: `pl.DataFrame`이면 `.columns`, `pl.Series`이면 `.name` 반환
  - `select_by_dtype(kind)`: `'category'|'numeric'|'int'|'float'|'str'|'bool'`에 해당하는 컬럼명(numpy는 정수 offset) 리스트 반환 — `col.py`의 `@numeric` 등 dtype selector가 사용하는 primitive. (예전 `get_column_list(ColSelector(col_type=, pattern=))`는 제거됨 — pattern 부분은 이제 DSL의 `Pattern` 노드가 담당)
- **_edge_dsl.py**: edges DSL 파서/평가기 — 위 "Edge DSL" 섹션 참조
- **_serialize.py**: ref 기반 직렬화/해석
  - `serialize_value`/`deserialize_value` (JSON 왕복), `_obj_to_ref`/`_ref_to_obj`
  - `resolve_processor(x)`: `"module.ClassName"` str → 클래스, else passthrough
  - `resolve_instance(spec)`: str→인스턴스(기본값) / `{__ref__, __params__}`→`cls(**params)` / else passthrough. `resolve_adapter`가 위임
  - `resolve_ref_values(value)`: params 값 재귀 해석 — `{"__callable__": "mod.fn"}`→**호출 안 하고** 그 객체 참조(metric_func 등), `{"__ref__": ..., "__params__": {...}}`→인스턴스화, 문자열/스칼라는 그대로. `set_grp`/`set_node`/`set_collector`의 params에 적용
- **_experimenter_store.py**: `ExperimenterStore` — Experimenter의 `__exp.db`(meta/collectors 테이블) SQLite 저장
- **_executor.py**: 빌드/실험 실제 실행 — `_build_flow_single/multi`, `_experiment_single/multi`, `ProcessWorker`(spawn). 멀티워커 시 fd 리다이렉트로 네이티브 출력을 `__worker_logs/`에 캡처, fit/predict 경고는 `catch_warnings`로 잡아 logger 채널로 forward(node prefix)
- **_tracker.py**: `LoggerExecuteTracker` — 워커 이벤트→logger. 메시지 `typ`에 따라 `logger.info`/`logger.warning` 라우팅(경고는 verbosity로 게이팅 + `warning_list` 수집)
- **_describer.py**: desc_spec, desc_status, desc_pipeline, desc_node
- **_logger.py**: BaseLogger, DefaultLogger (start/update/end_progress, adhoc_progress, rename_progress)
- **col.py**: `@name` column-selector 레지스트리 — 위 "col.py" 섹션 참조
- **_connector.py**: Connector (노드 매칭)
- **collector/**: Collector, MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector
- **filter/**: DataFilter, RandomFilter(n/frac/random_state), IndexFilter(index)
- **adapter/**: sklearn, xgboost, lightgbm, catboost, keras, `_nn.py` (NNAdapter)
- **processor/**: CatConverter, CatPairCombiner, CatOOVFilter, FrequencyEncoder, TypeConverter, CrossFitTransformer (`ColSelector`는 `_pipeline.py`에 있음 — processor/ 소속 아님)
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
  pipeline.db                       # PipelineBuilder 노드/그룹 정의 (SQLite)

{experimenter.path}/
  __exp.db                          # SQLite: meta(단순값) + collectors(클래스 ref) 테이블
  __splitters.pkl                   # sp, sp_v, splitter_params (ref-직렬화 불가라 pickle)
  pipeline.pkl                      # set_pipeline()으로 저장된 빌드 결과 Pipeline (있으면 load() 시 self.pipeline으로 복원)
  __worker_logs/worker_{i}.log      # 멀티워커 실행이 캡처한 네이티브 stdout/stderr (get_worker_logs())
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
  pipeline.pkl                      # set_pipeline()으로 저장된 빌드 결과 Pipeline
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
